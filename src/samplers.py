"""
Diffusion sampler interface + adapters.

All downstream metrics (frontier, judge-PPL, per-position entropy) consume a
`DiffusionSampler`. We standardize the interface so that swapping LLaDA for
SEDD for CANDI is a config change, not a code change.

Model-specific code lives in adapters. LLaDA and SEDD have non-HF APIs; rather
than paper over them with brittle wrappers, we expose clearly-marked
integration points (`TODO: INTEGRATE ...`) where you paste in your existing
inference code. The sampler contract is well-defined enough that the glue is
minimal.

Interface:
    sample(n_sequences, seq_length, nfe, temperature, seed) -> list[list[int]]
    logits_at(token_ids, positions, noise_level) -> Tensor   # for diagnostics
    attention_maps(token_ids) -> list[Tensor]                # optional
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------
@dataclass
class SamplerConfig:
    model_name: str                   # "llada-8b", "sedd-medium", "candi-owt"
    checkpoint_path: str
    tokenizer_name: str               # key into tokenizers_bench.TOKENIZER_REGISTRY
    device: str = "cuda"
    dtype: str = "bfloat16"


class DiffusionSampler(ABC):
    """Abstract base class for diffusion LM samplers."""

    def __init__(self, config: SamplerConfig) -> None:
        self.config = config
        self.device = config.device
        self.dtype = getattr(torch, config.dtype)

    @abstractmethod
    def sample(
        self,
        n_sequences: int,
        seq_length: int,
        nfe: int,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> list[list[int]]:
        """Unconditional generation. Returns a list of token id sequences."""

    @abstractmethod
    def logits_at(
        self,
        token_ids: torch.LongTensor,    # [batch, seq_length]
        noise_level: float,             # t or σ depending on sampler
        mask: Optional[torch.BoolTensor] = None,  # True where noised
    ) -> torch.Tensor:                  # [batch, seq_length, vocab]
        """
        Model's predicted logits P(x_0 | x_t) at every position.
        Used for per-position entropy diagnostics.
        """

    def attention_maps(
        self,
        token_ids: torch.LongTensor,
        noise_level: float,
    ) -> Optional[list[torch.Tensor]]:
        """
        Return per-layer attention maps (list of [batch, heads, seq, seq]).
        Default: not supported; override in adapters that can capture them.
        """
        return None


# ===========================================================================
# LLaDA adapter
# ===========================================================================
class LLaDASampler(DiffusionSampler):
    """
    LLaDA follows a masked-diffusion paradigm with an HF-compatible checkpoint
    (GSAI-ML/LLaDA-8B-Base). It uses a LLaMA tokenizer and trust_remote_code=True
    to load the custom modeling class.

    The official repo exposes a `generate` method on the model. We wrap it here
    and add temperature control at the per-step denoising distribution.
    """

    def __init__(self, config: SamplerConfig) -> None:
        super().__init__(config)
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(
            config.checkpoint_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(
            config.checkpoint_path, trust_remote_code=True
        )
        # LLaDA's mask token id — verify against your checkpoint config
        self.mask_id = getattr(self.model.config, "mask_token_id", 126336)

    @torch.no_grad()
    def sample(
        self,
        n_sequences: int,
        seq_length: int,
        nfe: int,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> list[list[int]]:
        torch.manual_seed(seed)
        # Start from all-mask
        x = torch.full(
            (n_sequences, seq_length), self.mask_id,
            dtype=torch.long, device=self.device,
        )
        # Standard LLaDA denoising loop: pick a fraction of masked positions to
        # unmask each step, using low-confidence remasking from the paper.
        ts = torch.linspace(1.0, 0.0, nfe + 1, device=self.device)
        for step in range(nfe):
            t, s = ts[step].item(), ts[step + 1].item()
            mask = (x == self.mask_id)
            if not mask.any():
                break
            logits = self.model(x).logits       # [B, L, V]
            # Temperature scaling
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            # Sample predicted token at each position
            pred = torch.distributions.Categorical(probs=probs).sample()
            # Fraction to unmask this step (linear schedule)
            n_mask_curr = mask.sum(dim=-1)
            keep_frac = 1.0 - (s / max(t, 1e-6))
            # Low-confidence remasking: keep highest-confidence positions
            conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            conf = conf.masked_fill(~mask, -1.0)
            n_to_fill = (keep_frac * n_mask_curr.float()).long()
            for b in range(n_sequences):
                k = int(n_to_fill[b].item())
                if k <= 0:
                    continue
                _, idx = conf[b].topk(k)
                x[b, idx] = pred[b, idx]
        return x.cpu().tolist()

    @torch.no_grad()
    def logits_at(
        self,
        token_ids: torch.LongTensor,
        noise_level: float,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        x = token_ids.to(self.device)
        if mask is not None:
            x = x.clone()
            x[mask.to(self.device)] = self.mask_id
        return self.model(x).logits


# ===========================================================================
# SEDD adapter
# ===========================================================================
class SEDDSampler(DiffusionSampler):
    """
    SEDD (Lou et al. 2024) uses a custom repo layout (louaaron/Score-Entropy-
    Discrete-Diffusion) with `Graph`, `Noise`, and `model.SEDD` abstractions
    not expressible as HF AutoModel.

    INTEGRATION POINT: paste in the SEDD `sampling.get_pc_sampler` call from
    your existing eval pipeline. The commented skeleton below matches the
    upstream API.
    """

    def __init__(self, config: SamplerConfig) -> None:
        super().__init__(config)
        # TODO: INTEGRATE SEDD. Example:
        #
        # from sedd_repo import model as sedd_model, graph_lib, noise_lib, sampling
        # self.net = sedd_model.SEDD.from_pretrained(config.checkpoint_path).to(self.device)
        # self.graph = graph_lib.get_graph(self.cfg)   # "absorb" or "uniform"
        # self.noise = noise_lib.get_noise(self.cfg)
        #
        # self._sampling_fn = lambda batch_size, steps, temp: sampling.get_pc_sampler(
        #     graph=self.graph, noise=self.noise, batch_dims=(batch_size, seq_length),
        #     predictor="analytic", steps=steps, denoise=True,
        # )(self.net)
        self._initialized = False

    def _require_init(self) -> None:
        if not self._initialized:
            raise NotImplementedError(
                "SEDDSampler: paste your SEDD loading code into "
                "src/samplers.py::SEDDSampler.__init__. See TODO marker. "
                "The interface contract is: self._sampling_fn(batch_size, steps, "
                "temperature) -> LongTensor[batch, seq_length]."
            )

    @torch.no_grad()
    def sample(
        self,
        n_sequences: int,
        seq_length: int,
        nfe: int,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> list[list[int]]:
        self._require_init()
        torch.manual_seed(seed)
        out = self._sampling_fn(n_sequences, nfe, temperature)  # type: ignore
        return out.cpu().tolist()

    @torch.no_grad()
    def logits_at(
        self,
        token_ids: torch.LongTensor,
        noise_level: float,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        self._require_init()
        # SEDD's score network returns scores, not logits directly. Convert via
        # the graph's reverse-rate formulation. TODO: wire up.
        raise NotImplementedError("SEDD logits_at: use graph.reverse_rate helpers.")


# ===========================================================================
# CANDI adapter
# ===========================================================================
class CANDISampler(DiffusionSampler):
    """
    CANDI (Pynadath et al. 2026) hybrid discrete+continuous sampler.
    Code at https://github.com/patrickpynadath1/candi-lander (as referenced in
    the paper). The sampler materializes one-hot vectors only for the initial
    prior and otherwise uses the embedding-lookup approximation from §5.3.

    INTEGRATION POINT: replace the `_step` method with the repo's `hybrid_sample`
    or equivalent. Interface contract preserved.
    """

    def __init__(self, config: SamplerConfig) -> None:
        super().__init__(config)
        # TODO: INTEGRATE CANDI. Example:
        #
        # from candi_repo import CANDIModel, load_candi_ckpt
        # self.model = load_candi_ckpt(config.checkpoint_path).to(self.device).eval()
        #
        # self.vocab_size = self.model.config.vocab_size
        # self.mask_id = self.model.config.mask_token_id
        # self.r_min, self.r_max = self.model.config.r_range   # from training
        self._initialized = False

    def _require_init(self) -> None:
        if not self._initialized:
            raise NotImplementedError(
                "CANDISampler: paste candi-lander loading code into __init__. "
                "Sampler interface expects _step(x, mask, t, s, temperature)."
            )

    @torch.no_grad()
    def sample(
        self,
        n_sequences: int,
        seq_length: int,
        nfe: int,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> list[list[int]]:
        self._require_init()
        torch.manual_seed(seed)

        V = self.vocab_size  # type: ignore[attr-defined]
        mask_id = self.mask_id  # type: ignore[attr-defined]

        # Initial state: all masked. Continuous component initialized from
        # prior at t=1 once any positions need it.
        x = torch.full((n_sequences, seq_length), mask_id,
                       dtype=torch.long, device=self.device)
        m = torch.zeros((n_sequences, seq_length), dtype=torch.bool, device=self.device)

        ts = torch.linspace(1.0, 0.0, nfe + 1, device=self.device)
        for step in range(nfe):
            t, s = ts[step].item(), ts[step + 1].item()
            # TODO: call self._step(x, m, t, s, temperature)  from candi-lander
            # which applies Eq. 15: x_s = Mt * x_t + ¬Mt * M'_s * x'_s + ¬Mt * ¬M'_s * x''_s
            raise NotImplementedError
        return x.cpu().tolist()

    @torch.no_grad()
    def logits_at(
        self,
        token_ids: torch.LongTensor,
        noise_level: float,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        self._require_init()
        # CANDI's model predicts P(x_0 | x_t); noise_level is the current t.
        # TODO: wire up.
        raise NotImplementedError


# ===========================================================================
# Reference MDLM sampler (fully working — baseline for sanity checks)
# ===========================================================================
class MDLMSampler(DiffusionSampler):
    """
    Minimal masked-diffusion sampler using a pre-trained HF model.

    Kept fully working so that the rest of the harness (metrics, frontier
    sweeps, length-gen) can be validated before LLaDA/SEDD/CANDI are wired up.
    Assumes the checkpoint exposes a standard causal/HF logits interface with
    a dedicated mask token in its vocab.
    """

    def __init__(self, config: SamplerConfig) -> None:
        super().__init__(config)
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.model = AutoModelForMaskedLM.from_pretrained(
            config.checkpoint_path, torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(config.checkpoint_path)
        self.mask_id = self.tok.mask_token_id

    @torch.no_grad()
    def sample(
        self,
        n_sequences: int,
        seq_length: int,
        nfe: int,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> list[list[int]]:
        torch.manual_seed(seed)
        x = torch.full((n_sequences, seq_length), self.mask_id,
                       dtype=torch.long, device=self.device)
        ts = torch.linspace(1.0, 0.0, nfe + 1, device=self.device)
        for step in range(nfe):
            t, s = ts[step].item(), ts[step + 1].item()
            mask = (x == self.mask_id)
            if not mask.any():
                break
            logits = self.model(x).logits / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            pred = torch.distributions.Categorical(probs=probs).sample()
            # Unmask a Bernoulli fraction (1 - s/t)
            keep = (s / max(t, 1e-6))
            unmask = (torch.rand_like(x, dtype=torch.float) > keep) & mask
            x = torch.where(unmask, pred, x)
        # Any residual masks: sample argmax
        if (x == self.mask_id).any():
            logits = self.model(x).logits
            pred = logits.argmax(dim=-1)
            x = torch.where(x == self.mask_id, pred, x)
        return x.cpu().tolist()

    @torch.no_grad()
    def logits_at(
        self,
        token_ids: torch.LongTensor,
        noise_level: float,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        x = token_ids.to(self.device)
        if mask is not None:
            x = x.clone()
            x[mask.to(self.device)] = self.mask_id
        return self.model(x).logits

    def attention_maps(
        self, token_ids: torch.LongTensor, noise_level: float,
    ) -> Optional[list[torch.Tensor]]:
        x = token_ids.to(self.device)
        out = self.model(x, output_attentions=True)
        return list(out.attentions) if out.attentions is not None else None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_SAMPLER_REGISTRY: dict[str, type[DiffusionSampler]] = {
    "mdlm": MDLMSampler,
    "llada": LLaDASampler,
    "sedd": SEDDSampler,
    "candi": CANDISampler,
}


def build_sampler(kind: str, config: SamplerConfig) -> DiffusionSampler:
    if kind not in _SAMPLER_REGISTRY:
        raise KeyError(
            f"Unknown sampler '{kind}'. Known: {sorted(_SAMPLER_REGISTRY)}"
        )
    return _SAMPLER_REGISTRY[kind](config)

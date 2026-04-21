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
    logits_at(token_ids, noise_level, mask) -> Tensor   # for diagnostics
    attention_maps(token_ids, noise_level) -> list[Tensor] | None
    tokenizer -> PreTrainedTokenizerBase                 # property

The sampler owns its tokenizer (via the `tokenizer` property). Downstream code
(JudgePerplexity, text8_word_frontier, length_gen) should consume it from the
sampler rather than re-looking-up via TOKENIZER_REGISTRY, because samplers
with non-standard tokenizers (e.g. LLaDA-8B-Base's ~126K vocab) would get
silently decoded with the wrong tokenizer otherwise.

Noise-level contract:
    `logits_at(x, noise_level=t, mask=m)` must return P(x_0 | x_t) at the
    specified t. Absorbing-discrete methods (MDLM, LLaDA) can infer t from the
    mask pattern and may ignore `noise_level`; score-based methods (SEDD) and
    hybrid methods (CANDI) MUST honor it. `_internal_perplexity` passes t=0.0
    explicitly; new adapters must plumb that through.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


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

        Absorbing-diffusion adapters may ignore `noise_level` (the mask pattern
        carries t implicitly); score-based / hybrid adapters must honor it.
        """

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The tokenizer that produced the ids this sampler operates over.

        Downstream code (JudgePerplexity.perplexity_from_token_ids,
        text8_word_frontier) consumes this directly so that samplers with
        non-registry tokenizers (e.g. LLaDA's 126K vocab) work without silently
        falling back to a registry entry with a different vocab.
        """

    def attention_maps(
        self,
        token_ids: torch.LongTensor,
        noise_level: float,
    ) -> Optional[list[torch.Tensor]]:
        """
        Return per-layer attention maps (list of [batch, heads, seq, seq]).
        Default: not supported; override in adapters that can capture them.
        Weights must sum to 1 across the key dimension (standard softmax
        attention); if your backend returns pre-softmax scores, renormalize
        before returning.
        """
        return None


# ===========================================================================
# LLaDA adapter
# ===========================================================================
class LLaDASampler(DiffusionSampler):
    """
    LLaDA follows a masked-diffusion paradigm with an HF-compatible checkpoint
    (GSAI-ML/LLaDA-8B-Base). It uses a LLaMA-derived tokenizer (~126K vocab)
    and trust_remote_code=True to load the custom modeling class. We use
    AutoModel rather than AutoModelForMaskedLM because LLaDA's custom class
    isn't registered under the MaskedLM auto-mapping; the forward returns a
    `.logits` attribute regardless.

    The official repo exposes a `generate` method on the model. We wrap it here
    with a low-confidence remasking loop and add temperature control at the
    per-step denoising distribution.
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

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tok

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
        noise_level: float,                # absorbing; mask carries t
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
import sys
import os
import torch
import torch.nn.functional as F
from typing import Optional

class SEDDSampler(DiffusionSampler):
    """
    SEDD (Lou et al. 2024) adapter utilizing the louaaron/Score-Entropy-Discrete-Diffusion repo.
    """

    def __init__(self, config: SamplerConfig) -> None:
        super().__init__(config)
        
        # Dynamically add the SEDD repo path to sys.path
        sedd_repo_path = "/home/aneek/src/dLLM-eval/aneek/Score-Entropy-Discrete-Diffusion"
        if sedd_repo_path not in sys.path:
            sys.path.insert(0, sedd_repo_path)
            
        try:
            from model import SEDD
            from graph_lib import get_graph
            from noise_lib import get_noise
            from sampling import get_pc_sampler
        except ImportError as e:
            raise RuntimeError(
                f"Could not import SEDD modules from {sedd_repo_path}. "
                "Ensure the path is correct and contains the required python files."
            ) from e

        print(f"[sampler] loading SEDD model from {config.checkpoint_path}")
        # Load the SEDD model from the local checkpoint
        self.net = SEDD.from_pretrained(config.checkpoint_path).to(self.device).eval()
        
        # Extract graph and noise configurations from the loaded model
        self.cfg = self.net.config
        self.graph = get_graph(self.cfg)
        self.noise = get_noise(self.cfg)
        self.get_pc_sampler = get_pc_sampler
        
        self._initialized = True

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
        
        # Instantiate the sampler function for the requested batch size and NFE
        sampling_fn = self.get_pc_sampler(
            graph=self.graph,
            noise=self.noise,
            batch_dims=(n_sequences, seq_length),
            predictor="analytic",
            steps=nfe,
            denoise=True,
        )
        
        # Execute the reverse diffusion process
        out = sampling_fn(self.net)
        return out.cpu().tolist()

    @torch.no_grad()
    def logits_at(
        self,
        token_ids: torch.LongTensor,
        noise_level: float,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        # SEDD models predict scores, not standard HF causal logits.
        # Translating SEDD scores to raw logits requires reverse-rate conversions 
        # from the graph library. 
        raise NotImplementedError("SEDD logits_at requires graph reverse-rate implementation for diagnostic entropy.")

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

    IMPORTANT: `logits_at` MUST honor `noise_level` — CANDI's predictor is
    time-conditioned and uses t to gate the discrete vs continuous update
    contributions (Eq. 14–15).
    """

    def __init__(self, config: SamplerConfig) -> None:
        super().__init__(config)
        from .tokenizers_bench import load_tokenizer
        self.tok = load_tokenizer(config.tokenizer_name)

        # TODO: INTEGRATE CANDI. Example:
        #
        # from candi_repo import CANDIModel, load_candi_ckpt
        # self.model = load_candi_ckpt(config.checkpoint_path).to(self.device).eval()
        # self.vocab_size = self.model.config.vocab_size
        # self.mask_id = self.model.config.mask_token_id
        # self.r_min, self.r_max = self.model.config.r_range   # from training
        self._initialized = False

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tok

    def _require_init(self) -> None:
        if not self._initialized:
            raise NotImplementedError(
                "CANDISampler: paste candi-lander loading code into __init__. "
                "Sampler interface expects _step(x, mask, t, s, temperature), "
                "and `logits_at` must honor the `noise_level` argument."
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
        # Once _step is wired up, the reverse-time loop below applies:
        #   torch.manual_seed(seed)
        #   V, mask_id = self.vocab_size, self.mask_id
        #   x = torch.full((n_sequences, seq_length), mask_id,
        #                  dtype=torch.long, device=self.device)
        #   m = torch.zeros_like(x, dtype=torch.bool)
        #   ts = torch.linspace(1.0, 0.0, nfe + 1, device=self.device)
        #   for step in range(nfe):
        #       t, s = ts[step].item(), ts[step + 1].item()
        #       x, m = self._step(x, m, t, s, temperature)  # Eq. 15
        #   return x.cpu().tolist()
        raise NotImplementedError(
            "CANDISampler.sample: wire up self._step from candi-lander "
            "(reverse-ODE + masked-ancestral update, paper eqs. 14–15)."
        )

    @torch.no_grad()
    def logits_at(
        self,
        token_ids: torch.LongTensor,
        noise_level: float,                # REQUIRED for CANDI (= diffusion time t)
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Predict logits P(x_0 | x_t) at the specified diffusion time t.

        CANDI's forward process is hybrid at every t:
          - discrete channel: each position is independently mask_id with some
            probability α(t); otherwise the clean token.
          - continuous channel: the embedding at each UNMASKED position gets
            additive Gaussian noise of scale σ(t). Masked positions carry no
            continuous information (the embedding is conditionally irrelevant
            given the mask).

        The caller supplies the discrete pattern via `mask` (True = masked),
        and the scalar time via `noise_level`. σ(t) is derived from the
        training range (r_min, r_max) with a linear schedule — replace this
        with whatever schedule your checkpoint was trained under if different;
        `self.sigma_schedule`, if your wired-up model exposes one, overrides.

        Returns logits of shape [batch, seq_length, vocab_size].
        """
        self._require_init()

        device = self.device
        x = token_ids.to(device)
        B, L = x.shape
        t = float(noise_level)

        # ---- Discrete channel: materialize x_discrete with mask_id ---------
        if mask is not None:
            m = mask.to(device)
            x_discrete = x.clone()
            x_discrete[m] = self.mask_id
        else:
            m = torch.zeros_like(x, dtype=torch.bool)
            x_discrete = x

        # ---- Continuous channel: noised embeddings -------------------------
        # σ(t): prefer an explicit schedule if the checkpoint exposes one
        # (e.g. cosine / VP), else fall back to a linear interpolation over the
        # training range [r_min, r_max].
        t_clamped = max(0.0, min(1.0, t))
        if hasattr(self, "sigma_schedule") and callable(self.sigma_schedule):
            sigma_t = float(self.sigma_schedule(t_clamped))
        else:
            sigma_t = self.r_min + t_clamped * (self.r_max - self.r_min)

        # Embedding-lookup approximation (paper §5.3): we feed the predictor
        # actual embeddings rather than |V|-dim one-hots at inference, which is
        # the only way this is tractable at |V|=262K.
        emb = self.model.get_input_embeddings()(x_discrete)        # [B, L, D]
        if sigma_t > 0.0:
            noise = torch.randn(emb.shape, device=device, dtype=emb.dtype) * sigma_t
            # Don't add continuous noise to masked positions — they don't carry
            # continuous information in the forward process.
            noise = noise.masked_fill(m.unsqueeze(-1), 0.0)
            emb = emb + noise

        # ---- Time conditioning --------------------------------------------
        t_tensor = torch.full((B,), t, device=device, dtype=emb.dtype)

        # ---- Predictor call ------------------------------------------------
        # candi-lander's predictor accepts (input_ids, inputs_embeds, t).
        # Some forks use positional args or rename `t` → `timesteps`; we try
        # the canonical keyword form first and fall back on TypeError.
        try:
            out = self.model(
                input_ids=x_discrete,
                inputs_embeds=emb,
                t=t_tensor,
            )
        except TypeError:
            try:
                out = self.model(
                    input_ids=x_discrete,
                    inputs_embeds=emb,
                    timesteps=t_tensor,
                )
            except TypeError:
                out = self.model(x_discrete, emb, t_tensor)

        logits = out.logits if hasattr(out, "logits") else out

        # Sanity check: final dim must be vocab_size. Catches the case where
        # somebody accidentally wires up a score-function head (dim D) instead
        # of a classifier head (dim V) — which would silently return garbage
        # through _internal_perplexity otherwise.
        if logits.shape[-1] != self.vocab_size:
            raise RuntimeError(
                f"CANDI model returned logits with final dim {logits.shape[-1]}, "
                f"expected vocab_size={self.vocab_size}. Your predictor is likely "
                f"returning scores / embeddings instead of classifier logits."
            )
        return logits


# ===========================================================================
# Reference MDLM sampler (fully working — baseline for sanity checks)
# ===========================================================================
class MDLMSampler(DiffusionSampler):
    """
    Minimal masked-diffusion sampler using a pre-trained HF model.

    Uses `AutoModelForMaskedLM` because the public MDLM checkpoints
    (e.g. kuleshov-group/mdlm-owt) are registered under the MaskedLM auto-class
    and return output objects with a `.logits` attribute. If you bring in a
    custom MDLM variant that isn't registered there, either register it or
    switch this to `AutoModel` (LLaDA's pattern).

    Kept fully working so that the rest of the harness (metrics, frontier
    sweeps, length-gen) can be validated before LLaDA/SEDD/CANDI are wired up.

    Unmasking scheme: deterministic top-k by predicted confidence, matching
    LLaDASampler. This deviates from the original MDLM paper's stochastic
    Bernoulli unmask (which has high per-sequence variance at low NFE and
    causes the final residual-argmax pass to do systematically more cleanup
    for small NFE than large NFE — a confound for NFE-axis frontier plots).
    """

    def __init__(self, config: SamplerConfig) -> None:
        super().__init__(config)
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.model = AutoModelForMaskedLM.from_pretrained(
            config.checkpoint_path, torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(config.checkpoint_path)
        self.mask_id = self.tok.mask_token_id

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tok

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
            # Deterministic top-k confidence-based unmasking (see class docstring).
            # unmask_frac = fraction of currently-masked positions to unmask now.
            n_mask_curr = mask.sum(dim=-1)
            unmask_frac = 1.0 - (s / max(t, 1e-6))
            conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            conf = conf.masked_fill(~mask, -1.0)
            n_to_fill = (unmask_frac * n_mask_curr.float()).long()
            for b in range(n_sequences):
                k = int(n_to_fill[b].item())
                if k <= 0:
                    continue
                _, idx = conf[b].topk(k)
                x[b, idx] = pred[b, idx]
        # Residual cleanup: any positions that top-k didn't cover (rare, can
        # happen if n_to_fill rounded down to 0 on all steps for some seq).
        if (x == self.mask_id).any():
            logits = self.model(x).logits
            pred = logits.argmax(dim=-1)
            x = torch.where(x == self.mask_id, pred, x)
        return x.cpu().tolist()

    @torch.no_grad()
    def logits_at(
        self,
        token_ids: torch.LongTensor,
        noise_level: float,                # absorbing; mask carries t
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
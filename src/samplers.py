"""
Diffusion sampler interface + adapters.

All downstream metrics (frontier, judge-PPL, per-position entropy) consume a
`DiffusionSampler`. We standardize the interface so that swapping LLaDA for
SEDD for CANDI is a config change, not a code change.

Interface:
    sample(n_sequences, seq_length, nfe, temperature, seed) -> list[list[int]]
    logits_at(token_ids, noise_level, mask) -> Tensor   # for diagnostics
    attention_maps(token_ids, noise_level) -> list[Tensor] | None
    tokenizer -> PreTrainedTokenizerBase                 # property

The sampler owns its tokenizer (via the `tokenizer` property). Downstream code
consumes it from the sampler rather than re-looking-up via TOKENIZER_REGISTRY,
because samplers with non-standard tokenizers (e.g. LLaDA-8B-Base's ~126K
vocab) would get silently decoded with the wrong tokenizer otherwise.

Noise-level contract:
    `logits_at(x, noise_level=t, mask=m)` must return P(x_0 | x_t) at the
    specified t. Absorbing-discrete methods (MDLM, LLaDA) can infer t from the
    mask pattern and may ignore `noise_level`; score-based methods (SEDD) and
    hybrid methods (CANDI) MUST honor it.

Precision / memory note:
    Several adapters load the model in bf16 for speed but upcast logits to fp32
    before softmax. In bf16 the per-row sum of a large-vocab softmax deviates
    from 1 enough to violate Categorical's simplex check; fp32 fixes that. For
    large vocabs this means we can't hold [B, L, V] fp32 at full batch, so
    LLaDA micro-batches internally.
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from .tokenizers_bench import load_tokenizer


# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------
@dataclass
class SamplerConfig:
    model_name: str
    checkpoint_path: str
    tokenizer_name: str
    device: str = "cuda"
    dtype: str = "bfloat16"


class DiffusionSampler(ABC):
    def __init__(self, config: SamplerConfig) -> None:
        self.config = config
        self.device = config.device
        self.dtype = getattr(torch, config.dtype)

    @abstractmethod
    def sample(self, n_sequences: int, seq_length: int, nfe: int,
               temperature: float = 1.0, seed: int = 0) -> list[list[int]]: ...

    @abstractmethod
    def logits_at(self, token_ids: torch.LongTensor, noise_level: float,
                  mask: Optional[torch.BoolTensor] = None) -> torch.Tensor: ...

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase: ...

    def attention_maps(self, token_ids: torch.LongTensor,
                       noise_level: float) -> Optional[list[torch.Tensor]]:
        return None


# ===========================================================================
# LLaDA adapter
# ===========================================================================
class LLaDASampler(DiffusionSampler):
    """
    LLaDA (GSAI-ML/LLaDA-8B-Base): masked-diffusion with ~126K-vocab LLaMA
    tokenizer. Loaded via AutoModel with trust_remote_code=True.

    Memory at defaults without chunking:
        n_sequences × seq_length × vocab_size × 4 bytes (fp32 logits)
        = 64 × 1024 × 126 464 × 4 ≈ 33 GB
    so we forward in MICRO_BATCH-sized chunks; peak logit memory is ~4 GB
    independent of n_sequences. Sampling and confidence-gather happen per chunk.
    """

    MICRO_BATCH: int = 8

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
        self.mask_id = getattr(self.model.config, "mask_token_id", 126336)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tok

    @torch.no_grad()
    def _forward_chunk_fp32(self, x_chunk: torch.LongTensor) -> torch.Tensor:
        return self.model(x_chunk).logits.float()

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
        x = torch.full(
            (n_sequences, seq_length), self.mask_id,
            dtype=torch.long, device=self.device,
        )
        ts = torch.linspace(1.0, 0.0, nfe + 1, device=self.device)
        mb = max(1, self.MICRO_BATCH)
        inv_temp = 1.0 / max(temperature, 1e-6)

        for step in range(nfe):
            t, s = ts[step].item(), ts[step + 1].item()
            mask = (x == self.mask_id)
            if not mask.any():
                break

            pred = torch.empty_like(x)
            conf = torch.empty(x.shape, dtype=torch.float32, device=self.device)

            for i in range(0, n_sequences, mb):
                j = min(i + mb, n_sequences)
                logits = self._forward_chunk_fp32(x[i:j])       # [mb, L, V] fp32
                if temperature != 1.0:
                    logits = logits * inv_temp
                probs = F.softmax(logits, dim=-1)
                # validate_args=False: skip the O(B·L·V) simplex check; the
                # fp32 upcast already guarantees rows sum to ~1.
                pred_chunk = torch.distributions.Categorical(
                    probs=probs, validate_args=False,
                ).sample()
                conf_chunk = probs.gather(-1, pred_chunk.unsqueeze(-1)).squeeze(-1)
                pred[i:j] = pred_chunk
                conf[i:j] = conf_chunk
                del logits, probs, pred_chunk, conf_chunk

            conf = conf.masked_fill(~mask, -1.0)
            n_mask_curr = mask.sum(dim=-1)
            unmask_frac = 1.0 - (s / max(t, 1e-6))
            n_to_fill = (unmask_frac * n_mask_curr.float()).long()
            for b in range(n_sequences):
                k = int(n_to_fill[b].item())
                if k <= 0:
                    continue
                _, idx = conf[b].topk(k)
                x[b, idx] = pred[b, idx]

        # Residual cleanup — positions top-k never reached.
        if (x == self.mask_id).any():
            for i in range(0, n_sequences, mb):
                j = min(i + mb, n_sequences)
                logits = self._forward_chunk_fp32(x[i:j])
                argmax_chunk = logits.argmax(dim=-1)
                chunk_mask = x[i:j] == self.mask_id
                x[i:j] = torch.where(chunk_mask, argmax_chunk, x[i:j])
                del logits, argmax_chunk

        return x.cpu().tolist()

    @torch.no_grad()
    def logits_at(
        self,
        token_ids: torch.LongTensor,
        noise_level: float,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Returns fp32 logits, forwarded in MICRO_BATCH chunks. Caller is
        responsible for keeping B small — concatenating [128, 1024, 126K]
        fp32 logits is ~66 GB and will OOM. Diagnostic callers in this repo
        use B ≤ 8 so they're fine.
        """
        x = token_ids.to(self.device)
        if mask is not None:
            x = x.clone()
            x[mask.to(self.device)] = self.mask_id
        mb = max(1, self.MICRO_BATCH)
        outs: list[torch.Tensor] = []
        for i in range(0, x.shape[0], mb):
            outs.append(self._forward_chunk_fp32(x[i:i + mb]))
        return torch.cat(outs, dim=0)


# ===========================================================================
# SEDD adapter
# ===========================================================================
class SEDDSampler(DiffusionSampler):
    """
    SEDD (Lou et al. 2024) adapter using the louaaron/Score-Entropy-
    Discrete-Diffusion repo. Path injected into sys.path at __init__;
    override via SEDD_REPO_PATH env var.
    """

    _DEFAULT_REPO_PATH = "/home/aneek/src/dLLM-eval/aneek/Score-Entropy-Discrete-Diffusion"

    def __init__(self, config: SamplerConfig) -> None:
        super().__init__(config)

        import os
        sedd_repo_path = os.environ.get("SEDD_REPO_PATH", self._DEFAULT_REPO_PATH)
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
                f"Set SEDD_REPO_PATH env var to your clone of "
                f"louaaron/Score-Entropy-Discrete-Diffusion."
            ) from e

        print(f"[sampler] loading SEDD model from {config.checkpoint_path}")
        self.net = SEDD.from_pretrained(config.checkpoint_path).to(self.device).eval()
        self.cfg = self.net.config
        self.graph = get_graph(self.cfg, device=self.device)
        self.noise = get_noise(self.cfg)
        self._get_pc_sampler = get_pc_sampler
        self.tok = load_tokenizer(config.tokenizer_name)

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
        if temperature != 1.0:
            print(
                f"[warn] SEDDSampler.sample: temperature={temperature} is not "
                f"supported by the upstream analytic predictor and will be ignored. "
                f"This means the τ-axis of any frontier plot is meaningless for SEDD."
            )
        sampling_fn = self._get_pc_sampler(
            graph=self.graph,
            noise=self.noise,
            batch_dims=(n_sequences, seq_length),
            predictor="analytic",
            steps=nfe,
            denoise=True,
            device=self.device,
        )
        out = sampling_fn(self.net)
        return out.cpu().tolist()

    @torch.no_grad()    
    def logits_at(
        self,
        token_ids: torch.LongTensor,
        noise_level: float,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Computes standard autoregressive-style logits P(x_0 | x_t) from SEDD's score network.
        """
        token_ids = token_ids.to(self.device)
        safe_noise = max(noise_level, 1e-4)
        # 1. Expand the scalar noise_level (t) to a batched continuous tensor
        t_tensor = torch.full((token_ids.shape[0],), safe_noise, device=self.device)
        
        # 2. Get the raw scores from the SEDD network
        # SEDD networks typically output unnormalized values representing the score
        raw_scores = self.net(token_ids, t_tensor)
        
        # 3. Convert Scores to Logits based on the Diffusion Graph
        # In SEDD, the relationship between the network output and P(x_0 | x_t)
        # depends heavily on whether you are using the 'Uniform' or 'Absorbing' graph.
        
        # For the Absorbing (Masking) graph (which is standard for their language models):
        if self.graph.absorb:
            # For absorbing, the score at masked positions *is* directly proportional 
            # to the predicted clean distribution P(x_0 | x_t = MASK). 
            # We just need to mask out the absorbing token's own logit to prevent self-transitions.
            
            mask_token_id = self.tokenizer.mask_token_id if hasattr(self.tokenizer, "mask_token_id") else self.tokenizer.eos_token_id
            
            logits = raw_scores.clone()
            # Force the probability of predicting the mask token as the clean token to 0 (-inf in log space)
            if mask_token_id is not None:
                logits[..., mask_token_id] = -float('inf')
                
        else:
            # For the Uniform graph:
            # We must weight the scores by the forward transition rates to extract P(x_0)
            # normalized_rate = self.graph.transp_rate(token_ids) * raw_scores
            # (Note: depending on the exact SEDD version, you might just be able to 
            # use raw_scores directly as they often parameterize the network to output 
            # standard logits prior to the score conversion layer).
            
            # Safe fallback for uniform models in the official repo:
            logits = raw_scores 
            
        # Optional: If your pipeline specifically passes in the boolean `mask` (where True = masked),
        # you can optimization by only returning meaningful logits for masked positions, 
        # though returning the full tensor is safer for standard protocol compliance.
        if mask is not None:
            # If you want to explicitly zero out predictions on clean tokens
            pass 
            
        return logits


# ===========================================================================
# CANDI adapter (stub)
# ===========================================================================
class CANDISampler(DiffusionSampler):
    """
    CANDI (Pynadath et al. 2026) hybrid discrete+continuous sampler.
    Repo: https://github.com/patrickpynadath1/candi-lander.
    Uses the embedding-lookup approximation from §5.3.
    """

    _REQUIRED_ATTRS = ("model", "vocab_size", "mask_id", "r_min", "r_max")

    def __init__(self, config: SamplerConfig) -> None:
        super().__init__(config)
        self.tok = load_tokenizer(config.tokenizer_name)

        # TODO: INTEGRATE CANDI. The wired-up __init__ must set ALL of:
        #   self.model        — loaded predictor; .get_input_embeddings() usable
        #   self.vocab_size   — int, classifier head output dim
        #   self.mask_id      — int, the discrete mask token id
        #   self.r_min/r_max  — float, σ training range for linear fallback
        #
        # Example:
        #   from candi_repo import load_candi_ckpt
        #   self.model = load_candi_ckpt(config.checkpoint_path).to(self.device).eval()
        #   self.vocab_size = self.model.config.vocab_size
        #   self.mask_id    = self.model.config.mask_token_id
        #   self.r_min, self.r_max = self.model.config.r_range
        self._initialized = False

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tok

    def _require_init(self) -> None:
        if self._initialized:
            return
        missing = [a for a in self._REQUIRED_ATTRS if not hasattr(self, a)]
        raise NotImplementedError(
            f"CANDISampler not wired up. Paste candi-lander loading code into "
            f"__init__ and set self._initialized = True. Missing attributes: "
            f"{missing}."
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
        raise NotImplementedError(
            "CANDISampler.sample: wire up self._step from candi-lander "
            "(reverse-ODE + masked-ancestral update, paper eqs. 14–15)."
        )

    @torch.no_grad()
    def logits_at(
        self,
        token_ids: torch.LongTensor,
        noise_level: float,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Predict logits P(x_0 | x_t) at the specified diffusion time t.

        CANDI's forward process is hybrid at every t:
          - discrete: each position is mask_id with prob α(t), else clean
          - continuous: unmasked positions' embeddings get Gaussian noise of
            scale σ(t); masked positions carry no continuous info (noise zeroed)

        σ(t) uses self.sigma_schedule(t) if available, else linear [r_min, r_max].
        """
        self._require_init()

        device = self.device
        x = token_ids.to(device)
        B, L = x.shape
        t = float(noise_level)

        if mask is not None:
            m = mask.to(device)
            x_discrete = x.clone()
            x_discrete[m] = self.mask_id
        else:
            m = torch.zeros_like(x, dtype=torch.bool)
            x_discrete = x

        t_clamped = max(0.0, min(1.0, t))
        if hasattr(self, "sigma_schedule") and callable(self.sigma_schedule):
            sigma_t = float(self.sigma_schedule(t_clamped))
        else:
            sigma_t = self.r_min + t_clamped * (self.r_max - self.r_min)

        emb = self.model.get_input_embeddings()(x_discrete)
        if sigma_t > 0.0:
            noise = torch.randn(emb.shape, device=device, dtype=emb.dtype) * sigma_t
            noise = noise.masked_fill(m.unsqueeze(-1), 0.0)
            emb = emb + noise

        t_tensor = torch.full((B,), t, device=device, dtype=emb.dtype)

        try:
            out = self.model(input_ids=x_discrete, inputs_embeds=emb, t=t_tensor)
        except TypeError:
            try:
                out = self.model(input_ids=x_discrete, inputs_embeds=emb, timesteps=t_tensor)
            except TypeError:
                out = self.model(x_discrete, emb, t_tensor)

        logits = out.logits if hasattr(out, "logits") else out
        if logits.shape[-1] != self.vocab_size:
            raise RuntimeError(
                f"CANDI model returned logits with final dim {logits.shape[-1]}, "
                f"expected vocab_size={self.vocab_size}. Your predictor is likely "
                f"returning scores / embeddings instead of classifier logits."
            )
        return logits


# ===========================================================================
# MDLM sampler
# ===========================================================================
class MDLMSampler(DiffusionSampler):
    """
    Masked-diffusion sampler for kuleshov-group/mdlm-owt.

    Design notes:
    - Loaded with torch_dtype=fp32. MDLM's custom modeling wraps transformer
      blocks in a hard-coded bf16 autocast internally, and its timestep-embed
      MLP does a `.float()` upcast — if outer weights are bf16 the two paths
      collide (Float vs BFloat16 matmul).
    - Top-level MDLM.forward() doesn't expose `sigma`, but `.backbone(x, sigma)`
      does and requires a real tensor (not None).
    - The backbone returns bf16 logits regardless of outer dtype; sample and
      logits_at upcast to fp32 before softmax (bf16 rows don't sum to 1 tightly).

    Unmasking: deterministic top-k by confidence, matching LLaDA.
    """

    def __init__(self, config: SamplerConfig) -> None:
        super().__init__(config)
        from transformers import AutoModelForMaskedLM
        self.model = AutoModelForMaskedLM.from_pretrained(
            config.checkpoint_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(self.device).eval()
        self.tok = load_tokenizer(config.tokenizer_name)
        self.mask_id = getattr(self.model.config, "mask_token_id", self.tok.vocab_size)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tok

    def _zero_sigma(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, device=self.device, dtype=torch.float32)

    def _model_call(self, x: torch.Tensor, output_attentions: bool = False):
        """Three-fallback forward: sigma-kwarg → backbone → plain."""
        x = x.to(self.device)
        sigma = self._zero_sigma(x.shape[0])
        kwargs = {"output_attentions": output_attentions} if output_attentions else {}

        try:
            return self.model(x, sigma=sigma, **kwargs)
        except TypeError:
            pass

        if hasattr(self.model, "backbone"):
            try:
                out = self.model.backbone(x, sigma)
                return out[0] if isinstance(out, tuple) else out
            except TypeError:
                pass

        return self.model(x, **kwargs)

    @staticmethod
    def _extract_logits(out) -> torch.Tensor:
        t = out.logits if hasattr(out, "logits") else out
        return t.float()

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

            logits = self._extract_logits(self._model_call(x)) / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            pred = torch.distributions.Categorical(
                probs=probs, validate_args=False,
            ).sample()

            n_mask_curr = mask.sum(dim=-1)
            unmask_frac = 1.0 - (s / max(t, 1e-6))

            conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            conf = conf.masked_fill(~mask, -1.0)

            n_to_fill = (unmask_frac * n_mask_curr.float()).long()
            for b in range(n_sequences):
                k = int(n_to_fill[b].item())
                if k > 0:
                    _, idx = conf[b].topk(k)
                    x[b, idx] = pred[b, idx]

        if (x == self.mask_id).any():
            logits = self._extract_logits(self._model_call(x))
            x = torch.where(x == self.mask_id, logits.argmax(dim=-1), x)

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
        return self._extract_logits(self._model_call(x))

    def attention_maps(
        self, token_ids: torch.LongTensor, noise_level: float,
    ) -> Optional[list[torch.Tensor]]:
        x = token_ids.to(self.device)
        out = self._model_call(x, output_attentions=True)
        return list(out.attentions) if hasattr(out, "attentions") else None


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
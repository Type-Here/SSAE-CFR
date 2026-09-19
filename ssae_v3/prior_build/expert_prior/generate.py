"""One call to the frozen model, with the generation budget computed rather than guessed.

The expert prior is a single global generation: one prompt in, one structured document
out. That is a deliberate constraint, not a limitation to work around - a document
stitched from several calls would have no single authority behind its internal
consistency, and the per-card cross-references are exactly what makes it a prior
rather than a pile of glosses.

Budget
------
A hardcoded `max_new_tokens` is a guess about a number the model already knows. The
context window and the prompt length are both measurable before any weight is loaded,
so the budget is derived:

    budget = context_limit - prompt_tokens - margin

and if that is below the floor the run fails *before* inference, with the arithmetic
printed. Failing at minute zero beats failing after a 7B model has been placed on the
GPU and has spent minutes emitting a document that was always going to be truncated.
`AutoConfig` and `AutoTokenizer` are small downloads, so the check is cheap.

Decoding is greedy by default. A research artifact should re-derive from what the
manifest records, and greedy decoding is the only setting for which that is true of
the text as well as the parameters.

Placement follows the same recipe as the embedding path: `device_map="auto"` on CUDA
and no `.to()` afterwards, because `from_pretrained(...).to(device)` materializes the
whole model in host RAM first and OOMs a host smaller than the model even when the
GPU had room.

Quantization
------------
`device_map="auto"` solves host RAM; it does not create VRAM. A 7B model at float16
is about 13.5 GiB against a T4's 15 GiB, and this prompt is long enough (roughly 8.5k
tokens on IHDP) that the KV cache for the prefill alone does not fit in the ~1.5 GiB
left over. The symptom is a CUDA OOM during generation, or accelerate silently
spilling layers to CPU and turning minutes into hours. `load_in_4bit` puts the weights
near 4 GiB, which fits with room for the cache.

It is offered for generation and deliberately not for embedding. The embedding stage
stays at the width that produced V, because the point of pinning the pooling rule is
that a later comparison between those vectors and V is about what was written rather
than how it was produced, and a quantized encoder would reintroduce exactly the
confound the pin removes. Generation carries no such constraint: the artifact is the
text the model wrote, and the manifest records the width it was written at, so a
quantized run reproduces on its own terms and is honestly labelled against an fp16
one. The two are not expected to be token-identical, and nothing here pretends they
are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

DEFAULT_MODEL = "BioMistral/BioMistral-7B"

# Below this many tokens of headroom a full FeatureCard set plus a ConceptBank cannot
# fit, so generating would only produce a document that fails validation on truncation.
DEFAULT_MIN_BUDGET = 3000
DEFAULT_CONTEXT_MARGIN = 64

# NF4 with double quantization: the QLoRA defaults, and the only 4-bit setting in wide
# enough independent use to be treated as a known quantity rather than one more knob.
# Not exposed, for the same reason the pooling rule is not exposed.
QUANT_TYPE_4BIT = "nf4"
DOUBLE_QUANT_4BIT = True

# transformers uses a sentinel this large when a tokenizer declares no limit
_NO_LIMIT = 1_000_000


class GenerationError(RuntimeError):
    """Generation could not be attempted, or did not produce any text."""


def resolve_device(device: str) -> str:
    """Turn "auto" into the device that will really be used.

    Lives here rather than at the call site so the dry run, which reports what a real
    run would do, cannot disagree with the real run about where it would happen.
    """
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype: str, device: str) -> str:
    """The width the weights will actually be loaded at.

    Pure and shared, so the dry run reports the same answer the loader will reach
    rather than a second copy of the rule that can drift from it.
    """
    if dtype != "auto":
        return dtype
    return "float16" if device.startswith("cuda") else "float32"


def quantization_record(
    load_in_4bit: bool,
    device: str,
    dtype: str,
    *,
    double_quant: bool = DOUBLE_QUANT_4BIT,
) -> Optional[Dict[str, Any]]:
    """Decide 4-bit loading before any weight is read, and describe it for the manifest.

    Returns None when the weights are loaded at full width, so a manifest distinguishes
    "not quantized" from "quantized and not written down". Kept free of the
    bitsandbytes import: whether a run is quantized is a fact about the artifact and
    must be checkable without the library that would perform it.
    """
    if not load_in_4bit:
        return None
    if not device.startswith("cuda"):
        raise GenerationError(
            "4-bit loading needs a CUDA device; bitsandbytes has no CPU 4-bit path. "
            "Drop --load-in-4bit to generate at full width."
        )
    return {
        "load_in_4bit": True,
        "quant_type": QUANT_TYPE_4BIT,
        "compute_dtype": resolve_dtype(dtype, device),
        "double_quant": bool(double_quant),
    }


@dataclass
class GenerationPlan:
    """What will be asked of the model, decided before the weights are loaded."""

    model_name: str
    prompt_tokens: int
    context_limit: int
    margin: int
    max_new_tokens: int
    context_source: str
    sliding_window: Optional[int] = None
    notes: list = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "context_limit": self.context_limit,
            "context_source": self.context_source,
            "context_margin": self.margin,
            "max_new_tokens": self.max_new_tokens,
            "sliding_window": self.sliding_window,
            "notes": list(self.notes),
        }

    def describe(self) -> str:
        return (
            f"context {self.context_limit} ({self.context_source}) "
            f"- prompt {self.prompt_tokens} - margin {self.margin} "
            f"=> max_new_tokens {self.max_new_tokens}"
        )


def _context_limit(config: Any, tokenizer: Any) -> tuple:
    """Smallest credible context limit, and where it came from."""
    candidates = []
    for source, value in (
        ("config.max_position_embeddings", getattr(config, "max_position_embeddings", None)),
        ("tokenizer.model_max_length", getattr(tokenizer, "model_max_length", None)),
    ):
        if isinstance(value, int) and 0 < value < _NO_LIMIT:
            candidates.append((value, source))
    if not candidates:
        raise GenerationError(
            "the model declares no usable context limit; pass --context-limit explicitly"
        )
    limit, source = min(candidates)
    return limit, source


def size_budget(
    prompt_tokens: int,
    context_limit: int,
    *,
    margin: int = DEFAULT_CONTEXT_MARGIN,
    min_budget: int = DEFAULT_MIN_BUDGET,
    max_new_tokens: Optional[int] = None,
    context_source: str = "model config",
) -> int:
    """How many tokens may be generated. Raises when the answer is 'not enough'.

    Kept pure and separate from the model so the arithmetic that decides whether a run
    is worth starting can be exercised without downloading anything.
    """
    budget = context_limit - prompt_tokens - margin
    if budget < min_budget:
        raise GenerationError(
            f"generation cannot fit: context {context_limit} ({context_source}) - prompt "
            f"{prompt_tokens} - margin {margin} leaves {budget} tokens, below the floor of "
            f"{min_budget}. Shorten the prompt, raise --context-limit if the model really "
            "supports more, or lower --min-budget deliberately."
        )
    if max_new_tokens is None:
        return budget

    requested = int(max_new_tokens)
    if requested > budget:
        raise GenerationError(
            f"--max-new-tokens {requested} does not fit: only {budget} tokens are left after "
            f"a {prompt_tokens}-token prompt in a {context_limit}-token context "
            f"(margin {margin})"
        )
    if requested < min_budget:
        raise GenerationError(
            f"--max-new-tokens {requested} is below the floor of {min_budget}; a full "
            "FeatureCard set plus a ConceptBank will not fit in it"
        )
    return requested


def plan_generation(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    *,
    max_new_tokens: Optional[int] = None,
    min_budget: int = DEFAULT_MIN_BUDGET,
    margin: int = DEFAULT_CONTEXT_MARGIN,
    context_limit: Optional[int] = None,
) -> GenerationPlan:
    """Size the generation against the real context window, before loading any weights."""
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    if context_limit is not None:
        limit, source = int(context_limit), "explicit --context-limit"
    else:
        limit, source = _context_limit(config, tokenizer)

    prompt_tokens = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    allowed = size_budget(
        prompt_tokens,
        limit,
        margin=margin,
        min_budget=min_budget,
        max_new_tokens=max_new_tokens,
        context_source=source,
    )

    plan = GenerationPlan(
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        context_limit=limit,
        margin=margin,
        max_new_tokens=allowed,
        context_source=source,
        sliding_window=getattr(config, "sliding_window", None),
    )
    if isinstance(plan.sliding_window, int) and prompt_tokens > plan.sliding_window:
        plan.notes.append(
            f"prompt ({prompt_tokens} tokens) is longer than the model's sliding window "
            f"({plan.sliding_window}); attention over the earliest sections may be limited"
        )
    return plan


def _quantization_config(quantization: Dict[str, Any]):
    """Turn the recorded decision into the transformers object that performs it."""
    import torch

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - depends on the installed stack
        raise GenerationError(
            "4-bit loading needs a transformers build that provides BitsAndBytesConfig"
        ) from exc
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise GenerationError(
            "4-bit loading needs bitsandbytes, which is not installed. "
            "Install it (pip install bitsandbytes) or drop --load-in-4bit."
        ) from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quantization["quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quantization["compute_dtype"]),
        bnb_4bit_use_double_quant=quantization["double_quant"],
    )


def _load_causal_lm(
    model_name: str,
    device: str,
    dtype: str,
    quantization: Optional[Dict[str, Any]] = None,
):
    """Load the generation model. Same placement recipe as the embedding path."""
    import torch
    from transformers import AutoModelForCausalLM

    dtype = resolve_dtype(dtype, device)
    torch_dtype = getattr(torch, dtype)

    kwargs = {"low_cpu_mem_usage": True}
    if device.startswith("cuda"):
        kwargs["device_map"] = "auto"
    if quantization is not None:
        # bitsandbytes places the quantized weights itself; a device_map is required
        # and `.to()` afterwards is an error rather than a slow path.
        kwargs["device_map"] = "auto"
        kwargs["quantization_config"] = _quantization_config(quantization)

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype, **kwargs)
    except TypeError:
        # older transformers spell it `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, **kwargs
        )

    if "device_map" not in kwargs:
        model = model.to(device)
    model.eval()
    return model, next(model.parameters()).device, dtype


def generate_expert_prior(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    *,
    plan: Optional[GenerationPlan] = None,
    dtype: str = "auto",
    device: str = "auto",
    load_in_4bit: bool = False,
    do_sample: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: int = 0,
    **plan_kwargs: Any,
) -> tuple:
    """Run the single generation. Returns `(raw_text, params)` for the manifest."""
    import torch
    from transformers import AutoTokenizer

    if plan is None:
        plan = plan_generation(prompt, model_name, **plan_kwargs)

    device = resolve_device(device)

    # raises before the tokenizer downloads if 4-bit was asked for on a CPU device
    quantization = quantization_record(load_in_4bit, device, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, input_device, resolved_dtype = _load_causal_lm(
        model_name, device, dtype, quantization
    )
    torch.manual_seed(seed)

    enc = tokenizer(prompt, return_tensors="pt").to(input_device)
    kwargs: Dict[str, Any] = {
        "max_new_tokens": plan.max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)

    with torch.no_grad():
        out = model.generate(**enc, **kwargs)

    # decode only what was generated, so the prompt never contaminates the response
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    if not raw.strip():
        raise GenerationError("the model returned no text")

    params = {
        "model_name": model_name,
        "dtype": resolved_dtype,
        "quantization": quantization,
        "device": str(input_device),
        "seed": seed,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "num_return_sequences": 1,
        "generated_tokens": int(new_tokens.shape[0]),
        "hit_token_cap": int(new_tokens.shape[0]) >= plan.max_new_tokens,
        **plan.as_dict(),
    }
    return raw, params

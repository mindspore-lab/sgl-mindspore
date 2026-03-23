# CLAUDE.md

This file provides guidance to Claude Code when working with the sgl-mindspore repository.

## Overview

sgl-mindspore is the MindSpore/Ascend NPU backend for [SGLang](https://github.com/sgl-project/sglang). It provides inference-only model implementations that run on Ascend 910B/910C/310P hardware via MindSpore.

## Repository Structure

```
sgl-mindspore/
├── sgl_mindspore/
│   ├── models/              # Model implementations (one file per architecture)
│   │   ├── mindspore_model_base.py  # Base class all models inherit from
│   │   ├── llama.py
│   │   ├── llama_eagle3.py
│   │   ├── qwen3.py
│   │   ├── qwen3_moe.py
│   │   ├── qwen3_5.py       # Hybrid attention (full-attn + GatedDeltaNet)
│   │   └── deepseekv3.py
│   ├── layers/              # Reusable layer primitives
│   │   ├── __init__.py      # Public exports — update when adding new layers
│   │   ├── attention.py     # MsNativeAttnBackend
│   │   ├── linear.py        # ColParallelLinear, RowParallelLinear, QKVParallelLinear,
│   │   │                    #   MLPColParallelLinear, MergedColParallelLinear, ...
│   │   ├── norm.py          # RMSNorm, GemmaRMSNorm
│   │   ├── rope.py          # BaseRotaryEmbedding, YaRNScalingRotaryEmbedding,
│   │   │                    #   PartialRotaryEmbedding, DeepseekScalingRotaryEmbedding
│   │   ├── activation.py    # SwiGLU
│   │   ├── vocab_embedding.py
│   │   └── moe/             # Mixture-of-experts layers
│   │   └── quantization/    # W8A8 int8 quant support
├── examples/                # Offline/server inference scripts and benchmarks
├── doc/                     # Additional documentation
└── patch/                   # Patches for SGLang (e.g., 310P Triton workaround)
```

## Model Implementation Pattern

Every model file must:
1. Inherit from `MindSporeModelBase` (not directly from `ms.nn.Cell`)
2. Implement `construct(self, **model_inputs) -> Tensor`
3. Implement `load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]])`
4. Optionally override `prepare_inputs(self, forward_batch, model_inputs)` for custom pre-processing (e.g., state management for recurrent models)
5. End with `EntryClass = <ModelClass>` for dynamic registration by SGLang

```python
class MyForCausalLM(MindSporeModelBase):
    def construct(self, **model_inputs) -> Tensor: ...
    def load_weights(self, weights): ...
    def set_model_inputs(self, is_prefill): ...  # called when prefill/decode mode switches

EntryClass = MyForCausalLM
```

## Key Conventions

### Weight Loading
- `self.parameters_dict()` returns params keyed by their full path (e.g., `model.layers.0.self_attn.qkv_proj.weight`)
- Parameter names are set via the `prefix` argument passed down through constructors using `add_prefix(name, prefix)`
- Parameters created with `quant_method.create_weights(layer=self, ...)` get a `weight_load` attribute for custom sharding
- Stacked weights (e.g., `qkv_proj` from separate `q_proj`/`k_proj`/`v_proj`) use the `weight_load(param, weight, shard_id)` pattern
- Default fallback: `param.set_data(tensor_torch2ms(weight).move_to("Ascend"))`

### Tensor Parallelism
- Use `get_attention_tp_rank()` / `get_attention_tp_size()` for attention-side TP (may differ from global TP)
- Use `get_tensor_model_parallel_world_size()` for global TP
- `ColParallelLinear`: shards output dim; `RowParallelLinear`: shards input dim
- `MergedColParallelLinear`: for fused weights with multiple sections (e.g., QKV with unequal sizes), each section sharded independently

### Prefill vs Decode
- Models receive `is_prefill: bool` in `construct()`
- JIT-compiled models call `set_model_inputs(is_prefill)` when the mode changes; models without JIT implement it as `pass`
- The `model.phase` attribute (`"prefill"` or `"increment"`) is set before calling into the inner `nn.Cell`

### 310P Specifics
- Does not support bfloat16 → convert to float16
- Use `is_310p()` helper from `sgl_mindspore.utils`
- Weights for linear layers should be cast to NZ format via `format_cast(param, "nz")` after loading
- Set `MS_ENABLE_INTERNAL_BOOST=off`

### GemmaRMSNorm vs RMSNorm
- `RMSNorm`: weight initialized to ones, formula `rms_norm(x) * w`
- `GemmaRMSNorm`: weight initialized to zeros, formula `rms_norm(x) * (1 + w)` — used by Qwen3.5 and Gemma

## Adding a New Model

1. Create `sgl_mindspore/models/<arch>.py` following the pattern above
2. Add new layer primitives to `sgl_mindspore/layers/` if needed, and export from `layers/__init__.py`
3. Check hardware-specific requirements (bfloat16, NZ format, 310P patch)
4. Run `pre-commit run --files <changed files>` before committing

## Environment Variables (Runtime)

| Variable | Purpose |
|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | NPU device index |
| `MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST` | Disable specific Ascend kernels (e.g., `FlashAttentionScore,PagedAttention`) |
| `MS_DISABLE_INTERNAL_KERNELS_LIST` | Disable internal kernels (e.g., `RmsNorm`) |
| `MS_ENABLE_INTERNAL_BOOST` | Set to `off` for 310P |
| `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` | Set to `python` to avoid protobuf mismatch |

## Dependencies

- MindSpore 2.8.0 (or 2.7.1 nightly for CANN 8.3)
- CANN 8.5 (toolkit + kernels + nnal)
- sgl-kernel-npu (Ascend-specific AOT kernels)
- SGLang (installed from source with `[all_npu]` extras)

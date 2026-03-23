# AGENTS.md

This guide is for agentic coding assistants working in the sgl-mindspore repository.

## Build, Lint, and Test Commands

```bash
# Run pre-commit hooks on all files
pre-commit run --all-files

# Run pre-commit on specific files
pre-commit run --files <file1> <file2>

# Run individual linters/formatters
ruff check .
ruff check --select=F401,F821 .
black .
isort .

# Format code
black sgl_mindspore/
isort sgl_mindspore/

# Check for code quality issues
codespell

# Format C++/CUDA files
clang-format --style=file --verbose <file>
```

Note: This repo does not contain test files. Tests are located in the SGLang main repository.

## Code Style Guidelines

### File Headers
All Python files must include SPDX license headers:
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
```

### Import Ordering
Follow the order defined in `.isort.cfg` (black profile):
1. Standard library imports
2. Third-party imports
3. First-party imports (`sgl_mindspore`)

Use `isort` to auto-format:
```python
import logging
import os
from typing import Iterable, Optional

import mindspore as ms
import torch
from mindspore import Tensor, nn, ops

from sgl_mindspore.layers import ColParallelLinear, RowParallelLinear
```

### Type Hints
Always include type hints for function signatures and return types:
```python
def construct(self, input: Tensor) -> Tuple[Tensor, bool]:
    ...
```

Use `Optional` for nullable parameters and `Union` for multiple types:
```python
def __init__(
    self,
    input_size: int,
    output_size: int,
    bias: bool = True,
    param_dtype: Optional[ms.dtype] = None,
    quant_config: Optional[QuantizationConfig] = None,
) -> None:
    ...
```

### Naming Conventions
- Classes: `PascalCase` (e.g., `LlamaMLP`, `ColParallelLinear`)
- Functions/methods: `snake_case` (e.g., `construct`, `weight_load`, `add_prefix`)
- Constants: `UPPER_CASE` or `UPPER_SNAKE_CASE`
- Parameters/variables: `snake_case`
- Modules: `snake_case`

### MindSpore-Specific Patterns

#### Model Implementation
All models must:
1. Inherit from `MindSporeModelBase` (not `ms.nn.Cell`)
2. Implement `construct(self, **model_inputs) -> Tensor`
3. Implement `load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]])`
4. End with `EntryClass = <ModelClass>` for dynamic registration

```python
class MyForCausalLM(MindSporeModelBase):
    def construct(self, **model_inputs) -> Tensor:
        ...

    def load_weights(self, weights):
        ...

EntryClass = MyForCausalLM
```

#### Weight Loading
Use `prefix` parameter to construct weight paths:
```python
self.linear = ColParallelLinear(
    input_size=hidden_size,
    output_size=output_size,
    prefix=add_prefix("linear", prefix),
)
```

Use `self.parameters_dict()` to get parameters by full path:
```python
for name, loaded_weight in weights:
    param = self.parameters_dict().get(name)
    if param is not None:
        param.set_data(tensor_torch2ms(loaded_weight))
```

#### Tensor Parallelism
- Use `get_attention_tp_rank()` / `get_attention_tp_size()` for attention-side TP
- Use `get_tensor_model_parallel_world_size()` for global TP
- `ColParallelLinear`: shards output dimension
- `RowParallelLinear`: shards input dimension

#### 310P Hardware Specifics
- Convert bfloat16 to float16 (310P does not support bfloat16)
- Cast weights to NZ format: `format_cast(param, "nz")`
- Set environment variable: `MS_ENABLE_INTERNAL_BOOST=off`
- Use `is_310p()` helper from `sgl_mindspore.utils`

### Logging
Use standard Python logging pattern:
```python
import logging

logger = logging.getLogger(__name__)
```

### Error Handling
Use standard Python exceptions. Raise `ValueError` for invalid arguments and `NotImplementedError` for abstract methods:
```python
raise ValueError(f"shard_dim:{shard_dim} is not supported.")
raise NotImplementedError()
```

### No Comments
Do not add comments to code unless explicitly requested.

### MindSpore vs PyTorch API
- MindSpore uses `construct()` instead of PyTorch's `forward()`
- Parameters are `ms.Parameter` objects
- Use `ms.ops.MatMul` for matrix multiplication
- Convert tensors with `tensor_torch2ms()` and `tensor_ms2torch()` utilities

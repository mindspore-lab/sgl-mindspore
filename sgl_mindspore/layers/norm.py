# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
from typing import Optional, Tuple, Type, Union

from mindspore import Parameter, Tensor, dtype, mint, nn, ops
from mindspore.ops.auto_generate import add_rms_norm, rms_norm


class GemmaRMSNorm(nn.Cell):
    """RMSNorm variant used in Gemma/Qwen3.5: output = rms_norm(x) * (1 + weight)."""

    def __init__(
        self, norm_dim: int, eps: float, param_dtype: Optional[Type], prefix: str = ""
    ) -> None:
        super().__init__()
        self.weight = Parameter(mint.zeros(norm_dim, dtype=param_dtype))
        self.eps = eps

    def construct(
        self, x: Tensor, residual: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        gamma = self.weight + 1.0
        if residual is not None:
            output, _, residual = add_rms_norm(
                x1=x, x2=residual, gamma=gamma, epsilon=self.eps
            )
            return output, residual
        else:
            output = rms_norm(x=x, gamma=gamma, epsilon=self.eps)[0]
            return output


class RMSNorm(nn.Cell):
    def __init__(
        self, norm_dim: int, eps: float, param_dtype: Optional[Type], prefix: str = ""
    ) -> None:
        super().__init__()

        self.weight = Parameter(mint.ones(norm_dim, dtype=param_dtype))
        self.eps = eps

    def construct(
        self, x: Tensor, residual: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if residual is not None:
            output, _, residual = add_rms_norm(
                x1=x, x2=residual, gamma=self.weight, epsilon=self.eps
            )
            return output, residual
        else:
            output = rms_norm(x=x, gamma=self.weight, epsilon=self.eps)[0]
            return output

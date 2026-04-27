# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.utils import get_moe_weight_sizes
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz
from sglang.srt.layers.quantization.quark.schemes import QuarkMoEScheme
from sglang.srt.layers.quantization.utils import all_close_1d
from sglang.srt.utils import get_bool_env_var, is_gfx95_supported, is_hip, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

__all__ = ["QuarkW4A8MXFp4MoE"]

_is_fp8_fnuz = is_fp8_fnuz()
_is_shuffle_moe_mxfp4 = is_gfx95_supported()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _is_hip:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

OCP_MX_BLOCK_SIZE = 32


class QuarkW4A8MXFp4MoE(QuarkMoEScheme):
    """MoE scheme for MXFP4 weights with FP8 per-tensor static activations.

    Weights are stored in MXFP4 (fp4 elements packed as uint8) with e8m0
    block scales (group_size=32). Activations are quantized to fp8_e4m3
    with static per-tensor scales stored in the checkpoint.
    """

    def __init__(self, weight_config: dict[str, Any], input_config: dict[str, Any]):
        self.weight_quant = weight_config
        self.input_quant = input_config

        weight_qscheme = self.weight_quant.get("qscheme")
        if weight_qscheme != "per_group":
            raise ValueError(
                "For W4A8 MXFP4 MoE, weights must use per-group scales. "
                f"Found {weight_qscheme}"
            )

        self.is_static_input_scheme: bool = False
        if input_config is not None:
            self.is_static_input_scheme = not input_config.get("is_dynamic")

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes(
            intermediate_size_per_partition,
            is_aiter_moe=_is_hip,
            is_concat=True,
            is_packed=True,
        )

        extra_weight_attrs.update(
            {
                "quant_method": FusedMoeWeightScaleSupported.BLOCK.value,
                "weight_padded": weight_padded,
            },
        )

        params_dtype = torch.uint8

        # WEIGHTS (MXFP4 packed as uint8)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size // 2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                w2_down_dim,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT SCALES (e8m0 block scales, group_size=32)
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                w13_up_dim,
                hidden_size // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                (w2_down_dim * 2) // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        # INPUT SCALES (FP8 per-tensor, one per expert)
        if self.is_static_input_scheme:
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Process static FP8 input scales: collapse per-expert to single max
        if self.is_static_input_scheme:
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                layer.w2_input_scale
            ):
                logger.warning(
                    "Found input_scales that are not equal for "
                    "W4A8 MoE layer. Using the maximum across experts "
                    "for each layer."
                )
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )

            if _is_fp8_fnuz:
                _, _, w13_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    torch.empty_like(layer.w13_weight, dtype=torch.float8_e4m3fn),
                    torch.empty_like(
                        layer.w13_weight_scale, dtype=layer.w13_weight_scale.dtype
                    ),
                    layer.w13_input_scale,
                )
                _, _, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    torch.empty_like(layer.w2_weight, dtype=torch.float8_e4m3fn),
                    torch.empty_like(
                        layer.w2_weight_scale, dtype=layer.w2_weight_scale.dtype
                    ),
                    layer.w2_input_scale,
                )
                if w13_input_scale is not None:
                    layer.w13_input_scale = torch.nn.Parameter(
                        w13_input_scale, requires_grad=False
                    )
                if w2_input_scale is not None:
                    layer.w2_input_scale = torch.nn.Parameter(
                        w2_input_scale, requires_grad=False
                    )

        # CK MoE kernels require hidden_size to be a multiple of 128.
        # If not, skip shuffling and use the torch fallback path.
        hidden_size = layer.w13_weight.shape[2] * 2
        self._use_ck_moe = (hidden_size % 128 == 0)

        if self._use_ck_moe:
            s0, s1, _ = layer.w13_weight_scale.shape
            w13_weight_scale = layer.w13_weight_scale.view(s0 * s1, -1)
            w13_weight_scale = e8m0_shuffle(w13_weight_scale)
            layer.w13_weight_scale.data = w13_weight_scale.view(s0, s1, -1)

            s0, s1, _ = layer.w2_weight_scale.shape
            w2_weight_scale = layer.w2_weight_scale.view(s0 * s1, -1)
            w2_weight_scale = e8m0_shuffle(w2_weight_scale)
            layer.w2_weight_scale.data = w2_weight_scale.view(s0, s1, -1)

            if _is_shuffle_moe_mxfp4:
                layer.w13_weight.data = shuffle_weight(
                    layer.w13_weight.contiguous(), (16, 16)
                )
                layer.w2_weight.data = shuffle_weight(
                    layer.w2_weight.contiguous(), (16, 16)
                )
                layer.w13_weight.is_shuffled = True
                layer.w2_weight.is_shuffled = True
        else:
            logger.info(
                "hidden_size=%d not divisible by 128, using torch fallback MoE",
                hidden_size,
            )

        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"weight_dtype": torch.float4_e2m1fn_x2})

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        moe_runner_config = self.moe_runner_config
        topk_weights, topk_ids, _ = topk_output
        if _is_hip:
            topk_weights = topk_weights.to(torch.float32)

        if hasattr(torch, "float4_e2m1fn_x2"):
            w13_weight = layer.w13_weight.view(torch.float4_e2m1fn_x2)
            w2_weight = layer.w2_weight.view(torch.float4_e2m1fn_x2)
        else:
            w13_weight = layer.w13_weight
            w2_weight = layer.w2_weight

        if self._use_ck_moe:
            if hasattr(layer.w13_weight, "is_shuffled"):
                w13_weight.is_shuffled = True
                w2_weight.is_shuffled = True

            output = fused_moe(
                x,
                w13_weight,
                w2_weight,
                topk_weights,
                topk_ids,
                quant_type=QuantType.per_Token,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=None,
                a2_scale=None,
                activation=(
                    ActivationType.Silu
                    if moe_runner_config.activation == "silu"
                    else ActivationType.Gelu
                ),
                doweight_stage1=False,
                expert_mask=layer.dispatcher.expert_mask_gpu,
            )
        else:
            output = self._torch_fallback_moe(
                x,
                w13_weight,
                w2_weight,
                topk_weights,
                topk_ids,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                moe_runner_config.activation,
            )
        return StandardCombineInput(hidden_states=output)

    @staticmethod
    def _dequant_mxfp4(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize MXFP4 weights with e8m0 block scales to bf16."""
        from aiter.utility import fp4_utils

        w_f = fp4_utils.mxfp4_to_f32(w)
        s_f = fp4_utils.e8m0_to_f32(scale)
        E, N, K_packed = w_f.shape
        K_blocks = s_f.shape[2]
        block_size = K_packed // K_blocks
        w_f = w_f.view(E, N, K_blocks, block_size)
        w_f = w_f * s_f.unsqueeze(-1)
        return w_f.view(E, N, K_packed).to(torch.bfloat16)

    @staticmethod
    def _torch_fallback_moe(
        x: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        activation: str,
    ) -> torch.Tensor:
        """Torch-native fallback for when CK GEMM dims are unsupported."""
        if not hasattr(QuarkW4A8MXFp4MoE, "_fallback_debug_printed"):
            QuarkW4A8MXFp4MoE._fallback_debug_printed = True
            logger.info(
                "Torch fallback MoE shapes: x=%s w13=%s w2=%s w13_scale=%s w2_scale=%s "
                "topk_weights=%s topk_ids=%s",
                x.shape, w13.shape, w2.shape, w13_scale.shape, w2_scale.shape,
                topk_weights.shape, topk_ids.shape,
            )
        B, D = x.shape
        topk = topk_ids.shape[1]
        E = w13.shape[0]

        w13_bf = QuarkW4A8MXFp4MoE._dequant_mxfp4(w13, w13_scale)
        w2_bf = QuarkW4A8MXFp4MoE._dequant_mxfp4(w2, w2_scale)

        x_bf = x.to(torch.bfloat16)
        out = torch.zeros(B, D, dtype=torch.bfloat16, device=x.device)

        for i in range(topk):
            expert_ids = topk_ids[:, i]
            weights = topk_weights[:, i]
            for e_id in range(E):
                mask = expert_ids == e_id
                if not mask.any():
                    continue
                tokens = x_bf[mask]
                h = tokens @ w13_bf[e_id].T
                gate, up = h.split(h.shape[-1] // 2, dim=-1)
                if activation == "silu":
                    h = torch.nn.functional.silu(gate) * up
                else:
                    h = torch.nn.functional.gelu(gate) * up
                h = h @ w2_bf[e_id].T
                out[mask] += h * weights[mask].unsqueeze(-1).to(torch.bfloat16)

        return out

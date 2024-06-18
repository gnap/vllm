import enum
from enum import Enum
from typing import Any, Dict, List, Optional

from tllm_qmm import W4A16, W4A8_FP8
from tllm_qmm import WeightOnlyGroupwiseQuantGEMM
import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs


class TLLMAWQConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits."
            )
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (
            f"AWQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point})"
        )

    def get_name(self) -> str:
        return "awq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    def get_min_capability(self) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        return cls(weight_bits, group_size, zero_point)

    def get_quant_method(
        self, layer: torch.nn.Module
    ) -> Optional["TLLMAWQLinearMethod"]:
        if isinstance(layer, LinearBase):
            return TLLMAWQLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


class TLLMPluginState(Enum):

    UNINITIALIZED = enum.auto()
    READY = enum.auto()


class TLLMAWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: TLLMAWQConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        qweight = Parameter(
            torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )
        qzeros = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros,
            {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )
        scales = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "input_dim": 0,
                "output_dim": 1,
            },
        )

        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qzeros", qzeros)
        set_weight_attrs(qzeros, extra_weight_attrs)
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)

        self.fp8_alpha = torch.tensor([1], dtype=torch.float32)
        self.pre_quant_scale = self.fp8_alpha.reciprocal().half()
        self.plugin_state = TLLMPluginState.UNINITIALIZED

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        if self.plugin_state == TLLMPluginState.UNINITIALIZED:
            self.tllm_matmul = WeightOnlyGroupwiseQuantGEMM(
                W4A16,
                1,
                reshaped_x.shape[0],
                qweight.shape[0],
                qweight.shape[1] * pack_factor,
                self.quant_config.group_size,
                False,
            )
            qweight, qzeros = self.tllm_matmul.preprocess_weights(
                layer.qweight, layer.qzeros, layer.scales
            )
            layer.qweight = Parameter(qweight.view(torch.int32), requires_grad=False)
            layer.qzeros = Parameter(qzeros, requires_grad=False)
            self.plugin_state = TLLMPluginState.READY

        out = self.tllm_matmul.forward(
            reshaped_x,
            torch.empty_like(reshaped_x),
            qweight.view(torch.half),
            scales,
            qzeros,
            torch.empty_like(reshaped_x),
        )
        return out.reshape(out_shape)


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    # Calculate the scale as dtype max divided by absmax
    """
    if x.abs().max() > finfo.max:
        scale_orig = (finfo.max - 0) / x.abs().max()
    else:
        scale_orig = torch.tensor(1.0, dtype=x.dtype, device=x.device)
    """
    scale_orig = (finfo.max - 0) / x.abs().max()
    scale = scale_orig.clamp(min=1e-12, max=1)
    # scale = finfo.max / x.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    # return x_scl_sat.to(dtype), scale.float().reciprocal()
    return x_scl_sat.to(dtype), scale.float().reciprocal()


class TLLMAWQFP8LinearMethod(TLLMAWQLinearMethod):

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        qweight = Parameter(
            torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )
        qzeros = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros,
            {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )
        scales = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "input_dim": 0,
                "output_dim": 1,
            },
        )
        fp8_alpha = Parameter(
            torch.empty(
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qzeros", qzeros)
        set_weight_attrs(qzeros, extra_weight_attrs)
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)
        layer.register_parameter("fp8_alpha", fp8_alpha)
        set_weight_attrs(fp8_alpha, extra_weight_attrs)

        self.plugin_state = TLLMPluginState.UNINITIALIZED

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        """

        # perform the float8 matmul
        out_scaled, _ = torch._scaled_mm(x_f8, w_f8, out_dtype=torch.float16,
                scale_a=x_inv_s , scale_b=w_inv_s)
        out_torch = torch.matmul(x, out_dq)
        """
        if self.plugin_state == TLLMPluginState.UNINITIALIZED:
            self.tllm_matmul = WeightOnlyGroupwiseQuantGEMM(
                W4A8_FP8,
                1,
                reshaped_x.shape[0],
                qweight.shape[0],
                qweight.shape[1] * pack_factor,
                self.quant_config.group_size,
                False,
            )
            # qweight, qzeros = self.tllm_matmul.preprocess_weights(
            #    layer.qweight, layer.qzeros, layer.scales
            # )
            out_dq = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
            out_cm = torch.empty(
                tuple(reversed(out_dq.shape)), dtype=out_dq.dtype, device=out_dq.device
            )
            out_cm.t()[:] = out_dq[:]

            # x_view = x.view((-1, x.size(-1))) # to 2-D matrix shape
            # dyamic act scale
            # x_f8, x_inv_s = to_float8(x_view, dtype=torch.float8_e4m3fn)
            # x_f8, x_inv_s = ops.scaled_fp8_quant(x_view)
            # if x_inv_s.item() > 1.0:
            # quant to fp8
            _, w_inv_s = ops.scaled_fp8_quant(out_cm.t())
            # print(x_view.abs().max().item(), x_inv_s.item(), out_cm.abs().max().item(), w_inv_s.item())
            # w_f8, w_inv_s = to_float8(out_cm.t())

            scales = (layer.scales.float() / w_inv_s).half()
            qweight, qzeros = self.tllm_matmul.preprocess_weights(
                layer.qweight,
                layer.qzeros,
                scales,
            )
            layer.qweight = Parameter(qweight.view(torch.int32), requires_grad=False)
            layer.qzeros = Parameter(qzeros, requires_grad=False)
            layer.scales = Parameter(scales, requires_grad=False)
            layer.fp8_alpha = Parameter(w_inv_s, requires_grad=False)
            self.plugin_state = TLLMPluginState.READY
        # fp8_alpha = to_float8(reshaped_x)[1]
        # _, fp8_alpha = to_float8(reshaped_x)
        x_fp8, x_inv_s = ops.scaled_fp8_quant(reshaped_x)
        # if x_inv_s.item() > 1.0:
        #    print(reshaped_x.abs().max().item(), x_inv_s.item())
        # fp8_alpha = self.fp8_alpha
        pre_quant_scale = x_inv_s.reciprocal().half().repeat(reshaped_x.shape)
        # pre_quant_scale = self.pre_quant_scale.repeat(reshaped_x.shape)
        # print(fp8_alpha)
        # print(pre_quant_scale)

        out = self.tllm_matmul.forward(
            x_fp8.to(torch.half),
            torch.ones_like(pre_quant_scale),
            # reshaped_x,
            # pre_quant_scale,
            qweight.view(torch.half),
            scales,
            qzeros,
            x_inv_s * layer.fp8_alpha,
        )
        # print(x_view.abs().max().item(), x_inv_s.item(), out_cm.abs().max().item(), w_inv_s.item(), (out - out_scaled).abs().max().item())
        # print((out - out_torch).abs().max().item() - (out_scaled - out_torch).abs().max().item(), "tllm:", (out - out_torch).abs().max().item(), "scaled:", (out_scaled - out_torch).abs().max().item(), x_view.abs().max().item(), x_inv_s.reciprocal().item(), out_cm.abs().max().item(), w_inv_s.reciprocal().item())

        return out.reshape(out_shape)

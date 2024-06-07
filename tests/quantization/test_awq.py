# from py_binding import woq_groupwise_matmul, WeightOnlyGroupwiseQuantMatmul
import torch

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

from vllm import _custom_ops as ops


def woq_assert_near_eq(ref, act, wTypeId):
    # match the scale in cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp
    if wTypeId == 1:
        bits_in_type = 8
    else:
        bits_in_type = 4
    quant_range_scale = 1.0 / float(1 << (bits_in_type - 1))

    max_val = torch.max(abs(ref)).item()
    atol = (max_val * quant_range_scale) * 1.5  # allow for rounding
    torch.testing.assert_close(ref, act, atol=atol, rtol=1e-7)


input_rows = 8192
in_features = 49152
out_features = 8192
w_bit = 4
group_size = 128

MAX_INT32 = 0x7FFFFFFF
MIN_INT32 = -MAX_INT32 - 1

qweight = torch.randint(
    # -(2**31),
    # 2**31,
    MIN_INT32,
    MAX_INT32,
    (in_features, out_features // (32 // w_bit)),
    dtype=torch.int32,
    device="cuda",
)
inputs = torch.randn((input_rows, in_features), dtype=torch.half, device="cuda")

qzeros = torch.randint(
    MIN_INT32,
    MAX_INT32,
    (in_features // group_size, out_features // (32 // w_bit)),
    dtype=torch.int32,
    device="cuda",
)

scales = torch.randn(
    (in_features // group_size, out_features),
    dtype=torch.float16,
    device="cuda",
)
pre_quant_scale = torch.rand(1, in_features, dtype=torch.half, device="cuda")
print(pre_quant_scale.shape)
# fp8_alpha = torch.rand(1, dtype=torch.float32, device="cuda").reciprocal()
fp8_alpha = torch.tensor([4], dtype=torch.float32, device="cuda")
pre_quant_scale = fp8_alpha.reciprocal().half().repeat(1, in_features)
print(pre_quant_scale.shape)
# fp8_alpha = torch.ones_like(fp8_alpha)
# pre_quant_scale = torch.ones_like(pre_quant_scale)

print(fp8_alpha.half() * pre_quant_scale)

# inputs = torch.ones([input_rows], device="cuda").to(torch.float16).diag()
# qweight = torch.zeros_like(qweight)
# print(inputs.shape)
# scales = torch.ones_like(scales).to(torch.float16)
# qzeros = torch.zeros_like(qzeros)

with torch.no_grad():

    # qiweight, izeros = awq_to_tllm(qweight, qzeros, scales, 14)

    tinputs = torch.mul(
        inputs * fp8_alpha.half(), pre_quant_scale.repeat(input_rows, 1)
    )
    for i in range(10):
        cuda_out = ops.awq_gemm(inputs, qweight, scales, qzeros, 4)
    # awq dequantize
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    """
    torch_dq = dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)
    torch_out = torch.matmul(
        tinputs,
        torch_dq,
    )
    """
    cuda_out = ops.awq_gemm(inputs, qweight, scales, qzeros, 4)
    stop_event.record()
    torch.cuda.synchronize()
    print(f"cuda time: {start_event.elapsed_time(stop_event)}ms")

    # torch_out = torch.matmul(inputs, unpack(iweight, direction="column").to(torch.half))
    """
    print(cuda_out.shape, torch_out.shape)
    print("torch_dq:", torch_dq)
    print("cuda_out:", cuda_out)
    print("torch_out:", torch_out)
    print((cuda_out - torch_out).max())
    # loss = (cuda_out - torch_out).float().pow(2).mean().item()
    # assert loss < 1e-1, loss
    woq_assert_near_eq(torch_out, cuda_out, 2)
    """

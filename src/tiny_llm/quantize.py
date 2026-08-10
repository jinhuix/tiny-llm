from typing import Any

import mlx.core as mx

from extensions import tiny_llm_ext


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w.astype(mx.bfloat16)


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
        use_simdgroup_matmul: bool = False,
        use_simdgroup_matvec: bool = True,
        use_split_k_matmul: bool = False,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight
        self.use_simdgroup_matmul = use_simdgroup_matmul    # 针对大 M 的 tiled SIMD matmul
        self.use_simdgroup_matvec = use_simdgroup_matvec    # 小输入是否使用 SIMD matvec
        self.use_split_k_matmul = use_split_k_matmul        # 沿输入维度拆分 reduction

    @staticmethod
    def from_mlx_layer(
        mlx_layer: Any,
        use_simdgroup_matmul: bool = False,
        use_simdgroup_matvec: bool = True,
        use_split_k_matmul: bool = False,
    ) -> "QuantizedWeights":
        biases = mlx_layer.biases
        return QuantizedWeights(
            scales=mlx_layer.scales.astype(mx.bfloat16),
            biases=None if biases is None else biases.astype(mx.bfloat16),
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
            use_simdgroup_matmul=use_simdgroup_matmul,
            use_simdgroup_matvec=use_simdgroup_matvec,
            use_split_k_matmul=use_split_k_matmul,
        )


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,                      # (D_out, D_in / 128)
    a: mx.array,                    # (..., D_in)
    b: mx.array,                    # (D_out, D_in / 8)
    transpose_b: bool = False,
    use_simdgroup: bool = False,
    use_split_k: bool = False,
) -> mx.array:
    *B, D_in = a.shape
    a_2d = a.reshape(-1, D_in)      # (M, D_in) 展平成二维矩阵

    result = tiny_llm_ext.quantized_matmul(
        mx.contiguous(scales),      # 转成连续内存
        mx.contiguous(biases),
        group_size,
        bits,
        mx.contiguous(a_2d),
        mx.contiguous(b),
        transpose_b,
        use_simdgroup,
        use_split_k,
    )                               # (M, D_out)
    return result.reshape(*B, -1)   # (..., D_out)


def dequantize_weights(
    weight: mx.array,            # (..., D_in/P), packed uint32
    scales: mx.array,            # (..., D_in/G)
    biases: mx.array | None,     # (..., D_in/G)
    group_size: int,            # G, normally 128
    bits: int,                  # normally 4, so P = 8
) -> mx.array:
    """
    packed (..., D_in/P), uint32 -> dense (..., D_in), BF16
    value[j] = q[j] * scale[j // G] + bias[j // G]
    """
    if bits <= 0 or 32 % bits != 0:
        raise ValueError("bits must divide a 32-bit packed weight")

    values_per_word = 32 // bits                               # P; 8 for 4-bit
    shifts = mx.arange(0, 32, bits, dtype=mx.uint32)           # [0, 4, ..., 28]
    values = (weight[..., None] >> shifts) & ((1 << bits) - 1) # (..., D_in/P, P)
    values = values.reshape(                                   # (..., D_in)
        *weight.shape[:-1], weight.shape[-1] * values_per_word
        ).astype(mx.float32)                                   # FP32 dequantization arithmetic

    # scales[..., g] and biases[..., g] are shared by G=group_size
    # consecutive dense values in quantization group g.
    scales = mx.repeat(scales, group_size, axis=-1).astype(mx.float32)  # (..., D_in/P, P) -> (..., D_in)
    if biases is None:
        return (values * scales).astype(mx.bfloat16)                    # (..., D_in)
    biases = mx.repeat(biases, group_size, axis=-1).astype(mx.float32)  # (..., D_in)
    return (values * scales + biases).astype(mx.bfloat16)               # (..., D_in)


def quantized_matvec_custom(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    return quantized_matmul(
        scales,
        biases,
        group_size,
        bits,
        a,
        b,
        transpose_b,
        use_simdgroup=True,
    )


def quantized_matmul_vanilla(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    return quantized_matmul(
        scales,
        biases,
        group_size,
        bits,
        a,
        b,
        transpose_b,
        use_simdgroup=False,
    )


def quantized_linear(
    x: mx.array,                         # (..., D_in)
    w: QuantizedWeights,                 # (D_out, D_in)
    bias: mx.array | None = None,        # (D_out,)
) -> mx.array:
    """Y(..., D_out) = X(..., D_in) @ W(D_out, D_in).T + bias."""
    M = 1
    for size in x.shape[:-1]:
        M *= size
    operation = (
        quantized_matvec_custom
        if M <= 8 and w.use_simdgroup_matvec
        else quantized_matmul
    )
    result = operation(
        w.scales,
        w.biases,
        w.group_size,
        w.bits,
        x,
        w.weight,
        transpose_b=True,
    )
    return result + bias if bias is not None else result

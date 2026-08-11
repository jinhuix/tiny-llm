import mlx.core as mx
from extensions import tiny_llm_ext


class FastRMSNorm:
    """Fused RMSNorm over the last tensor dimension.

    Shapes:
        x:      (..., D)
        weight: (D,)
        output: (..., D)

    y = x / sqrt(mean(x², axis=-1) + eps) · weight
    The sum and normalization are evaluated in FP32.
    """

    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        if dim <= 0:
            raise ValueError("FastRMSNorm dim must be positive")
        if weight.ndim != 1 or weight.shape[0] != dim:
            raise ValueError(f"FastRMSNorm weight must have shape ({dim},)")
        if eps <= 0:
            raise ValueError("FastRMSNorm eps must be positive")
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 0 or x.shape[-1] != self.dim:
            raise ValueError(
                f"FastRMSNorm expected x[..., {self.dim}], got {x.shape}"
            )
        if x.dtype != mx.bfloat16:
            raise ValueError("FastRMSNorm expects BF16 model activations")

        weight = mx.contiguous(self.weight.astype(x.dtype))
        return tiny_llm_ext.rms_norm(mx.contiguous(x), weight, self.eps)


class FastRoPE:
    """Fused rotary position encoding for model-native ``(B, L, H, D)``.

        theta[m,i] = m * base ** (-i / (dims / 2))
        real' = real*cos(theta) - imag*sin(theta)
        imag' = real*sin(theta) + imag*cos(theta)

    Non-traditional pairing (Qwen's default) pairs: x[..., i + dims/2]
    Traditional pairing: x[..., 2*i] / x[..., 2*i+1]
    """

    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        if dims <= 0 or dims % 2 != 0:
            raise ValueError("FastRoPE dims must be positive and even")
        if seq_len <= 0:
            raise ValueError("FastRoPE seq_len must be positive")
        if base <= 0:
            raise ValueError("FastRoPE base must be positive")
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

    def __call__(self, x: mx.array, offset: int | list[int] | mx.array = 0) -> mx.array:
        if x.ndim != 4:
            raise ValueError(f"FastRoPE expects x(B,L,H,D), got {x.shape}")
        if x.dtype != mx.bfloat16:
            raise ValueError("FastRoPE expects BF16 model activations")

        B, _, _, D = x.shape
        if self.dims > D:
            raise ValueError(
                f"FastRoPE dims={self.dims} exceeds head dimension D={D}"
            )

        if isinstance(offset, int):
            offsets = mx.full((B,), offset, dtype=mx.int32)
        elif isinstance(offset, list):
            if len(offset) != B:
                raise ValueError("FastRoPE needs one offset per batch row")
            offsets = mx.array(offset, dtype=mx.int32)
        elif offset.ndim == 0:
            offsets = mx.broadcast_to(offset.astype(mx.int32), (B,))
        elif offset.shape == (B,):
            offsets = offset.astype(mx.int32)
        else:
            raise ValueError("FastRoPE needs one offset per batch row")

        return tiny_llm_ext.rope(
            mx.contiguous(x),
            mx.contiguous(offsets),
            self.dims,
            float(self.base),
            self.traditional,
        )


def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """Fused ``SiLU(gate) * up`` with shape ``(..., D_ff) -> (..., D_ff)``.

    Each Metal thread computes one element:

        SiLU(g) * u = (g / (1 + exp(-g))) * u
    """
    if gate.shape != up.shape:
        raise ValueError("swiglu gate and up must have the same shape")
    if gate.dtype != up.dtype or gate.dtype != mx.bfloat16:
        raise ValueError("swiglu gate and up must both be BF16")
    return tiny_llm_ext.swiglu(mx.contiguous(gate), mx.contiguous(up))


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def decode_attention_custom(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass

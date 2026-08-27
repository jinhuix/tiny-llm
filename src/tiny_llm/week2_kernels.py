import mlx.core as mx
from extensions import tiny_llm_ext

from .basics import softmax


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
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return tiny_llm_ext.rms_norm(
            mx.contiguous(x), mx.contiguous(self.weight.astype(x.dtype)), self.eps
        )


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
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

    def __call__(self, x: mx.array, offset: int | list[int] | mx.array = 0) -> mx.array:
        B, _, _, D = x.shape
        if isinstance(offset, int):
            offset = mx.full((B,), offset, dtype=mx.int32)
        elif isinstance(offset, list):
            if len(offset) != B:
                raise ValueError("FastRoPE needs one offset per batch row")
            offset = mx.array(offset, dtype=mx.int32)
        elif offset.ndim == 0:
            offset = mx.broadcast_to(offset.astype(mx.int32), (B,))
        elif offset.shape != (B,):
            raise ValueError("FastRoPE needs one offset per batch row")

        return tiny_llm_ext.rope(
            mx.contiguous(x),
            mx.contiguous(offset.astype(mx.int32)),
            self.dims,
            float(self.base),
            self.traditional,
        )


def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    return tiny_llm_ext.swiglu(mx.contiguous(gate), mx.contiguous(up))


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    shape = query.shape
    *B, H_q, L, D = shape
    H, S, _ = key.shape[-3:]
    n_repeats = H_q // H

    q = query.reshape(-1, H, n_repeats, L, D)
    k = key.reshape(-1, H, 1, S, D)
    v = value.reshape(-1, H, 1, S, D)
    scores = mx.matmul(q, k.swapaxes(-2, -1)) * mx.array(scale, dtype=query.dtype)  # (*B, H, n_repeats, L, S)

    if mask is not None:
        if mask == "causal":
            causal= mx.tril(mx.ones((L, S)), k=S - L)
            scores = scores + mx.where(causal, 0, -mx.inf).astype(scores.dtype)
        else:
            scores = scores + mx.broadcast_to(mask, (*B, H_q, L, S)).reshape(-1, H, n_repeats, L, S)

    return mx.matmul(softmax(scores, axis=-1), v).reshape(shape)    # (*B, H_q, L, D)


def decode_attention_custom(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    B, H_q, L, D = query.shape
    _, H, S, _ = key.shape
    n_repeats = H_q // H

    q = mx.contiguous(query.reshape(B * H_q, L, D))
    k = mx.contiguous(key.reshape(B * H, S, D))
    v = mx.contiguous(value.reshape(B * H, S, D))

    is_causal = isinstance(mask, str) and mask == "causal"
    has_mask = isinstance(mask, mx.array)
    if has_mask:
        mask = mx.broadcast_to(mask, (B, H_q, L, S))
        mask = mx.contiguous(mask.astype(mx.float32).reshape(B * H_q, L, S))
    else:
        mask = mx.zeros((1,), dtype=mx.float32)

    result = tiny_llm_ext.decode_attention_custom(
        q, k, v, mask, is_causal, has_mask, H_q, H
    )
    return result.reshape(B, H_q, L, D)


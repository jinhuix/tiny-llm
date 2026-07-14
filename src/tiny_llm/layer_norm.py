import mlx.core as mx


class RMSNorm:
    """
    y = x / sqrt(mean(x², axis=-1) + eps) · weight
    """
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x32 = x.astype(mx.float32)
        norm = x32 * mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + self.eps)
        return norm.astype(dtype) * self.weight.astype(dtype)

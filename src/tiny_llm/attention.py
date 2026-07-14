import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,    # (N, H, L, d_k)
    key: mx.array,      # (N, H, S, d_k)
    value: mx.array,    # (N, H, S, d_k)    
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    ''' attention(q, k, v) = softmax(q @ k^T / sqrt(d_k)) @ v '''
    scale = mx.rsqrt(query.shape[-1]) if scale is None else scale
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    return mx.matmul(softmax(scores, axis=-1), value)   # (N, H, L, d_k)

class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo # (hidden_size, hidden_size)

    def _project(self, x: mx.array, w: mx.array) -> mx.array:
        # (N, L, hidden_size) -> (N, L, num_heads, head_dim) -> (N, num_heads, L, head_dim)
        N, L, _ = x.shape
        return linear(x, w).reshape(N, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape
        q = self._project(query, self.wq)   # (N, H, L, d_k)
        k = self._project(key, self.wk)
        v = self._project(value, self.wv)
        out = scaled_dot_product_attention_simple(q, k, v, scale=self.scale, mask=mask) # (N, H, L, d_k)
        out = out.transpose(0, 2, 1, 3).reshape(N, L, self.hidden_size)
        return linear(out, self.wo) # (N, L, hidden_size)

def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass

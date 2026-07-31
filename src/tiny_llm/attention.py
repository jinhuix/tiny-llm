import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,    # (B, H, L, D)
    key: mx.array,      # (B, H, S, D)
    value: mx.array,    # (B, H, S, D)
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    ''' attention(q, k, v) = softmax(q @ k^T / sqrt(D)) @ v '''
    scale = mx.rsqrt(query.shape[-1]) if scale is None else scale
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    return mx.matmul(softmax(scores, axis=-1), value)   # (B, H, L, D)

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
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo  # (E, E)

    def _project(self, x: mx.array, w: mx.array) -> mx.array:
        # (B, L, E) -> (B, L, H, D) -> (B, H, L, D)
        B, L, _ = x.shape
        return linear(x, w).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        B, L, _ = query.shape
        q = self._project(query, self.wq)   # (B, H, L, D)
        k = self._project(key, self.wk)
        v = self._project(value, self.wv)
        out = scaled_dot_product_attention_simple(q, k, v, scale=self.scale, mask=mask)  # (B, H, L, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return linear(out, self.wo)  # (B, L, E)

def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    """
    Right-aligned causal mask of shape (L, S): 0 where attendable, -inf otherwise.
    Diagonal offset = S - L, so the last L tokens of K/V are aligned with Q.
        e.g. L=3, S=5  ->  [[0, 0, 0, -inf, -inf],
                            [0, 0, 0,    0, -inf],
                            [0, 0, 0,    0,    0]]
    """
    keep = mx.tril(mx.ones((L, S)), k=S - L)
    return mx.where(keep, 0, -mx.inf).astype(dtype)

def scaled_dot_product_attention_grouped(
    query: mx.array,    # (*B, H_q, L, D)
    key: mx.array,      # (*B, H,   S, D)
    value: mx.array,    # (*B, H,   S, D)
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    shape = query.shape
    *B, H_q, L, D = shape
    H, S, _ = key.shape[-3:]
    n_repeats = H_q // H

    q = query.reshape(-1, H, n_repeats, L, D)
    k = key.reshape(-1, H, 1, S, D)
    v = value.reshape(-1, H, 1, S, D)
    factor = (mx.rsqrt(D) if scale is None else mx.array(scale)).astype(query.dtype)
    scores = mx.matmul(q, k.swapaxes(-2, -1)) * factor  # (*B, H, n_repeats, L, S)

    if mask is not None:
        if mask == "causal":
            scores = scores + causal_mask(L, S, scores.dtype)
        else:
            scores = scores + mx.broadcast_to(mask, (*B, H_q, L, S)).reshape(-1, H, n_repeats, L, S)

    return mx.matmul(softmax(scores, axis=-1), v).reshape(shape)    # (*B, H_q, L, D)

def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass

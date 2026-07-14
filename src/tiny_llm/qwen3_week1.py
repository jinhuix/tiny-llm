import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
    ):
        assert num_heads % num_kv_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo
        self.q_norm, self.k_norm = q_norm, k_norm
        self.rope = RoPE(head_dim, max_seq_len, theta, traditional=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        N, L, _ = x.shape
        q = linear(x, self.wq).reshape(N, L, self.num_heads,    self.head_dim)
        k = linear(x, self.wk).reshape(N, L, self.num_kv_heads, self.head_dim)
        v = linear(x, self.wv).reshape(N, L, self.num_kv_heads, self.head_dim)

        # RMSNorm, RoPE
        q = self.rope(mx.fast.rms_norm(q, self.q_norm, self.rms_norm_eps), offset=slice(0, L))
        k = self.rope(mx.fast.rms_norm(k, self.k_norm, self.rms_norm_eps), offset=slice(0, L))

        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))  # (B, H_*, L, D)
        out = scaled_dot_product_attention_grouped(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
            scale=mx.rsqrt(self.head_dim), mask=mask,
        ).astype(x.dtype)
        out = out.transpose(0, 2, 1, 3).reshape(N, L, self.num_heads * self.head_dim)
        return linear(out, self.wo) # (B, L, hidden_size)

class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass

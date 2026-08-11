"""
Shape notation used throughout this package:
    B  = batch size (may be multiple dims, written *B when arbitrary)
    L  = query / current sequence length
    S  = key-value / context sequence length
    E  = hidden_size (model embedding dimension)
    H  = number of attention heads (H_q for query, H_kv for key/value)
    D  = head_dim (per-head dimension, E // H)
    I  = intermediate_size (MLP inner dimension)
    V  = vocab_size
    G  = num GQA groups (H_q // H_kv)
"""

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
        wq: mx.array,   # (H_q * D, E)
        wk: mx.array,   # (H_kv * D, E)
        wv: mx.array,   # (H_kv * D, E)
        wo: mx.array,   # (E, H_q * D)
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
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo
        self.q_norm = RMSNorm(head_dim, q_norm, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, k_norm, eps=rms_norm_eps)
        self.rope = RoPE(head_dim, max_seq_len, theta, traditional=False)

    def __call__(
        self,
        x: mx.array,  # (B, L, E)
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        q = linear(x, self.wq).reshape(B, L, self.num_heads,    self.head_dim)   # (B, L, H_q, D)
        k = linear(x, self.wk).reshape(B, L, self.num_kv_heads, self.head_dim)   # (B, L, H_kv, D)
        v = linear(x, self.wv).reshape(B, L, self.num_kv_heads, self.head_dim)   # (B, L, H_kv, D)

        q = self.rope(self.q_norm(q), offset=slice(0, L))
        k = self.rope(self.k_norm(k), offset=slice(0, L))

        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))  # (B, H_*, L, D)
        out = scaled_dot_product_attention_grouped(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
            scale=mx.rsqrt(self.head_dim), mask=mask,
        ).astype(x.dtype)                                         # (B, H_q, L, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        return linear(out, self.wo)  # (B, L, E)


class Qwen3MLP:
    """
    SwiGLU FFN:  out = ( SiLU(x · W_gateᵀ) ⊙ (x · W_upᵀ) ) · W_downᵀ
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,   # (I, E)
        w_up: mx.array,     # (I, E)
        w_down: mx.array,   # (E, I)
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate, self.w_up, self.w_down = w_gate, w_up, w_down

    def __call__(self, x: mx.array) -> mx.array:
        # (B, L, E) → (B, L, I) → (B, L, E)
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)


class Qwen3TransformerBlock:
    """
    Pre-norm transformer block (Qwen3):
        h = x + Attn( RMSNorm₁(x) )
        y = h + MLP ( RMSNorm₂(h) )
    """

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
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, eps=rms_norm_eps)
        self.self_attn = Qwen3MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            wq=wq, wk=wk, wv=wv, wo=wo,
            q_norm=q_norm, k_norm=k_norm,
            max_seq_len=max_seq_len, theta=theta, rms_norm_eps=rms_norm_eps,
        )
        self.mlp = Qwen3MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)

    def __call__(
        self,
        x: mx.array,    # (B, L, E)
        mask: mx.array | str | None = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask)
        return h + self.mlp(self.post_attention_layernorm(h))   # (B, L, E)


class Qwen3ModelWeek1:
    """
    Full Qwen3 forward path:
        tokens → Embedding → h → [TransformerBlock] × L → RMSNorm → lm_head → logits
    """

    def __init__(self, mlx_model: Any):
        args = mlx_model.args
        self.num_hidden_layers = args.num_hidden_layers
        self.embedding = Embedding(
            vocab_size=args.vocab_size,
            embedding_dim=args.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens),
        )
        self.layers_inner = [
            Qwen3TransformerBlock(
                num_attention_heads=args.num_attention_heads,
                num_kv_heads=args.num_key_value_heads,
                hidden_size=args.hidden_size,
                head_dim=args.head_dim,
                intermediate_size=args.intermediate_size,
                rms_norm_eps=args.rms_norm_eps,
                wq=dequantize_linear(layer.self_attn.q_proj),   # (H, E)
                wk=dequantize_linear(layer.self_attn.k_proj),   # (H, E)
                wv=dequantize_linear(layer.self_attn.v_proj),
                wo=dequantize_linear(layer.self_attn.o_proj),
                q_norm=layer.self_attn.q_norm.weight,
                k_norm=layer.self_attn.k_norm.weight,
                w_gate=dequantize_linear(layer.mlp.gate_proj),  # (I, E)
                w_up=dequantize_linear(layer.mlp.up_proj),
                w_down=dequantize_linear(layer.mlp.down_proj),
                w_input_layernorm=layer.input_layernorm.weight,
                w_post_attention_layernorm=layer.post_attention_layernorm.weight,
                max_seq_len=args.max_position_embeddings,
                theta=args.rope_theta,
            )
            for layer in mlx_model.model.layers
        ]
        self.norm = RMSNorm(args.hidden_size, weight=mlx_model.model.norm.weight, eps=args.rms_norm_eps)
        # tied embeddings: small models reuse the input table; large models have lm_head
        self.w_lm_head = None if args.tie_word_embeddings else dequantize_linear(mlx_model.lm_head)

    def __call__(self, inputs: mx.array) -> mx.array:
        h = self.embedding(inputs)                          # (B, L, E) ->(B, L, E)
        mask = "causal" if inputs.shape[-1] > 1 else None
        for layer in self.layers_inner:
            h = layer(h, mask=mask)
        h = self.norm(h)                                    # (B, L, E)
        return self.embedding.as_linear(h) if self.w_lm_head is None else linear(h, self.w_lm_head) # (B, L, V)

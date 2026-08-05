"""
Shape notation used throughout this package:
    B     = batch size
    L     = sequence length
    D_in  = input feature dimension of a linear layer
    D_out = output feature dimension of a linear layer
    G     = quantization group size (128 for the course Qwen3 weights)
    P     = packed values per uint32 (8 values when bits=4)

For a dense linear weight W(D_out, D_in), its quantized representation is:
    weight  (D_out, D_in/P) uint32
    scales  (D_out, D_in/G) BF16
    biases  (D_out, D_in/G) BF16
"""

from typing import Any

import mlx.core as mx

from .attention import scaled_dot_product_attention_grouped
from .basics import linear, silu
from .embedding import Embedding, QuantizedEmbedding
from .kv_cache import TinyKvCache
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from .quantize import QuantizedWeights, dequantize_linear, quantized_linear
from .week2_kernels import (
    FastRMSNorm,
    FastRoPE,
    scaled_dot_product_attention,
    swiglu,
)

WEEK2_CHECKPOINTS = (
    "kv-cache",
    "quantized-matvec",
    "decode-attention",
    "rmsnorm",
    "rope",
    "swiglu",
    "simd-matmul",
    "split-k",
)


def _linear(x: mx.array, weight: mx.array | QuantizedWeights) -> mx.array:
    if isinstance(weight, QuantizedWeights):
        return quantized_linear(x, weight)
    return linear(x, weight)


def _readable_rope_offset(
    offset: int | list[int] | mx.array, sequence_length: int
) -> slice | list[slice]:
    if isinstance(offset, int):
        return slice(offset, offset + sequence_length)
    if isinstance(offset, list):
        return [slice(value, value + sequence_length) for value in offset]
    values = offset.tolist()
    if not isinstance(values, list):
        values = [values]
    return [slice(value, value + sequence_length) for value in values]


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_decode_attention: bool = True,
    ):
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.use_fast_rope = use_fast_rope
        self.use_decode_attention = use_decode_attention
        rope_cls = FastRoPE if use_fast_rope else RoPE
        norm_cls = FastRMSNorm if use_fast_rms_norm else RMSNorm
        self.rope = rope_cls(self.head_dim, max_seq_len, theta)
        self.q_norm = norm_cls(self.head_dim, q_norm, eps=rms_norm_eps)
        self.k_norm = norm_cls(self.head_dim, k_norm, eps=rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        offsets: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        q = _linear(x, self.wq).reshape(B, L, self.num_heads, self.head_dim)
        k = _linear(x, self.wk).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = _linear(x, self.wv).reshape(B, L, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        rope_offsets = offsets
        if not self.use_fast_rope:
            rope_offsets = _readable_rope_offset(offsets, L)
        q = self.rope(q, offset=rope_offsets)
        k = self.rope(k, offset=rope_offsets)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        k, v, _, mask = cache.update_and_fetch(k, v, mask_length=L, mask=mask)
        if self.use_decode_attention:
            out = scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        else:
            out = scaled_dot_product_attention_grouped(
                q.astype(mx.float32),
                k.astype(mx.float32),
                v.astype(mx.float32),
                scale=self.scale,
                mask=mask,
            ).astype(x.dtype)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        return _linear(out, self.wo)


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        use_fast_swiglu: bool = True,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.use_fast_swiglu = use_fast_swiglu

    def __call__(self, x: mx.array) -> mx.array:
        gate = _linear(x, self.w_gate)
        up = _linear(x, self.w_up)
        hidden = swiglu(gate, up) if self.use_fast_swiglu else silu(gate) * up
        return _linear(hidden, self.w_down)


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_fast_swiglu: bool = True,
        use_decode_attention: bool = True,
    ):
        norm_cls = FastRMSNorm if use_fast_rms_norm else RMSNorm
        self.input_layernorm = norm_cls(
            hidden_size, w_input_layernorm, eps=rms_norm_eps
        )
        self.post_attention_layernorm = norm_cls(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )
        self.self_attn = Qwen3MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            max_seq_len=max_seq_len,
            theta=theta,
            rms_norm_eps=rms_norm_eps,
            use_fast_rms_norm=use_fast_rms_norm,
            use_fast_rope=use_fast_rope,
            use_decode_attention=use_decode_attention,
        )
        self.mlp = Qwen3MLP(
            hidden_size,
            intermediate_size,
            w_gate,
            w_up,
            w_down,
            use_fast_swiglu=use_fast_swiglu,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), offset, cache, mask)
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3ModelWeek2:
    def __init__(self, mlx_model: Any, checkpoint: str = "split-k"):
        if checkpoint not in WEEK2_CHECKPOINTS:
            raise ValueError(
                f"unknown Week 2 checkpoint {checkpoint!r}; "
                f"choose one of {WEEK2_CHECKPOINTS}"
            )
        checkpoint_index = WEEK2_CHECKPOINTS.index(checkpoint)
        self.checkpoint = checkpoint
        use_quantized_weights = checkpoint_index >= WEEK2_CHECKPOINTS.index(
            "quantized-matvec"
        )
        use_decode_attention = checkpoint_index >= WEEK2_CHECKPOINTS.index(
            "decode-attention"
        )
        use_fast_rms_norm = checkpoint_index >= WEEK2_CHECKPOINTS.index("rmsnorm")
        use_fast_rope = checkpoint_index >= WEEK2_CHECKPOINTS.index("rope")
        use_fast_swiglu = checkpoint_index >= WEEK2_CHECKPOINTS.index("swiglu")
        use_simdgroup_matmul = checkpoint_index >= WEEK2_CHECKPOINTS.index(
            "simd-matmul"
        )
        use_split_k_matmul = checkpoint_index >= WEEK2_CHECKPOINTS.index("split-k")

        args = mlx_model.args
        self.num_hidden_layers = args.num_hidden_layers
        self.use_fast_rope = use_fast_rope
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size

        def model_weight(layer: Any) -> mx.array | QuantizedWeights:
            """Load one W(D_out, D_in) in the format required by checkpoint."""
            if use_quantized_weights:
                return QuantizedWeights.from_mlx_layer(
                    layer,
                    use_simdgroup_matmul=use_simdgroup_matmul,
                    use_simdgroup_matvec=True,
                    use_split_k_matmul=use_split_k_matmul,
                )
            return dequantize_linear(layer)

        embedding_weight = model_weight(mlx_model.model.embed_tokens)
        if isinstance(embedding_weight, QuantizedWeights):
            self.embedding = QuantizedEmbedding(
                vocab_size=self.vocab_size,
                embedding_dim=self.hidden_size,
                weight=embedding_weight,
                use_custom_kernel=use_simdgroup_matmul,
            )
        else:
            self.embedding = Embedding(
                vocab_size=self.vocab_size,
                embedding_dim=self.hidden_size,
                weight=embedding_weight,
            )

        self.layers_inner = []
        for layer in mlx_model.model.layers:
            self.layers_inner.append(
                Qwen3TransformerBlock(
                    num_attention_heads=args.num_attention_heads,
                    num_kv_heads=args.num_key_value_heads,
                    hidden_size=args.hidden_size,
                    head_dim=args.head_dim,
                    intermediate_size=args.intermediate_size,
                    rms_norm_eps=args.rms_norm_eps,
                    wq=model_weight(layer.self_attn.q_proj),
                    wk=model_weight(layer.self_attn.k_proj),
                    wv=model_weight(layer.self_attn.v_proj),
                    wo=model_weight(layer.self_attn.o_proj),
                    q_norm=layer.self_attn.q_norm.weight,
                    k_norm=layer.self_attn.k_norm.weight,
                    w_gate=model_weight(layer.mlp.gate_proj),
                    w_up=model_weight(layer.mlp.up_proj),
                    w_down=model_weight(layer.mlp.down_proj),
                    w_input_layernorm=layer.input_layernorm.weight,
                    w_post_attention_layernorm=layer.post_attention_layernorm.weight,
                    max_seq_len=args.max_position_embeddings,
                    theta=args.rope_theta,
                    use_fast_rms_norm=use_fast_rms_norm,
                    use_fast_rope=use_fast_rope,
                    use_fast_swiglu=use_fast_swiglu,
                    use_decode_attention=use_decode_attention,
                )
            )

        norm_cls = FastRMSNorm if use_fast_rms_norm else RMSNorm
        self.norm = norm_cls(
            args.hidden_size,
            weight=mlx_model.model.norm.weight,
            eps=args.rms_norm_eps,
        )
        self.w_lm_head = (
            None if args.tie_word_embeddings else model_weight(mlx_model.lm_head)
        )

    def create_kv_cache(self) -> list[TinyKvCache]:
        from .kv_cache import TinyKvFullCache

        return [TinyKvFullCache() for _ in range(self.num_hidden_layers)]

    def __call__(
        self,
        inputs: mx.array,
        offset: int | list[int] | mx.array,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> mx.array:
        if isinstance(offset, int):
            for layer_index, layer_cache in enumerate(cache):
                cache_offset = getattr(layer_cache, "offset", None)
                if cache_offset is not None and cache_offset != offset:
                    raise ValueError(
                        f"layer {layer_index} cache offset {cache_offset} "
                        f"does not match model offset {offset}"
                    )

        h = self.embedding(inputs)
        mask = None if inputs.shape[1] == 1 else "causal"
        if not self.use_fast_rope:
            rope_offsets = offset
        elif isinstance(offset, int):
            rope_offsets = mx.full((inputs.shape[0],), offset, dtype=mx.int32)
        elif isinstance(offset, list):
            rope_offsets = mx.array(offset, dtype=mx.int32)
        else:
            rope_offsets = offset

        for layer, layer_cache in zip(self.layers_inner, cache):
            h = layer(h, rope_offsets, layer_cache, mask=mask)
        if logits_to_keep is not None:
            if logits_to_keep <= 0:
                raise ValueError("logits_to_keep must be positive")
            h = h[:, -logits_to_keep:, :]
        h = self.norm(h)
        if self.w_lm_head is None:
            return self.embedding.as_linear(h)
        return _linear(h, self.w_lm_head)

import mlx.core as mx
from .basics import linear
from .quantize import QuantizedWeights, dequantize_weights, quantized_linear


class Embedding:
    """
    weight: (V, E)
    __call__:   ids (...,)  → vecs (..., E)    # row lookup
    as_linear:  vecs (..., E) → logits (..., V)  # x @ weightᵀ
    """

    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight  # (V, E)

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x]  # (...,) → (..., E)

    def as_linear(self, x: mx.array) -> mx.array:
        return linear(x, self.weight)  # (..., E) → (..., V)


class QuantizedEmbedding:
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        weight: QuantizedWeights,
        use_custom_kernel: bool = False,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight
        self.use_custom_kernel = use_custom_kernel

    def __call__(self, x: mx.array) -> mx.array:
        """Look up token ids ``x(B, L)`` and return embeddings ``(B, L, E)``."""
        packed_rows = self.weight.weight[x]  # (V,E/P)[B,L] -> (B,L,E/P)
        scales = self.weight.scales[x]       # (V,E/G)[B,L] -> (B,L,E/G)
        # Selecting the same token rows keeps quantization parameters aligned.
        biases = (
            None if self.weight.biases is None else self.weight.biases[x]
        )
        return dequantize_weights(
            packed_rows,
            scales,
            biases,
            self.weight.group_size,
            self.weight.bits,
        )

    def as_linear(self, x: mx.array) -> mx.array:
        """Use tied embeddings as LM head: ``(B, L, E) -> (B, L, V)``."""
        return quantized_linear(x, self.weight)  # (B,L,E) @ (V,E).T -> (B,L,V)

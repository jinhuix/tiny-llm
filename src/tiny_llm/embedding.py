import mlx.core as mx
from .basics import linear
from .quantize import QuantizedWeights


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
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass

    def as_linear(self, x: mx.array) -> mx.array:
        pass

import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    ''' y = x @ w^T + b '''
    y = mx.matmul(x, w.T)   # x(N, L, D_in), w(D_in, D_out), y(N, L, D_out)
    return y + bias if bias is not None else y

def silu(x: mx.array) -> mx.array:
    """
    SiLU(x) = x · σ(x) = x / (1 + exp(-x))
    """
    sigmod = 1 / (1 + mx.exp(-mx.abs(x)))   # stable sigmoid
    return x * mx.where(x < 0, 1 - sigmod, sigmod)
import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    """
    Args:
        temp:  0 = greedy; >0 scales distribution (higher → more random).
        top_p: Nucleus threshold (0/None = disabled).
        top_k: Keep only top-k tokens (0/None = disabled).

    Returns:
        Callable: logprobs (B, V) → token_ids (B,)
    """
    def sample(logprobs: mx.array):
        # Greedy: deterministic argmax
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)

        # Top-k: mask all but k highest-prob tokens
        if top_k is not None and top_k > 0:
            kth = mx.sort(logprobs, axis=-1)[:, -top_k]
            logprobs = mx.where(logprobs < kth[..., None], -mx.inf, logprobs)

        # Top-p: keep tokens until cumulative prob ≥ p
        if top_p:
            idx = mx.argsort(-logprobs, axis=-1)
            s = mx.take_along_axis(logprobs, idx, axis=-1)
            keep = mx.cumsum(mx.exp(s), axis=-1) < top_p
            keep[..., 0] = True
            s = mx.where(keep, s, -mx.inf)
            logprobs = mx.take_along_axis(s, mx.argsort(idx, axis=-1), axis=-1)

        # Temperature: scale then sample
        return mx.random.categorical(logprobs / temp, axis=-1)

    return sample

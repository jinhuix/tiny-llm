import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(y: mx.array) -> mx.array:
        logits = model(y[None])[:, -1, :]   # (S,) → (1, S) → (1, S, V)
        if sampler is None:
            return mx.argmax(logits, axis=-1)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return sampler(logprobs)

    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detok = tokenizer.detokenizer
    detok.reset()
    while True:
        token = _step(tokens)
        mx.eval(token)  # materialize so .item() works
        if token.item() == tokenizer.eos_token_id:
            break
        detok.add_token(token.item())
        print(detok.last_segment, end="", flush=True)
        tokens = mx.concat([tokens, token])

def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass

import mlx.core as mx


class RoPE:
    """
    θ_{m,i} = m · base^(-2i/D),  m = token position, i = pair index.
    z = x1 + i·x2, z' = z · e^{iθ}  =  (x1·cosθ - x2·sinθ) + i·(x1·sinθ + x2·cosθ)

    traditional:     pair = (x[2i], x[2i+1])
    non-traditional: pair = (x[i], x[i + D/2])
    """

    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.half_dims = dims // 2
        self.traditional = traditional

        # freq_i = base^(-2i/D),  i = 0..D/2-1
        inv = mx.arange(self.half_dims, dtype=mx.float32) / self.half_dims  # shape: (D/2,)
        freqs = mx.power(base, -inv)

        # angles[m, i] = m · freq_i
        theta = mx.outer(mx.arange(seq_len), freqs)    # shape: (seq_len, D/2)
        self.cos_freqs = mx.cos(theta)
        self.sin_freqs = mx.sin(theta)

    def __call__(
        self, x: mx.array, offset: slice | None = None
    ) -> mx.array:
        N, S, H, D = x.shape
        sl = slice(0, S) if offset is None else offset
        cos = self.cos_freqs[sl].reshape(1, S, 1, self.half_dims)   # (1, S, 1, D/2)
        sin = self.sin_freqs[sl].reshape(1, S, 1, self.half_dims)
        if self.traditional:
            pair = x.reshape(N, S, H, self.half_dims, 2)    # (N, S, H, D/2, 2), pairs: (x0,x1), (x2,x3), (x4,x5), ...
            x1, x2 = pair[..., 0], pair[..., 1]
        else:
            x1, x2 = x[..., : self.half_dims], x[..., self.half_dims :] # (N, S, H, D/2), pairs: (x0, x_{D/2}), (x1, x_{D/2+1}), ...
        
        # complex rotation
        real = x1 * cos - x2 * sin   # (N, S, H, D/2)
        imag = x1 * sin + x2 * cos   # (N, S, H, D/2)
        
        # reassemble in the original layout
        if self.traditional:
            y = mx.stack([real, imag], axis=-1).reshape(N, S, H, D) # (N,S,H,D/2) → (N,S,H,D/2,2) → (N,S,H,D)
        else:
            y = mx.concat([real, imag], axis=-1)
        return y.astype(x.dtype)
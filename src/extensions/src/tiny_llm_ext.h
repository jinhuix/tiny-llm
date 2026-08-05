#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

#include <stdexcept>

namespace mx = mlx::core;

namespace tiny_llm_ext {

void load_library(const char *path);

/**
 * Quantized linear algebra for the Week 2 W4A16 layout.
 *
 * a:       (M, D_in), bfloat16
 * b:       (D_out, D_in / 8), uint32 packed 4-bit weights
 * scales:  (D_out, D_in / 128), bfloat16
 * biases:  (D_out, D_in / 128), bfloat16
 * output:  (M, D_out), bfloat16
 */
mx::array quantized_matmul(const mx::array &scales, const mx::array &biases, int group_size, int bits,
                           const mx::array &a, const mx::array &b, bool transpose_b = false,
                           bool use_simdgroup = true,
                           mx::StreamOrDevice s = {});

class QuantizedMatmul : public mx::Primitive {
public:
    QuantizedMatmul(mx::Stream stream, int group_size, int bits, bool use_simdgroup)
        : mx::Primitive(stream), group_size_(group_size), bits_(bits), use_simdgroup_(use_simdgroup) {}

    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;

    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &,
                                                             const std::vector<int> &) override {
        throw std::runtime_error("QuantizedMatmul has no vmap implementation");
    }

    const char *name() const override { return "QuantizedMatmul"; }

private:
    int group_size_;
    int bits_;
    bool use_simdgroup_;    // true and M <= 8: SIMD matvec; otherwise: vanilla matmul.
};

}  // namespace tiny_llm_ext

#include "tiny_llm_ext.h"

#include <stdexcept>
#include <string>

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext {

namespace {

[[noreturn]] void checkpoint_todo(const char *function, const char *checkpoint) {
    throw std::runtime_error(std::string(function) + " is a starter stub; implement it in " + checkpoint);
}

}  // namespace

mx::array quantized_matmul(const mx::array &scales, const mx::array &biases, int group_size, int bits,
                           const mx::array &a, const mx::array &b, bool transpose_b, bool use_simdgroup,
                           bool use_split_k,
                           mx::StreamOrDevice s /* = {} */) {
    // W4A16: bit = 4, group_size = 128
    if (bits != 4) {
        throw std::runtime_error("quantized_matmul: bits must be 4");
    }
    if (group_size != 128) {
        throw std::runtime_error("quantized_matmul: group_size must be 128");
    }
    if (!transpose_b) {
        throw std::runtime_error("quantized_matmul: b must be transposed");
    }
    if (a.dtype() != mx::bfloat16 || scales.dtype() != mx::bfloat16 || biases.dtype() != mx::bfloat16) {
        throw std::runtime_error("quantized_matmul: a, scales, and biases must be bfloat16");
    }
    if (b.dtype() != mx::uint32) {
        throw std::runtime_error("quantized_matmul: packed b must be uint32");
    }

    if (a.shape().size() != 2 || b.shape().size() != 2 || scales.shape().size() != 2 ||
        biases.shape().size() != 2) {
        throw std::runtime_error("quantized_matmul: all inputs must be 2D arrays");
    }
    if (scales.shape() != biases.shape()) {
        throw std::runtime_error("quantized_matmul: scales and biases must have the same shape");
    }

    // Dense-equivalent multiplication:
    // a(M, D_in) @ W(D_out, D_in).T -> output(M, D_out)
    // Packed b stores P=8 int4 values per uint32.
    const int values_per_word = 32 / bits;
    const auto M = a.shape()[0];
    const auto D_in = a.shape()[1];
    const auto D_out = b.shape()[0];

    if (D_in % group_size != 0) {
        throw std::runtime_error("quantized_matmul: D_in must be divisible by group_size");
    }
    if (b.shape()[1] != D_in / values_per_word) {
        throw std::runtime_error("quantized_matmul: packed b has the wrong input width");
    }
    if (scales.shape()[0] != D_out || scales.shape()[1] != D_in / group_size) {
        throw std::runtime_error("quantized_matmul: scales and biases have incompatible shapes");
    }

    // No computation happens here. MLX records this primitive in its lazy
    // graph and calls eval_cpu/eval_gpu only when the output is evaluated.
    return mx::array({M, D_out}, a.dtype(),
                     std::make_shared<QuantizedMatmul>(to_stream(s), use_simdgroup, use_split_k),
                     {scales, biases, a, b});
}

void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("quantized_matmul: this course primitive is GPU-only");
}

#ifdef _METAL_

void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &scales = inputs[0];  // (D_out, D_in / 128), BF16
    const auto &biases = inputs[1];  // (D_out, D_in / 128), BF16
    const auto &a = inputs[2];       // (M, D_in), BF16
    const auto &b = inputs[3];       // (D_out, D_in / 8), uint32
    auto &out = outputs[0];          // (M, D_out), BF16

    // 检查连续内存
    if (!scales.flags().row_contiguous || !biases.flags().row_contiguous || !a.flags().row_contiguous ||
        !b.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: all GPU inputs must be row-contiguous");
    }

    const int M = static_cast<int>(a.shape()[0]);
    const int D_in = static_cast<int>(a.shape()[1]);
    const int D_out = static_cast<int>(b.shape()[0]);

    out.set_data(mx::allocator::malloc(out.nbytes()));  // 给输出分配GPU内存

    auto &stream = this->stream();  // 当前 primitive 所属的 MLX stream
    auto &device = mx::metal::device(stream.device);
    auto library = device.get_library("tiny_llm_ext");

    // M <= 8 时用 SIMD matvec；否则用一线程计算一个输出的 vanilla。
    const bool use_matvec = use_simdgroup_ && M <= 8;
    const char *kernel_name = use_matvec ? "quantized_matvec_x4_fast_w4a16_g128_bf16"
                                         : "quantized_matmul_vanilla_w4a16_g128_bf16";
    auto kernel = device.get_kernel(kernel_name, library);
    auto &encoder = mx::metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);

    encoder.set_input_array(scales, 0);
    encoder.set_input_array(biases, 1);
    encoder.set_input_array(a, 2);
    encoder.set_input_array(b, 3);
    encoder.set_output_array(out, 4);
    encoder.set_bytes(M, 5);
    encoder.set_bytes(D_in, 6);
    encoder.set_bytes(D_out, 7);

    if (use_matvec) {
        // 保留 Task 3 的 x4 调度：grid.x 是 8-column tile，grid.y 是输出行。
        // 每个 threadgroup 有两个 SIMD group，每组 32 lanes、计算 4 列。
        constexpr int simd_width = 32;
        constexpr int outputs_per_simdgroup = 4;
        constexpr int simdgroups_per_threadgroup = 2;
        constexpr int outputs_per_threadgroup = outputs_per_simdgroup * simdgroups_per_threadgroup;
        const int output_tiles = (D_out + outputs_per_threadgroup - 1) / outputs_per_threadgroup;
        encoder.dispatch_threadgroups(MTL::Size(output_tiles, M, 1),
                                      MTL::Size(simd_width * simdgroups_per_threadgroup, 1, 1));
        return;
    }

    // Vanilla: one GPU thread computes one out[m, k].
    constexpr int threads_x = 8;
    constexpr int threads_y = 8;
    const int row_groups = (M + threads_x - 1) / threads_x;
    const int column_groups = (D_out + threads_y - 1) / threads_y;
    encoder.dispatch_threadgroups(MTL::Size(row_groups, column_groups, 1),
                                  MTL::Size(threads_x, threads_y, 1));
}

#else

void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("quantized_matmul: Metal support is not available");
}

#endif

// Week 3, Day 4. The earlier Week 2 checkpoints keep the readable row lookup.
mx::array quantized_embedding(const mx::array &, const mx::array &, const mx::array &, const mx::array &, int, int,
                              mx::StreamOrDevice) {
    checkpoint_todo("quantized_embedding", "Week 3, Day 4");
}

void QuantizedEmbedding::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedEmbedding::eval_cpu", "Week 3, Day 4");
}

void QuantizedEmbedding::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedEmbedding::eval_gpu", "Week 3, Day 4");
}

}  // namespace tiny_llm_ext

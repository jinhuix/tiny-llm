#include <algorithm>
#include <stdexcept>
#include <string>

#include "tiny_llm_ext.h"

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

// Week 2, Day 4.
mx::array rms_norm(const mx::array &x, const mx::array &weight, float eps, mx::StreamOrDevice s) {
    if (x.dtype() != mx::bfloat16 || weight.dtype() != mx::bfloat16) {
        throw std::runtime_error("rms_norm: x and weight must be bfloat16");
    }
    if (x.ndim() == 0 || weight.ndim() != 1 || weight.shape()[0] != x.shape().back()) {
        throw std::runtime_error("rms_norm: weight must match x's final dimension");
    }
    if (eps <= 0.0f) {
        throw std::runtime_error("rms_norm: eps must be positive");
    }
    return mx::array(x.shape(), x.dtype(), std::make_shared<Week2RMSNorm>(to_stream(s), eps), {x, weight});
}

mx::array rope(const mx::array &x, const mx::array &offsets, int dims, float base, bool traditional,
               mx::StreamOrDevice s) {
    if (x.dtype() != mx::bfloat16) {
        throw std::runtime_error("rope: x must be bfloat16");
    }
    if (x.ndim() != 4 || offsets.dtype() != mx::int32 || offsets.ndim() != 1 ||
        offsets.shape()[0] != x.shape()[0]) {
        throw std::runtime_error("rope: expected x(B,L,H,D) and int32 offsets(B)");
    }
    if (dims <= 0 || dims > x.shape()[3] || dims % 2 != 0) {
        throw std::runtime_error("rope: dims must be positive, even, and <= D");
    }
    if (base <= 0.0f) {
        throw std::runtime_error("rope: base must be positive");
    }
    return mx::array(x.shape(), x.dtype(),
                     std::make_shared<Week2RoPE>(to_stream(s), dims, base, traditional), {x, offsets});
}

mx::array swiglu(const mx::array &gate, const mx::array &up, mx::StreamOrDevice s) {
    if (gate.dtype() != mx::bfloat16 || up.dtype() != mx::bfloat16) {
        throw std::runtime_error("swiglu: gate and up must be bfloat16");
    }
    if (gate.shape() != up.shape()) {
        throw std::runtime_error("swiglu: gate and up must have the same shape");
    }
    return mx::array(gate.shape(), gate.dtype(), std::make_shared<Week2SwiGLU>(to_stream(s)), {gate, up});
}

void Week2RMSNorm::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("rms_norm: this course primitive is GPU-only");
}

void Week2RMSNorm::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &x = inputs[0];       // (rows, D), flattened view of (..., D)
    const auto &weight = inputs[1];  // (D,)
    auto &out = outputs[0];
    if (!x.flags().row_contiguous || !weight.flags().row_contiguous) {
        throw std::runtime_error("rms_norm: GPU inputs must be row-contiguous");
    }

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &device = mx::metal::device(stream().device);
    auto kernel = device.get_kernel("week2_rms_norm_bf16", device.get_library("tiny_llm_ext"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(x, 0);
    encoder.set_input_array(weight, 1);
    encoder.set_output_array(out, 2);

    const int dim = static_cast<int>(x.shape().back());
    const int rows = static_cast<int>(x.size() / dim);
    encoder.set_bytes(rows, 3);
    encoder.set_bytes(dim, 4);
    encoder.set_bytes(eps_, 5);

    // 一行一个threadgroup：256 个线程 = 8 个 32-lane SIMD groups
    // 每个 threadgroup 分配一小块共享内存，大小是 8 个 float
    constexpr int threads_per_group = 256;
    encoder.set_threadgroup_memory_length(8 * sizeof(float), 0);
    encoder.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(threads_per_group, 1, 1));   // threadgroup grid(rows, 1, 1)
}

void Week2RoPE::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("rope: this course primitive is GPU-only");
}

void Week2RoPE::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &x = inputs[0];        // (B, L, H, D)
    const auto &offsets = inputs[1];  // (B,), int32
    auto &out = outputs[0];           // (B, L, H, D)
    if (!x.flags().row_contiguous || !offsets.flags().row_contiguous) {
        throw std::runtime_error("rope: GPU inputs must be row-contiguous");
    }

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &device = mx::metal::device(stream().device);
    auto kernel = device.get_kernel("week2_rope_bf16", device.get_library("tiny_llm_ext"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(x, 0);
    encoder.set_input_array(offsets, 1);
    encoder.set_output_array(out, 2);

    const int batch = static_cast<int>(x.shape()[0]);
    const int length = static_cast<int>(x.shape()[1]);
    const int heads = static_cast<int>(x.shape()[2]);
    const int head_dim = static_cast<int>(x.shape()[3]);
    const int traditional = traditional_ ? 1 : 0;
    encoder.set_bytes(batch, 3);
    encoder.set_bytes(length, 4);
    encoder.set_bytes(heads, 5);
    encoder.set_bytes(head_dim, 6);
    encoder.set_bytes(dims_, 7);
    encoder.set_bytes(base_, 8);
    encoder.set_bytes(traditional, 9);

    // 同一个batch row/position/pair index使用相同的theta
    // 一个tread处理4个head，计算一次theta在4个thread中复用，计算量减少4倍
    constexpr int heads_per_thread = 4;
    const int head_blocks = (heads + heads_per_thread - 1) / heads_per_thread;
    // 旋转工作(dims/2个pair) + tail copy工作(未参与旋转的 head_dim - dims)
    const int items_per_block = dims_ / 2 + (head_dim - dims_);
    // 总线程数
    const size_t work_items = static_cast<size_t>(batch) * length * head_blocks * items_per_block;
    const size_t threads_per_group = std::min<size_t>(work_items, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(work_items, 1, 1), MTL::Size(threads_per_group, 1, 1));
}

void Week2SwiGLU::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("swiglu: this course primitive is GPU-only");
}

void Week2SwiGLU::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &gate = inputs[0];  // (..., D_ff)
    const auto &up = inputs[1];    // (..., D_ff)
    auto &out = outputs[0];        // (..., D_ff)
    if (!gate.flags().row_contiguous || !up.flags().row_contiguous) {
        throw std::runtime_error("swiglu: GPU inputs must be row-contiguous");
    }

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &device = mx::metal::device(stream().device);
    auto kernel = device.get_kernel("week2_swiglu_bf16", device.get_library("tiny_llm_ext"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(gate, 0);
    encoder.set_input_array(up, 1);
    encoder.set_output_array(out, 2);
    const int size = static_cast<int>(out.size());
    encoder.set_bytes(size, 3);

    const size_t threads_per_group = std::min<size_t>(out.size(), kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(out.size(), 1, 1), MTL::Size(threads_per_group, 1, 1));
}

// Week 2, Day 5.
mx::array decode_attention(const mx::array &, const mx::array &, const mx::array &, const mx::array &, float, bool,
                           bool, int, int, mx::StreamOrDevice) {
    checkpoint_todo("decode_attention", "Week 2, Day 5");
}

void Week2DecodeAttention::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2DecodeAttention::eval_cpu", "Week 2, Day 5");
}

void Week2DecodeAttention::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2DecodeAttention::eval_gpu", "Week 2, Day 5");
}

}  // namespace tiny_llm_ext

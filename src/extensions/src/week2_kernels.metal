#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Week 2, Day 4:
//   week2_rms_norm
//   week2_rope
//   week2_swiglu
// Week 2, Day 5:
//   week2_decode_attention
//
// Add each [[kernel]] function when its task asks for it. The C++ starter
// wrapper and binding already exist, but remain fail-closed until then.


// Metal 常见地址空间
// device:      整个 GPU 可访问的全局内存
// constant:    仅当前 GPU 可访问的全局内存
// threadgroup: 同一个 threadgroup 共享的高速临时内存
// thread:      单个线程私有寄存器/局部变量

/*
 * RMSNorm: one threadgroup owns one flattened row x[row, :].
 *
 *   mean_square = sum_d x[d]^2 / D
 *   out[d] = x[d] / sqrt(mean_square + eps) * weight[d]
 *
 */
template <typename T>
[[kernel]] void week2_rms_norm(
    device const T* x [[buffer(0)]],                      // (B, L, H, D) -> (B * L * H, D)
    device const T* weight [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    constant const int& dim [[buffer(4)]],
    constant const float& eps [[buffer(5)]],
    threadgroup float* partial_sums [[threadgroup(0)]],   // 同一个 threadgroup 共享的高速临时内存
    uint row [[threadgroup_position_in_grid]],            // 处理行号
    uint thread_id [[thread_index_in_threadgroup]],       // 一个threadgroup有256个线程
    uint simdgroup_id [[simdgroup_index_in_threadgroup]], // 一个threadgroup被分成8个32-lanes
    uint lane [[thread_index_in_simdgroup]]) {            // 一个SIMD group有32个lanes
  if (row >= static_cast<uint>(rows)) return;

  constexpr uint threads_per_group = 256;
  constexpr uint simdgroups_per_group = threads_per_group / 32;
  const int row_start = static_cast<int>(row) * dim;      // 当前行起始地址

  // Level 1: 本行的256个thread分别归约
  float local_sum = 0.0f;   // 每个SIMD group的局部和
  for (int d = static_cast<int>(thread_id); d < dim; d += threads_per_group) {
    const float value = static_cast<float>(x[row_start + d]);
    local_sum += value * value;
  }
  local_sum = simd_sum(local_sum);  // 一个SIMD group的局部和
  if (lane == 0) partial_sums[simdgroup_id] = local_sum;  // lane 0写整个SIMD group的和
  threadgroup_barrier(mem_flags::mem_threadgroup);  // 同步threadgroup中的内存

  // Level 2: 合并8个结果到第一个SIMD group
  if (simdgroup_id == 0) {
    float value = lane < simdgroups_per_group ? partial_sums[lane] : 0.0f;
    value = simd_sum(value);
    if (lane == 0) partial_sums[0] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float inv_rms = rsqrt(partial_sums[0] / static_cast<float>(dim) + eps);     // 1/sqrt(mean_square + eps)
  for (int d = static_cast<int>(thread_id); d < dim; d += threads_per_group) {        // 256个线程并行计算整行
    const float normalized = static_cast<float>(x[row_start + d]) * inv_rms;
    out[row_start + d] = static_cast<T>(normalized * static_cast<float>(weight[d]));
  }
}

/*
 * RoPE
 *
 *   theta = (offset[b] + token) * base^(-pair / (dims/2))
 *   [real'] = [ cos -sin ] [real]
 *   [imag']   [ sin  cos ] [imag]
 *
 */
template <typename T>
[[kernel]] void week2_rope(
    device const T* x [[buffer(0)]],
    device const int32_t* offsets [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const int& batch [[buffer(3)]],
    constant const int& length [[buffer(4)]],
    constant const int& heads [[buffer(5)]],
    constant const int& head_dim [[buffer(6)]],
    constant const int& dims [[buffer(7)]],
    constant const float& base [[buffer(8)]],
    constant const int& traditional [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
  constexpr int heads_per_thread = 4;
  const int half_dims = dims / 2;
  const int tail_dims = head_dim - dims;
  const int items_per_head_block = half_dims + tail_dims;
  const int head_blocks = (heads + heads_per_thread - 1) / heads_per_thread;
  const int total = batch * length * head_blocks * items_per_head_block;
  if (index >= static_cast<uint>(total)) return;

  // 表示当前线程负责某个旋转pair或某个tail feature
  const int item = index % items_per_head_block;
  const int head_block = (index / items_per_head_block) % head_blocks;
  const int token = (index / (items_per_head_block * head_blocks)) % length;
  const int batch_row = index / (items_per_head_block * head_blocks * length);
  const int first_head = head_block * heads_per_thread;
  const int last_head = min(first_head + heads_per_thread, heads);
  const int token_base = (batch_row * length + token) * heads * head_dim;

  // A partial RoPE rotates x[..., :dims] and leaves x[..., dims:] intact.
  if (item >= half_dims) {
    const int d = dims + (item - half_dims);
    for (int head = first_head; head < last_head; ++head) {
      const int element = token_base + head * head_dim + d;
      out[element] = x[element];
    }
    return;
  }

  const int pair = item;
  const float frequency_power = -static_cast<float>(pair) / static_cast<float>(half_dims);
  const float frequency = fast::exp2(frequency_power * log2(base));
  const float angle = static_cast<float>(offsets[batch_row] + token) * frequency;
  const float cosine = fast::cos(angle);
  const float sine = fast::sin(angle);

  for (int head = first_head; head < last_head; ++head) {
    const int head_base = token_base + head * head_dim;
    const int real_index = traditional ? head_base + 2 * pair : head_base + pair;
    const int imag_index = traditional ? real_index + 1 : real_index + half_dims;
    const float real = static_cast<float>(x[real_index]);
    const float imag = static_cast<float>(x[imag_index]);
    out[real_index] = static_cast<T>(real * cosine - imag * sine);
    out[imag_index] = static_cast<T>(real * sine + imag * cosine);
  }
}

/*
 * SwiGLU is entirely element-wise.  This is the equation from the readable
 * implementation, fused into one dispatch with one exponential and one final
 * store per element:
 *
 *   out = (g / (1 + exp(-g))) * up
 */
template <typename T>
[[kernel]] void week2_swiglu(
    device const T* gate [[buffer(0)]],
    device const T* up [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const int& size [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= static_cast<uint>(size)) return;
  const float g = static_cast<float>(gate[index]);
  const float silu = g / (1.0f + fast::exp(-g));
  out[index] = static_cast<T>(silu * static_cast<float>(up[index]));
}

instantiate_kernel("week2_rms_norm_bf16", week2_rms_norm, bfloat16_t);
instantiate_kernel("week2_rope_bf16", week2_rope, bfloat16_t);
instantiate_kernel("week2_swiglu_bf16", week2_swiglu, bfloat16_t);

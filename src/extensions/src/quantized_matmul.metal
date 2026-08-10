#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Later checkpoints extend this file with:
//   quantized_matmul_simdgroup_w4a16_g128       (Week 2, Day 6)
//   quantized_matmul_simdgroup_splitk_w4a16_g128 (Week 2, Day 7)
//   quantized_matmul_splitk_reduce               (Week 2, Day 7)
//   quantized_embedding_w4a16_g128               (Week 3, Day 4)

inline uint32_t unpack_int4(uint32_t packed, int value_index) {
    return (packed >> (value_index * 4)) & 0xFu;
}

/**
 * Vanilla W4A16 G128 matmul（正确性基线）。
 *
 * 逻辑计算：out(M, D_out) = a(M, D_in) @ W(D_out, D_in).T
 * 存储形状：
 *   a       (M, D_in)              BF16
 *   b       (D_out, D_in / 8)      uint32，1 word = 8 个 int4
 *   scales  (D_out, D_in / 128)    BF16
 *   biases  (D_out, D_in / 128)    BF16
 *   out     (M, D_out)             BF16
 *
 * 一个 GPU thread 计算一个 out[m, k]。
 */
template <typename T>
[[kernel]] void quantized_matmul_vanilla_w4a16_g128(
    device const T *scales [[buffer(0)]],
    device const T *biases [[buffer(1)]],
    device const T *a [[buffer(2)]],
    device const uint32_t *b [[buffer(3)]],
    device T *out [[buffer(4)]],
    constant const int &M [[buffer(5)]],
    constant const int &D_in [[buffer(6)]],
    constant const int &D_out [[buffer(7)]],
    uint2 position [[thread_position_in_grid]]) {
    constexpr int group_size = 128;
    constexpr int values_per_word = 8;

    const int m = static_cast<int>(position.x);  // a/out 的行
    const int k = static_cast<int>(position.y);  // W 的行，也就是输出列
    if (m >= M || k >= D_out) {
        return;
    }

    const int packed_per_group = group_size / values_per_word;  // 128 / 8 = 16
    const int packed_per_row = D_in / values_per_word;          // b 每行有 D_in / 8 个 word
    const int groups_per_row = D_in / group_size;               // 每行有 D_in / 128 个量化组

    const int activation_row = m * D_in;           // a[m, 0]
    const int weight_row = k * packed_per_row;     // b[k, 0]
    const int parameter_row = k * groups_per_row;  // scales[k, 0] / biases[k, 0]

    float sum = 0.0f;  // BF16 输入，FP32 累加，最后转回 BF16。

    // 外层按量化组遍历：一组 128 个权重只读取一次 scale/bias。
    for (int group = 0; group < groups_per_row; ++group) {
        const float scale = static_cast<float>(scales[parameter_row + group]);
        const float bias = static_cast<float>(biases[parameter_row + group]);
        const int first_input = group * group_size;
        const int first_packed = group * packed_per_group;

        // 一个 group 有 16 个 uint32，每个 uint32 包含 8 个 int4。
        for (int pack = 0; pack < packed_per_group; ++pack) {
            const uint32_t packed = b[weight_row + first_packed + pack];
            const int input_base = activation_row + first_input + pack * values_per_word;

            #pragma clang loop unroll(full)
            for (int value = 0; value < values_per_word; ++value) {
                const float q = static_cast<float>(unpack_int4(packed, value));
                const float weight = q * scale + bias;  // 逻辑上的 W[k, j]
                const float activation = static_cast<float>(a[input_base + value]);  // a[m, j]
                sum += activation * weight;
            }
        }
    }

    out[m * D_out + k] = static_cast<T>(sum);
}

instantiate_kernel(
    "quantized_matmul_vanilla_w4a16_g128_bf16",
    quantized_matmul_vanilla_w4a16_g128,
    bfloat16_t);

/**
 * Task 3 的 x4 SIMD matvec（M <= 8）。
 *
 * 性能结构保持不变：
 *   - 一个 threadgroup 有 2 个 SIMD group，共 64 threads；
 *   - 一个 SIMD group 有 32 lanes，同时计算 4 个输出；
 *   - 一个 threadgroup 一共覆盖 8 个输出；
 *   - 每个 lane 一次读取 2 个 packed word = 16 个 activation；
 *   - 16 个 activation 复用于 4 个输出；
 *   - 使用仿射公式和 simd_sum 完成 reduction。
 */
template <typename T>
[[kernel]] void quantized_matvec_x4_fast_w4a16_g128(
    device const T *scales [[buffer(0)]],
    device const T *biases [[buffer(1)]],
    device const T *a [[buffer(2)]],
    device const uint32_t *b [[buffer(3)]],
    device T *out [[buffer(4)]],
    constant const int &M [[buffer(5)]],
    constant const int &D_in [[buffer(6)]],
    constant const int &D_out [[buffer(7)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int group_size = 128;
    constexpr int values_per_word = 8;
    constexpr int words_per_lane = 2;
    constexpr int values_per_lane = values_per_word * words_per_lane;  // 16
    constexpr int simd_width = 32;
    constexpr int outputs_per_simdgroup = 4;
    constexpr int simdgroups_per_threadgroup = 2;
    constexpr int outputs_per_threadgroup =
        outputs_per_simdgroup * simdgroups_per_threadgroup;  // 8

    // C++ 直接启动二维 grid，不再把 (m, tile) 压成一维后再除法恢复：
    //   group_position.x = 当前行的第几个 8-column tile
    //   group_position.y = 输出行 m
    const int output_tile = static_cast<int>(group_position.x);
    const int m = static_cast<int>(group_position.y);

    // SIMD group 0 负责 tile 的前 4 列，SIMD group 1 负责后 4 列。
    const int column_base = output_tile * outputs_per_threadgroup +
                            static_cast<int>(simdgroup) * outputs_per_simdgroup;
    if (m >= M || column_base >= D_out) {
        return;
    }

    const int packed_per_row = D_in / values_per_word;
    const int packed_per_group = group_size / values_per_word;  // 16 words
    const int groups_per_row = D_in / group_size;
    const int activation_row = m * D_in;

    // 每个 lane 对 4 个输出各自保留一个 FP32 局部和。
    float partial[outputs_per_simdgroup] = {0.0f};

    // lane 0: packed [0,1]，lane 1: [2,3]，...，下一轮整体前进 64 words。
    for (int packed_col = static_cast<int>(lane) * words_per_lane;
         packed_col < packed_per_row;
         packed_col += simd_width * words_per_lane) {
        // 每个 group 有 16 words；packed_col 总是偶数，所以连续两个 word 不跨 group。
        const int group = packed_col / packed_per_group;

        // 先读取当前 lane 的 16 个 activation，随后复用于 4 个输出。
        float activations[values_per_lane];
        float activation_sum = 0.0f;

        #pragma clang loop unroll(full)
        for (int word = 0; word < words_per_lane; ++word) {
            const int input_base = activation_row + (packed_col + word) * values_per_word;

            #pragma clang loop unroll(full)
            for (int value = 0; value < values_per_word; ++value) {
                const int local = word * values_per_word + value;
                const float activation = static_cast<float>(a[input_base + value]);
                activations[local] = activation;
                activation_sum += activation;
            }
        }

        // 相同的 16 个 activation 分别乘 4 行不同的量化权重。
        #pragma clang loop unroll(full)
        for (int output = 0; output < outputs_per_simdgroup; ++output) {
            const int k = column_base + output;
            if (k >= D_out) {
                continue;
            }

            const int parameter_index = k * groups_per_row + group;
            const float scale = static_cast<float>(scales[parameter_index]);
            const float bias = static_cast<float>(biases[parameter_index]);
            const int weight_base = k * packed_per_row + packed_col;
            float quantized_dot = 0.0f;

            #pragma clang loop unroll(full)
            for (int word = 0; word < words_per_lane; ++word) {
                const uint32_t packed = b[weight_base + word];

                #pragma clang loop unroll(full)
                for (int value = 0; value < values_per_word; ++value) {
                    const int local = word * values_per_word + value;
                    const float q = static_cast<float>(unpack_int4(packed, value));
                    quantized_dot += activations[local] * q;
                }
            }

            partial[output] += scale * quantized_dot + bias * activation_sum;
        }
    }

    // 每个 lane 只覆盖 D_in 的一部分；分别合并 4 个输出的 32 份局部和。
    #pragma clang loop unroll(full)
    for (int output = 0; output < outputs_per_simdgroup; ++output) {
        partial[output] = simd_sum(partial[output]);
    }

    // simd_sum 后 32 个 lane 都得到完整结果，只由 lane 0 写回一次。
    if (lane == 0) {
        #pragma clang loop unroll(full)
        for (int output = 0; output < outputs_per_simdgroup; ++output) {
            const int k = column_base + output;
            if (k < D_out) {
                out[m * D_out + k] = static_cast<T>(partial[output]);
            }
        }
    }
}

instantiate_kernel(
    "quantized_matvec_x4_fast_w4a16_g128_bf16",
    quantized_matvec_x4_fast_w4a16_g128,
    bfloat16_t);

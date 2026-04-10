#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

template <typename T>
__device__ inline float dotcache_qwen35_to_float(T value);

template <>
__device__ inline float dotcache_qwen35_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ inline float dotcache_qwen35_to_float<float>(float value) {
    return value;
}

template <>
__device__ inline float dotcache_qwen35_to_float<hip_bfloat16>(hip_bfloat16 value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ inline T dotcache_qwen35_from_float(float value);

template <>
__device__ inline __half dotcache_qwen35_from_float<__half>(float value) {
    return __float2half(value);
}

template <>
__device__ inline float dotcache_qwen35_from_float<float>(float value) {
    return value;
}

template <>
__device__ inline hip_bfloat16 dotcache_qwen35_from_float<hip_bfloat16>(float value) {
    return hip_bfloat16(value);
}

__device__ inline float dotcache_qwen35_wave_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down(value, offset);
    }
    return value;
}

template <typename T>
__global__ void dotcache_qwen35_full_attention_prefill_kernel(
    int batch_size,
    int q_heads,
    int kv_heads,
    int q_len,
    int kv_len,
    int head_dim,
    int num_kv_groups,
    float scale,
    int seqlen_offset,
    const T* query,
    const T* key,
    const T* value,
    float* out,
    unsigned int* row_counter
) {
    const int lane = threadIdx.x;
    if (lane >= warpSize) {
        return;
    }

    const int total_rows = batch_size * q_heads * q_len;
    while (true) {
        unsigned int row = 0;
        if (lane == 0) {
            row = atomicAdd(row_counter, 1u);
        }
        row = __shfl(row, 0);
        if ((int)row >= total_rows) {
            return;
        }

        const int q_pos = row % q_len;
        const int q_head = (row / q_len) % q_heads;
        const int batch = row / (q_len * q_heads);
        const int kv_head = q_head / num_kv_groups;
        const int causal_limit = min(kv_len, seqlen_offset + q_pos + 1);

        const T* q_row = query + (((batch * q_heads + q_head) * q_len + q_pos) * head_dim);
        const T* k_head_ptr = key + ((batch * kv_heads + kv_head) * kv_len * head_dim);
        const T* v_head_ptr = value + ((batch * kv_heads + kv_head) * kv_len * head_dim);
        float* out_row = out + (((batch * q_heads + q_head) * q_len + q_pos) * head_dim);

        float running_max = -INFINITY;
        float denom = 0.0f;
        float local_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        int local_dims[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        int local_count = 0;
        for (int d = lane; d < head_dim && local_count < 8; d += warpSize) {
            local_dims[local_count++] = d;
        }

        __shared__ float shared_prev_scale;
        __shared__ float shared_curr_scale;
        __shared__ float shared_denom;
        __shared__ float shared_inv_denom;
        __shared__ float shared_new_max;

        for (int k_pos = 0; k_pos < causal_limit; ++k_pos) {
            const T* k_row = k_head_ptr + k_pos * head_dim;
            const T* v_row = v_head_ptr + k_pos * head_dim;

            float partial = 0.0f;
            for (int d = lane; d < head_dim; d += warpSize) {
                partial += dotcache_qwen35_to_float(q_row[d]) * dotcache_qwen35_to_float(k_row[d]);
            }
            float score = dotcache_qwen35_wave_sum(partial) * scale;

            if (lane == 0) {
                if (!isfinite(running_max)) {
                    running_max = score;
                    denom = 1.0f;
                    shared_prev_scale = 0.0f;
                    shared_curr_scale = 1.0f;
                    shared_new_max = running_max;
                } else {
                    const float new_max = fmaxf(running_max, score);
                    const float prev_scale = expf(running_max - new_max);
                    const float curr_scale = expf(score - new_max);
                    denom = denom * prev_scale + curr_scale;
                    running_max = new_max;
                    shared_prev_scale = prev_scale;
                    shared_curr_scale = curr_scale;
                    shared_new_max = new_max;
                }
                shared_denom = denom;
            }
            __syncthreads();

            if (shared_curr_scale == 1.0f && shared_prev_scale == 0.0f) {
                for (int i = 0; i < local_count; ++i) {
                    local_acc[i] = dotcache_qwen35_to_float(v_row[local_dims[i]]);
                }
            } else {
                for (int i = 0; i < local_count; ++i) {
                    local_acc[i] = local_acc[i] * shared_prev_scale +
                        shared_curr_scale * dotcache_qwen35_to_float(v_row[local_dims[i]]);
                }
            }
            __syncthreads();
        }

        if (lane == 0) {
            shared_inv_denom = shared_denom > 0.0f ? 1.0f / shared_denom : 0.0f;
        }
        __syncthreads();
        for (int i = 0; i < local_count; ++i) {
            out_row[local_dims[i]] = local_acc[i] * shared_inv_denom;
        }
        __syncthreads();
    }
}

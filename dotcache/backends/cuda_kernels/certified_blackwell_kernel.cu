#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>

namespace {

constexpr int kThreads = 256;
constexpr int kBlockSize = 16;

__device__ __forceinline__ float block_reduce_max(float value, float* scratch) {
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ __forceinline__ float block_reduce_sum(float value, float* scratch) {
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

template <typename QueryT>
__device__ __forceinline__ float load_query(const QueryT* ptr, int idx) {
    return static_cast<float>(ptr[idx]);
}

template <>
__device__ __forceinline__ float load_query<at::Half>(const at::Half* ptr, int idx) {
    return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
}

template <>
__device__ __forceinline__ float load_query<at::BFloat16>(const at::BFloat16* ptr, int idx) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
}

__device__ __forceinline__ float load_half(const at::Half* ptr, int idx) {
    return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
}

template <typename QueryT>
__global__ void mixedv_partial_kernel(
    const int8_t* __restrict__ keys_int8,
    const float* __restrict__ keys_scale,
    const float* __restrict__ keys_zp,
    const at::Half* __restrict__ keys_fp16,
    const int32_t* __restrict__ topk_mask,
    const uint8_t* __restrict__ values_packed,
    const at::Half* __restrict__ values_scales,
    const at::Half* __restrict__ values_zeros,
    const at::Half* __restrict__ values_fp16_scratch,
    const int32_t* __restrict__ value_fp16_mask,
    const int32_t* __restrict__ value_block_slots,
    const QueryT* __restrict__ q_all,
    const int32_t* __restrict__ skip_mask,
    float* __restrict__ m_part,
    float* __restrict__ l_part,
    float* __restrict__ acc_part,
    int kv_heads,
    int tokens,
    int head_dim,
    int d_v,
    int groups,
    int q_heads,
    int blocks,
    int gqa_group,
    int group_size,
    int scratch_tokens,
    int num_splits,
    int blocks_per_split,
    int last_block_valid,
    float q_scale) {
    __shared__ float s_scores[kBlockSize];
    __shared__ float s_weights[kBlockSize];
    __shared__ float s_reduce[kThreads];

    const int prog = blockIdx.x;
    const int qh = prog / num_splits;
    const int split = prog - qh * num_splits;
    const int tid = threadIdx.x;
    if (qh >= q_heads) {
        return;
    }

    const int kvh = min(qh / gqa_group, kv_heads - 1);
    const int block_start = split * blocks_per_split;
    const int block_end = min(block_start + blocks_per_split, blocks);

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;

    for (int bid = block_start; bid < block_end; ++bid) {
        const bool skip = skip_mask[qh * blocks + bid] != 0;
        float score = -INFINITY;
        if (!skip && tid < kBlockSize) {
            const int tok = bid * kBlockSize + tid;
            const bool valid_tok = (bid != blocks - 1) || (tid < last_block_valid);
            if (valid_tok) {
                const int use_fp16_key = topk_mask[qh * blocks + bid] != 0;
                float dot = 0.0f;
                const int key_base = (kvh * tokens + tok) * head_dim;
                const int scale_base = (kvh * blocks + bid) * head_dim;
                const int q_base = qh * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    const float q = load_query<QueryT>(q_all, q_base + d);
                    float k;
                    if (use_fp16_key) {
                        k = load_half(keys_fp16, key_base + d);
                    } else {
                        k = static_cast<float>(keys_int8[key_base + d]) * keys_scale[scale_base + d]
                            + keys_zp[scale_base + d];
                    }
                    dot += q * k;
                }
                score = dot * q_scale;
            }
        }
        if (tid < kBlockSize) {
            s_scores[tid] = score;
        }
        __syncthreads();

        const float block_max = block_reduce_max(tid < kBlockSize ? score : -INFINITY, s_reduce);
        const float new_m = fmaxf(m, block_max);
        const float alpha = expf(m - new_m);
        if (tid < d_v) {
            acc *= alpha;
        }
        l *= alpha;

        float weight = 0.0f;
        if (tid < kBlockSize && isfinite(s_scores[tid])) {
            weight = expf(s_scores[tid] - new_m);
            s_weights[tid] = weight;
        } else if (tid < kBlockSize) {
            s_weights[tid] = 0.0f;
        }
        __syncthreads();
        l += block_reduce_sum(tid < kBlockSize ? weight : 0.0f, s_reduce);

        if (!skip && tid < d_v) {
            const int use_fp16_value = value_fp16_mask[qh * blocks + bid] != 0;
            const int slot = max(value_block_slots[bid], 0);
            float block_acc = 0.0f;
            for (int local_tok = 0; local_tok < kBlockSize; ++local_tok) {
                const float w = s_weights[local_tok];
                if (w == 0.0f) {
                    continue;
                }
                float v;
                if (use_fp16_value) {
                    const int idx = (kvh * scratch_tokens + slot * kBlockSize + local_tok) * d_v + tid;
                    v = load_half(values_fp16_scratch, idx);
                } else {
                    const int tok = bid * kBlockSize + local_tok;
                    const int packed_idx = (kvh * tokens + tok) * (d_v / 2) + (tid / 2);
                    const uint8_t packed = values_packed[packed_idx];
                    const int code = (tid & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
                    const int group = tid / group_size;
                    const int scale_idx = (kvh * tokens + tok) * groups + group;
                    v = static_cast<float>(code) * load_half(values_scales, scale_idx)
                        + load_half(values_zeros, scale_idx);
                }
                block_acc += w * v;
            }
            acc += block_acc;
        }
        m = new_m;
        __syncthreads();
    }

    const int part = qh * num_splits + split;
    if (tid == 0) {
        m_part[part] = m;
        l_part[part] = l;
    }
    if (tid < d_v) {
        acc_part[part * d_v + tid] = acc;
    }
}

__global__ void mixedv_reduce_kernel(
    const float* __restrict__ m_part,
    const float* __restrict__ l_part,
    const float* __restrict__ acc_part,
    float* __restrict__ output,
    int q_heads,
    int d_v,
    int num_splits) {
    const int qh = blockIdx.x;
    const int tid = threadIdx.x;
    if (qh >= q_heads) {
        return;
    }

    float m_global = -INFINITY;
    for (int split = 0; split < num_splits; ++split) {
        m_global = fmaxf(m_global, m_part[qh * num_splits + split]);
    }

    float l_total = 0.0f;
    float acc_total = 0.0f;
    for (int split = 0; split < num_splits; ++split) {
        const int part = qh * num_splits + split;
        const float scale = expf(m_part[part] - m_global);
        l_total += l_part[part] * scale;
        if (tid < d_v) {
            acc_total += acc_part[part * d_v + tid] * scale;
        }
    }
    if (tid < d_v) {
        output[qh * d_v + tid] = acc_total / fmaxf(l_total, 1e-20f);
    }
}

template <typename QueryT>
void launch_partial(
    const torch::Tensor& keys_int8,
    const torch::Tensor& keys_scale,
    const torch::Tensor& keys_zero_points,
    const torch::Tensor& keys_fp16,
    const torch::Tensor& topk_mask,
    const torch::Tensor& values_int4_packed,
    const torch::Tensor& values_int4_scales,
    const torch::Tensor& values_int4_zeros,
    const torch::Tensor& values_fp16_scratch,
    const torch::Tensor& value_fp16_mask,
    const torch::Tensor& value_block_slots,
    const torch::Tensor& q_all,
    const torch::Tensor& skip_mask_i32,
    torch::Tensor& m_part,
    torch::Tensor& l_part,
    torch::Tensor& acc_part,
    int gqa_group,
    int group_size,
    int last_block_valid,
    int num_splits,
    float q_scale,
    cudaStream_t stream) {
    const int kv_heads = keys_int8.size(0);
    const int tokens = keys_int8.size(1);
    const int head_dim = keys_int8.size(2);
    const int d_v = values_int4_packed.size(2) * 2;
    const int groups = values_int4_scales.size(2);
    const int q_heads = q_all.size(0);
    const int blocks = keys_scale.size(1);
    const int scratch_tokens = values_fp16_scratch.size(1);
    const int blocks_per_split = (blocks + num_splits - 1) / num_splits;

    mixedv_partial_kernel<QueryT><<<q_heads * num_splits, kThreads, 0, stream>>>(
        keys_int8.data_ptr<int8_t>(),
        keys_scale.data_ptr<float>(),
        keys_zero_points.data_ptr<float>(),
        keys_fp16.data_ptr<at::Half>(),
        topk_mask.data_ptr<int32_t>(),
        values_int4_packed.data_ptr<uint8_t>(),
        values_int4_scales.data_ptr<at::Half>(),
        values_int4_zeros.data_ptr<at::Half>(),
        values_fp16_scratch.data_ptr<at::Half>(),
        value_fp16_mask.data_ptr<int32_t>(),
        value_block_slots.data_ptr<int32_t>(),
        q_all.data_ptr<QueryT>(),
        skip_mask_i32.data_ptr<int32_t>(),
        m_part.data_ptr<float>(),
        l_part.data_ptr<float>(),
        acc_part.data_ptr<float>(),
        kv_heads,
        tokens,
        head_dim,
        d_v,
        groups,
        q_heads,
        blocks,
        gqa_group,
        group_size,
        scratch_tokens,
        num_splits,
        blocks_per_split,
        last_block_valid,
        q_scale);
}

}  // namespace

torch::Tensor hybrid_mixedv_split_k_cuda_launcher(
    torch::Tensor keys_int8,
    torch::Tensor keys_scale,
    torch::Tensor keys_zero_points,
    torch::Tensor keys_fp16,
    torch::Tensor topk_mask,
    torch::Tensor values_int4_packed,
    torch::Tensor values_int4_scales,
    torch::Tensor values_int4_zeros,
    torch::Tensor values_fp16_scratch,
    torch::Tensor value_fp16_mask,
    torch::Tensor value_block_slots,
    torch::Tensor q_all,
    torch::Tensor skip_mask_i32,
    int64_t gqa_group,
    int64_t block_size,
    int64_t group_size,
    double q_scale,
    int64_t last_block_valid,
    int64_t num_splits) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(keys_int8));
    const int q_heads = q_all.size(0);
    const int d_v = values_int4_packed.size(2) * 2;
    const int splits = static_cast<int>(num_splits);

    auto opts = torch::TensorOptions().device(keys_int8.device()).dtype(torch::kFloat32);
    auto m_part = torch::empty({q_heads, splits}, opts);
    auto l_part = torch::empty({q_heads, splits}, opts);
    auto acc_part = torch::empty({q_heads, splits, d_v}, opts);
    auto output = torch::empty({q_heads, d_v}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (q_all.scalar_type() == torch::kFloat32) {
        launch_partial<float>(
            keys_int8, keys_scale, keys_zero_points, keys_fp16, topk_mask,
            values_int4_packed, values_int4_scales, values_int4_zeros,
            values_fp16_scratch, value_fp16_mask, value_block_slots, q_all,
            skip_mask_i32, m_part, l_part, acc_part, static_cast<int>(gqa_group),
            static_cast<int>(group_size), static_cast<int>(last_block_valid), splits,
            static_cast<float>(q_scale), stream);
    } else if (q_all.scalar_type() == torch::kFloat16) {
        launch_partial<at::Half>(
            keys_int8, keys_scale, keys_zero_points, keys_fp16, topk_mask,
            values_int4_packed, values_int4_scales, values_int4_zeros,
            values_fp16_scratch, value_fp16_mask, value_block_slots, q_all,
            skip_mask_i32, m_part, l_part, acc_part, static_cast<int>(gqa_group),
            static_cast<int>(group_size), static_cast<int>(last_block_valid), splits,
            static_cast<float>(q_scale), stream);
    } else {
        launch_partial<at::BFloat16>(
            keys_int8, keys_scale, keys_zero_points, keys_fp16, topk_mask,
            values_int4_packed, values_int4_scales, values_int4_zeros,
            values_fp16_scratch, value_fp16_mask, value_block_slots, q_all,
            skip_mask_i32, m_part, l_part, acc_part, static_cast<int>(gqa_group),
            static_cast<int>(group_size), static_cast<int>(last_block_valid), splits,
            static_cast<float>(q_scale), stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    mixedv_reduce_kernel<<<q_heads, kThreads, 0, stream>>>(
        m_part.data_ptr<float>(),
        l_part.data_ptr<float>(),
        acc_part.data_ptr<float>(),
        output.data_ptr<float>(),
        q_heads,
        d_v,
        splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

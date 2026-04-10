#include "full_attention.hip"

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

struct ScopedHipDevice {
    int previous = -1;
    bool changed = false;

    explicit ScopedHipDevice(int target) {
        hipGetDevice(&previous);
        if (previous != target) {
            hipSetDevice(target);
            changed = true;
        }
    }

    ~ScopedHipDevice() {
        if (changed && previous >= 0) {
            hipSetDevice(previous);
        }
    }
};

template <typename T>
int full_attention_prefill_device(
    int device_ordinal,
    int batch_size,
    int q_heads,
    int kv_heads,
    int q_len,
    int kv_len,
    int head_dim,
    int num_kv_groups,
    float scale,
    int seqlen_offset,
    const void* query,
    const void* key,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) {
        return 1;
    }

    const T* d_query = static_cast<const T*>(query);
    const T* d_key = static_cast<const T*>(key);
    const T* d_value = static_cast<const T*>(value);
    float* d_out = static_cast<float*>(out);
    unsigned int* d_row_counter = nullptr;

    if (hipMalloc(&d_row_counter, sizeof(unsigned int)) != hipSuccess) return 2;
    if (hipMemset(d_row_counter, 0, sizeof(unsigned int)) != hipSuccess) return 10;

    const int grid = props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    const int block = props.warpSize > 0 ? props.warpSize : 32;
    if (head_dim > block * 8) return 14;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_full_attention_prefill_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        q_heads,
        kv_heads,
        q_len,
        kv_len,
        head_dim,
        num_kv_groups,
        scale,
        seqlen_offset,
        d_query,
        d_key,
        d_value,
        d_out,
        d_row_counter);
    if (hipGetLastError() != hipSuccess) return 11;
    if (hipDeviceSynchronize() != hipSuccess) return 12;

    hipFree(d_row_counter);
    return 0;
}

template <typename T>
int linear_prefill_conv_pack_device(
    int device_ordinal,
    int batch_size,
    int conv_dim,
    int total_len,
    int seq_len,
    int kernel_size,
    const void* mixed_qkv,
    const void* weights,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const size_t out_elems = static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) *
        static_cast<size_t>(conv_dim);
    const unsigned int grid = static_cast<unsigned int>((out_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_linear_prefill_conv_pack_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        conv_dim,
        total_len,
        seq_len,
        kernel_size,
        static_cast<const T*>(mixed_qkv),
        static_cast<const T*>(weights),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 60;
    if (hipDeviceSynchronize() != hipSuccess) return 61;
    return 0;
}

template <typename T>
int linear_stateful_conv_device(
    int device_ordinal,
    int batch_size,
    int conv_dim,
    int seq_len,
    int state_len,
    int kernel_size,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const size_t out_elems = static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) *
        static_cast<size_t>(conv_dim);
    const unsigned int grid = static_cast<unsigned int>((out_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_linear_stateful_conv_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        conv_dim,
        seq_len,
        state_len,
        kernel_size,
        static_cast<const T*>(mixed_qkv),
        static_cast<const T*>(prev_state),
        static_cast<const T*>(weights),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 62;
    if (hipDeviceSynchronize() != hipSuccess) return 63;
    return 0;
}

template <typename T>
int linear_stateful_conv_value_decay_device(
    int device_ordinal,
    int batch_size,
    int conv_dim,
    int seq_len,
    int state_len,
    int kernel_size,
    int num_heads,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const size_t out_width = static_cast<size_t>(conv_dim) + static_cast<size_t>(num_heads);
    const size_t out_elems =
        static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * out_width;
    const unsigned int grid = static_cast<unsigned int>((out_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_linear_stateful_conv_value_decay_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        conv_dim,
        seq_len,
        state_len,
        kernel_size,
        num_heads,
        static_cast<const T*>(mixed_qkv),
        static_cast<const T*>(prev_state),
        static_cast<const T*>(weights),
        static_cast<const T*>(a),
        static_cast<const T*>(dt_bias),
        static_cast<const T*>(a_log_exp),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 64;
    if (hipDeviceSynchronize() != hipSuccess) return 65;
    return 0;
}

template <typename T>
int delta_recurrent_prefill_device(
    int device_ordinal,
    int batch_heads,
    int seq_len,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 69;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_recurrent_prefill_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        seq_len,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 67;
    if (hipDeviceSynchronize() != hipSuccess) return 68;
    return 0;
}

template <typename T>
int delta_chunk_single_prefill_device(
    int device_ordinal,
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (chunk_size > 64 || k_head_dim > 256) return 76;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_chunk_single_prefill_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 77;
    if (hipDeviceSynchronize() != hipSuccess) return 78;
    return 0;
}

template <typename T>
int delta_chunk_step_device(
    int device_ordinal,
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* prev_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 80;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_chunk_step_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(prev_state),
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 81;
    if (hipDeviceSynchronize() != hipSuccess) return 82;
    return 0;
}

template <typename T>
int delta_chunk_scan_raw_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 83;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_chunk_scan_raw_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 84;
    if (hipDeviceSynchronize() != hipSuccess) return 85;
    return 0;
}

template <typename T>
int delta_state_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 88;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_state_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(packed_scan),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 89;
    if (hipDeviceSynchronize() != hipSuccess) return 96;
    return 0;
}

template <typename T>
int delta_chunk_fused_device(
    int device_ordinal,
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* prev_state,
    const void* packed_chunk,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 97;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_chunk_fused_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(prev_state),
        static_cast<const T*>(packed_chunk),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 98;
    if (hipDeviceSynchronize() != hipSuccess) return 99;
    return 0;
}

template <typename T>
int delta_full_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* weighted_key_scan,
    const void* k_cumdecay_scan,
    const void* q_state_scan,
    const void* local_attn_scan,
    const void* state_decay_scan,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 100;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_full_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(weighted_key_scan),
        static_cast<const T*>(k_cumdecay_scan),
        static_cast<const T*>(q_state_scan),
        static_cast<const T*>(local_attn_scan),
        static_cast<const T*>(state_decay_scan),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 101;
    if (hipDeviceSynchronize() != hipSuccess) return 102;
    return 0;
}

template <typename T>
int delta_local_attn_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 112;
    constexpr int block = 256;
    const size_t total =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(num_chunks) *
        static_cast<size_t>(chunk_size) * static_cast<size_t>(chunk_size);
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_local_attn_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        static_cast<const T*>(query_scan),
        static_cast<const T*>(key_scan),
        static_cast<const T*>(exp_g_scan),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 113;
    if (hipDeviceSynchronize() != hipSuccess) return 114;
    return 0;
}

template <typename T>
int delta_base_attn_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const void* k_beta_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 115;
    constexpr int block = 256;
    const size_t total =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(num_chunks) *
        static_cast<size_t>(chunk_size) * static_cast<size_t>(chunk_size);
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_base_attn_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        static_cast<const T*>(k_beta_scan),
        static_cast<const T*>(key_scan),
        static_cast<const T*>(exp_g_scan),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 116;
    if (hipDeviceSynchronize() != hipSuccess) return 117;
    return 0;
}

template <typename T>
int delta_attn_solve_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    const void* base_attn_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (chunk_size > 64) return 118;
    constexpr int block = 1;
    const unsigned int grid =
        static_cast<unsigned int>(batch_heads * num_chunks);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_attn_solve_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        static_cast<const T*>(base_attn_scan),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 119;
    if (hipDeviceSynchronize() != hipSuccess) return 120;
    return 0;
}

template <typename T>
int swiglu_mul_device(
    int device_ordinal,
    int elem_count,
    const void* gate,
    const void* up,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((elem_count + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_swiglu_mul_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        elem_count,
        static_cast<const T*>(gate),
        static_cast<const T*>(up),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 121;
    if (hipDeviceSynchronize() != hipSuccess) return 122;
    return 0;
}

template <typename T, typename IndexT>
int embedding_lookup_device(
    int device_ordinal,
    int token_count,
    int vocab_size,
    int hidden_size,
    const void* embeddings,
    const void* indexes,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const int total_elems = token_count * hidden_size;
    const unsigned int grid = static_cast<unsigned int>((total_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_embedding_lookup_kernel<T, IndexT>),
        dim3(grid),
        dim3(block),
        0,
        0,
        token_count,
        vocab_size,
        hidden_size,
        static_cast<const T*>(embeddings),
        static_cast<const IndexT*>(indexes),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 123;
    if (hipDeviceSynchronize() != hipSuccess) return 124;
    return 0;
}

template <typename T>
int causal_mask_device(
    int device_ordinal,
    int batch_size,
    int tgt_len,
    int seqlen_offset,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const int kv_len = tgt_len + seqlen_offset;
    const int total_elems = batch_size * tgt_len * kv_len;
    const unsigned int grid = static_cast<unsigned int>((total_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_causal_mask_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        tgt_len,
        seqlen_offset,
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 125;
    if (hipDeviceSynchronize() != hipSuccess) return 126;
    return 0;
}

template <typename T>
int cumsum_last_dim_device(
    int device_ordinal,
    int rows,
    int cols,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_cumsum_last_dim_kernel<T>),
        dim3(static_cast<unsigned int>(rows)),
        dim3(1),
        0,
        0,
        rows,
        cols,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 127;
    if (hipDeviceSynchronize() != hipSuccess) return 128;
    return 0;
}

template <typename T>
int delta_full_scan_pack_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    const void* k_cumdecay_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 106;
    constexpr int block = 256;
    const size_t total_rows =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(num_chunks) *
        static_cast<size_t>(chunk_size);
    const unsigned int grid = static_cast<unsigned int>((total_rows + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_full_scan_pack_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        static_cast<const T*>(query_scan),
        static_cast<const T*>(key_scan),
        static_cast<const T*>(exp_g_scan),
        static_cast<const T*>(k_cumdecay_scan),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 107;
    if (hipDeviceSynchronize() != hipSuccess) return 108;
    return 0;
}

template <typename T>
int delta_full_scan_packed_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* local_attn_scan,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 109;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_delta_full_scan_packed_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(packed_scan),
        static_cast<const T*>(local_attn_scan),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 110;
    if (hipDeviceSynchronize() != hipSuccess) return 111;
    return 0;
}

template <typename T>
int l2norm_device(
    int device_ordinal,
    int n_rows,
    int n_cols,
    float eps,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_l2norm_kernel<T>),
        dim3(static_cast<unsigned int>(n_rows)),
        dim3(block),
        0,
        0,
        n_rows,
        n_cols,
        eps,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 90;
    if (hipDeviceSynchronize() != hipSuccess) return 91;
    return 0;
}

template <typename T>
int value_decay_device(
    int device_ordinal,
    int total_elems,
    int num_heads,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_value_decay_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        num_heads,
        static_cast<const T*>(a),
        static_cast<const T*>(dt_bias),
        static_cast<const T*>(a_log_exp),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 93;
    if (hipDeviceSynchronize() != hipSuccess) return 94;
    return 0;
}

template <typename T, bool ADD_UNIT_OFFSET>
int rms_norm_device(
    int device_ordinal,
    int n_rows,
    int n_cols,
    float eps,
    const void* xs,
    const void* weight,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_rms_norm_kernel<T, ADD_UNIT_OFFSET>),
        dim3(static_cast<unsigned int>(n_rows)),
        dim3(block),
        0,
        0,
        n_rows,
        n_cols,
        eps,
        static_cast<const T*>(xs),
        static_cast<const T*>(weight),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 71;
    if (hipDeviceSynchronize() != hipSuccess) return 72;
    return 0;
}

template <typename T>
int rms_norm_gated_device(
    int device_ordinal,
    int n_rows,
    int n_cols,
    float eps,
    const void* hidden,
    const void* gate,
    const void* weight,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dotcache_qwen35_rms_norm_gated_kernel<T>),
        dim3(static_cast<unsigned int>(n_rows)),
        dim3(block),
        0,
        0,
        n_rows,
        n_cols,
        eps,
        static_cast<const T*>(hidden),
        static_cast<const T*>(gate),
        static_cast<const T*>(weight),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 81;
    if (hipDeviceSynchronize() != hipSuccess) return 82;
    return 0;
}

} // namespace

extern "C" int dotcache_qwen35_hip_full_attention_prefill(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t q_heads,
    size_t kv_heads,
    size_t q_len,
    size_t kv_len,
    size_t head_dim,
    size_t num_kv_groups,
    float scale,
    size_t seqlen_offset,
    const void* query,
    const void* key,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return full_attention_prefill_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(q_heads),
            static_cast<int>(kv_heads),
            static_cast<int>(q_len),
            static_cast<int>(kv_len),
            static_cast<int>(head_dim),
            static_cast<int>(num_kv_groups),
            scale,
            static_cast<int>(seqlen_offset),
            query,
            key,
            value,
            out);
    case 1:
        return full_attention_prefill_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(q_heads),
            static_cast<int>(kv_heads),
            static_cast<int>(q_len),
            static_cast<int>(kv_len),
            static_cast<int>(head_dim),
            static_cast<int>(num_kv_groups),
            scale,
            static_cast<int>(seqlen_offset),
            query,
            key,
            value,
            out);
    case 2:
        return full_attention_prefill_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(q_heads),
            static_cast<int>(kv_heads),
            static_cast<int>(q_len),
            static_cast<int>(kv_len),
            static_cast<int>(head_dim),
            static_cast<int>(num_kv_groups),
            scale,
            static_cast<int>(seqlen_offset),
            query,
            key,
            value,
            out);
    default:
        return 64;
    }
}

extern "C" int dotcache_qwen35_hip_linear_prefill_conv_pack(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t conv_dim,
    size_t total_len,
    size_t seq_len,
    size_t kernel_size,
    const void* mixed_qkv,
    const void* weights,
    void* out) {
    switch (dtype) {
    case 0:
        return linear_prefill_conv_pack_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(total_len),
            static_cast<int>(seq_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            weights,
            out);
    case 1:
        return linear_prefill_conv_pack_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(total_len),
            static_cast<int>(seq_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            weights,
            out);
    case 2:
        return linear_prefill_conv_pack_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(total_len),
            static_cast<int>(seq_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            weights,
            out);
    default:
        return 62;
    }
}

extern "C" int dotcache_qwen35_hip_linear_stateful_conv(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t conv_dim,
    size_t seq_len,
    size_t state_len,
    size_t kernel_size,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    void* out) {
    switch (dtype) {
    case 0:
        return linear_stateful_conv_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            prev_state,
            weights,
            out);
    case 1:
        return linear_stateful_conv_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            prev_state,
            weights,
            out);
    case 2:
        return linear_stateful_conv_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            prev_state,
            weights,
            out);
    default:
        return 63;
    }
}

extern "C" int dotcache_qwen35_hip_delta_recurrent_prefill(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t seq_len,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_recurrent_prefill_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(seq_len),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 1:
        return delta_recurrent_prefill_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(seq_len),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 2:
        return delta_recurrent_prefill_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(seq_len),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    default:
        return 66;
    }
}

extern "C" int dotcache_qwen35_hip_linear_stateful_conv_value_decay(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t conv_dim,
    size_t seq_len,
    size_t state_len,
    size_t kernel_size,
    size_t num_heads,
    const void* mixed_qkv,
    const void* prev_state,
    const void* weights,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out) {
    switch (dtype) {
    case 0:
        return linear_stateful_conv_value_decay_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(num_heads),
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out);
    case 1:
        return linear_stateful_conv_value_decay_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(num_heads),
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out);
    case 2:
        return linear_stateful_conv_value_decay_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(seq_len),
            static_cast<int>(state_len),
            static_cast<int>(kernel_size),
            static_cast<int>(num_heads),
            mixed_qkv,
            prev_state,
            weights,
            a,
            dt_bias,
            a_log_exp,
            out);
    default:
        return 67;
    }
}

extern "C" int dotcache_qwen35_hip_delta_chunk_single_prefill(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_single_prefill_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            query,
            key,
            value,
            beta,
            g,
            out);
    case 1:
        return delta_chunk_single_prefill_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            query,
            key,
            value,
            beta,
            g,
            out);
    case 2:
        return delta_chunk_single_prefill_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            query,
            key,
            value,
            beta,
            g,
            out);
    default:
        return 79;
    }
}

extern "C" int dotcache_qwen35_hip_delta_chunk_step(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* prev_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_step_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 1:
        return delta_chunk_step_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 2:
        return delta_chunk_step_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    default:
        return 86;
    }
}

extern "C" int dotcache_qwen35_hip_delta_chunk_scan_raw(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_scan_raw_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 1:
        return delta_chunk_scan_raw_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 2:
        return delta_chunk_scan_raw_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    default:
        return 87;
    }
}

extern "C" int dotcache_qwen35_hip_delta_state_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_state_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            value,
            out);
    case 1:
        return delta_state_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            value,
            out);
    case 2:
        return delta_state_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            value,
            out);
    default:
        return 103;
    }
}

extern "C" int dotcache_qwen35_hip_delta_chunk_fused(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* prev_state,
    const void* packed_chunk,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_fused_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            packed_chunk,
            value,
            out);
    case 1:
        return delta_chunk_fused_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            packed_chunk,
            value,
            out);
    case 2:
        return delta_chunk_fused_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            packed_chunk,
            value,
            out);
    default:
        return 104;
    }
}

extern "C" int dotcache_qwen35_hip_delta_full_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* weighted_key_scan,
    const void* k_cumdecay_scan,
    const void* q_state_scan,
    const void* local_attn_scan,
    const void* state_decay_scan,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_full_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            weighted_key_scan,
            k_cumdecay_scan,
            q_state_scan,
            local_attn_scan,
            state_decay_scan,
            value,
            out);
    case 1:
        return delta_full_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            weighted_key_scan,
            k_cumdecay_scan,
            q_state_scan,
            local_attn_scan,
            state_decay_scan,
            value,
            out);
    case 2:
        return delta_full_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            weighted_key_scan,
            k_cumdecay_scan,
            q_state_scan,
            local_attn_scan,
            state_decay_scan,
            value,
            out);
    default:
        return 105;
    }
}

extern "C" int dotcache_qwen35_hip_delta_full_scan_pack(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    const void* k_cumdecay_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_full_scan_pack_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            k_cumdecay_scan,
            out);
    case 1:
        return delta_full_scan_pack_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            k_cumdecay_scan,
            out);
    case 2:
        return delta_full_scan_pack_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            k_cumdecay_scan,
            out);
    default:
        return 112;
    }
}

extern "C" int dotcache_qwen35_hip_delta_local_attn_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_local_attn_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            out);
    case 1:
        return delta_local_attn_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            out);
    case 2:
        return delta_local_attn_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            out);
    default:
        return 114;
    }
}

extern "C" int dotcache_qwen35_hip_delta_base_attn_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    const void* k_beta_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_base_attn_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    case 1:
        return delta_base_attn_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    case 2:
        return delta_base_attn_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    default:
        return 117;
    }
}

extern "C" int dotcache_qwen35_hip_delta_attn_solve_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    const void* base_attn_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_attn_solve_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            base_attn_scan,
            out);
    case 1:
        return delta_attn_solve_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            base_attn_scan,
            out);
    case 2:
        return delta_attn_solve_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            base_attn_scan,
            out);
    default:
        return 120;
    }
}

extern "C" int dotcache_qwen35_hip_swiglu_mul(
    int dtype,
    size_t device_ordinal,
    size_t elem_count,
    const void* gate,
    const void* up,
    void* out) {
    switch (dtype) {
    case 0:
        return swiglu_mul_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(elem_count),
            gate,
            up,
            out);
    case 1:
        return swiglu_mul_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(elem_count),
            gate,
            up,
            out);
    case 2:
        return swiglu_mul_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(elem_count),
            gate,
            up,
            out);
    default:
        return 122;
    }
}

extern "C" int dotcache_qwen35_hip_embedding_lookup(
    int dtype,
    int index_dtype,
    size_t device_ordinal,
    size_t token_count,
    size_t vocab_size,
    size_t hidden_size,
    const void* embeddings,
    const void* indexes,
    void* out) {
    switch (dtype) {
    case 0:
        switch (index_dtype) {
        case 0:
            return embedding_lookup_device<half, uint8_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 1:
            return embedding_lookup_device<half, uint32_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 2:
            return embedding_lookup_device<half, int64_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        default:
            return 123;
        }
    case 1:
        switch (index_dtype) {
        case 0:
            return embedding_lookup_device<float, uint8_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 1:
            return embedding_lookup_device<float, uint32_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 2:
            return embedding_lookup_device<float, int64_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        default:
            return 123;
        }
    case 2:
        switch (index_dtype) {
        case 0:
            return embedding_lookup_device<hip_bfloat16, uint8_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 1:
            return embedding_lookup_device<hip_bfloat16, uint32_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 2:
            return embedding_lookup_device<hip_bfloat16, int64_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        default:
            return 123;
        }
    default:
        return 124;
    }
}

extern "C" int dotcache_qwen35_hip_causal_mask(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t tgt_len,
    size_t seqlen_offset,
    void* out) {
    switch (dtype) {
    case 0:
        return causal_mask_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(tgt_len),
            static_cast<int>(seqlen_offset),
            out);
    case 1:
        return causal_mask_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(tgt_len),
            static_cast<int>(seqlen_offset),
            out);
    case 2:
        return causal_mask_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(tgt_len),
            static_cast<int>(seqlen_offset),
            out);
    default:
        return 126;
    }
}

extern "C" int dotcache_qwen35_hip_cumsum_last_dim(
    int dtype,
    size_t device_ordinal,
    size_t rows,
    size_t cols,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return cumsum_last_dim_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(cols),
            xs,
            out);
    case 1:
        return cumsum_last_dim_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(cols),
            xs,
            out);
    case 2:
        return cumsum_last_dim_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(cols),
            xs,
            out);
    default:
        return 128;
    }
}

extern "C" int dotcache_qwen35_hip_delta_full_scan_packed(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* local_attn_scan,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_full_scan_packed_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            local_attn_scan,
            value,
            out);
    case 1:
        return delta_full_scan_packed_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            local_attn_scan,
            value,
            out);
    case 2:
        return delta_full_scan_packed_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            local_attn_scan,
            value,
            out);
    default:
        return 113;
    }
}

extern "C" int dotcache_qwen35_hip_l2norm(
    int dtype,
    size_t device_ordinal,
    size_t n_rows,
    size_t n_cols,
    float eps,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return l2norm_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            xs,
            out);
    case 1:
        return l2norm_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            xs,
            out);
    case 2:
        return l2norm_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            xs,
            out);
    default:
        return 92;
    }
}

extern "C" int dotcache_qwen35_hip_value_decay(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    size_t num_heads,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out) {
    switch (dtype) {
    case 0:
        return value_decay_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            static_cast<int>(num_heads),
            a,
            dt_bias,
            a_log_exp,
            out);
    case 1:
        return value_decay_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            static_cast<int>(num_heads),
            a,
            dt_bias,
            a_log_exp,
            out);
    case 2:
        return value_decay_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            static_cast<int>(num_heads),
            a,
            dt_bias,
            a_log_exp,
            out);
    default:
        return 95;
    }
}

extern "C" int dotcache_qwen35_hip_rms_norm(
    int dtype,
    size_t device_ordinal,
    size_t n_rows,
    size_t n_cols,
    float eps,
    int add_unit_offset,
    const void* xs,
    const void* weight,
    void* out) {
    switch (dtype) {
    case 0:
        return add_unit_offset
            ? rms_norm_device<half, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out)
            : rms_norm_device<half, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out);
    case 1:
        return add_unit_offset
            ? rms_norm_device<float, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out)
            : rms_norm_device<float, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out);
    case 2:
        return add_unit_offset
            ? rms_norm_device<hip_bfloat16, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out)
            : rms_norm_device<hip_bfloat16, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out);
    default:
        return 74;
    }
}

extern "C" int dotcache_qwen35_hip_rms_norm_gated(
    int dtype,
    size_t device_ordinal,
    size_t n_rows,
    size_t n_cols,
    float eps,
    const void* hidden,
    const void* gate,
    const void* weight,
    void* out) {
    switch (dtype) {
    case 0:
        return rms_norm_gated_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            hidden,
            gate,
            weight,
            out);
    case 1:
        return rms_norm_gated_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            hidden,
            gate,
            weight,
            out);
    case 2:
        return rms_norm_gated_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            hidden,
            gate,
            weight,
            out);
    default:
        return 84;
    }
}

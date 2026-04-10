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
int full_attention_prefill_host(
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

    const size_t query_elems =
        static_cast<size_t>(batch_size) * q_heads * q_len * head_dim;
    const size_t key_elems =
        static_cast<size_t>(batch_size) * kv_heads * kv_len * head_dim;
    const size_t out_elems = query_elems;

    T* d_query = nullptr;
    T* d_key = nullptr;
    T* d_value = nullptr;
    T* d_out = nullptr;
    unsigned int* d_row_counter = nullptr;

    if (hipMalloc(&d_query, query_elems * sizeof(T)) != hipSuccess) return 2;
    if (hipMalloc(&d_key, key_elems * sizeof(T)) != hipSuccess) return 3;
    if (hipMalloc(&d_value, key_elems * sizeof(T)) != hipSuccess) return 4;
    if (hipMalloc(&d_out, out_elems * sizeof(T)) != hipSuccess) return 5;
    if (hipMalloc(&d_row_counter, sizeof(unsigned int)) != hipSuccess) return 6;

    if (hipMemcpy(d_query, query, query_elems * sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
        return 7;
    if (hipMemcpy(d_key, key, key_elems * sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
        return 8;
    if (hipMemcpy(d_value, value, key_elems * sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
        return 9;
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
    if (hipMemcpy(out, d_out, out_elems * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess)
        return 13;

    hipFree(d_row_counter);
    hipFree(d_out);
    hipFree(d_value);
    hipFree(d_key);
    hipFree(d_query);
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
        return full_attention_prefill_host<half>(
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
        return full_attention_prefill_host<float>(
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
        return full_attention_prefill_host<hip_bfloat16>(
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

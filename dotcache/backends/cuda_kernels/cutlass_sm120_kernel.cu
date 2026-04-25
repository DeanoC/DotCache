#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cutlass/version.h>

namespace {

__global__ void probe_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t n) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

}  // namespace

torch::Tensor cutlass_sm120_probe_launcher(torch::Tensor input) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    auto output = torch::empty_like(input);
    const int64_t n = input.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    if (n > 0) {
        probe_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            n);
    }
    return output;
}

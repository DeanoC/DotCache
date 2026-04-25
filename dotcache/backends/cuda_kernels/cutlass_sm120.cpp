#include <torch/extension.h>

#include <sstream>
#include <string>

#include <cutlass/version.h>

torch::Tensor cutlass_sm120_probe_launcher(torch::Tensor input);

std::string cutlass_sm120_metadata() {
    std::ostringstream out;
    out << "cutlass=" << CUTLASS_MAJOR << "." << CUTLASS_MINOR << "." << CUTLASS_PATCH
        << " target=sm120";
    return out.str();
}

torch::Tensor cutlass_sm120_probe(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    return cutlass_sm120_probe_launcher(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_sm120_metadata", &cutlass_sm120_metadata, "CUTLASS SM120 backend metadata");
    m.def("cutlass_sm120_probe", &cutlass_sm120_probe, "CUTLASS SM120 CUDA ABI probe");
}

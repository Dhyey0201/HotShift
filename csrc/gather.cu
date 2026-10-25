#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "checks.h"

namespace {

__global__ void gather_kernel(
    const float* cache_rows,
    const float* miss_rows,
    const int32_t* source_index,
    float* output,
    int64_t unique_count,
    int64_t dimensions,
    int64_t cache_capacity,
    int64_t miss_capacity) {
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = unique_count * dimensions;
    if (flat >= total) return;
    const int64_t item_index = flat / dimensions;
    const int64_t dimension = flat % dimensions;
    const int32_t source = source_index[item_index];
    const int64_t row = source >= 0 ? source : -static_cast<int64_t>(source) - 1;
    const float* rows = source >= 0 ? cache_rows : miss_rows;
    if ((source >= 0 && row >= cache_capacity) || (source < 0 && row >= miss_capacity)) {
        output[flat] = 0.0f;
        return;
    }
    output[flat] = rows[row * dimensions + dimension];
}

}

torch::Tensor gather_rows_cuda(
    torch::Tensor cache_rows,
    torch::Tensor miss_rows,
    torch::Tensor source_index) {
    check_cuda_tensor(cache_rows, "cache_rows");
    check_cuda_tensor(miss_rows, "miss_rows");
    check_cuda_tensor(source_index, "source_index");
    check_same_device(cache_rows, miss_rows, "row buffers");
    check_same_device(cache_rows, source_index, "gather inputs");
    TORCH_CHECK(cache_rows.scalar_type() == torch::kFloat32 && miss_rows.scalar_type() == torch::kFloat32,
        "row buffers must be float32");
    TORCH_CHECK(source_index.scalar_type() == torch::kInt32, "source_index must be int32");
    TORCH_CHECK(cache_rows.dim() == 2 && miss_rows.dim() == 2 && source_index.dim() == 1,
        "gather input dimensions are invalid");
    TORCH_CHECK(cache_rows.size(1) == miss_rows.size(1), "row buffer dimensions differ");
    TORCH_CHECK(cache_rows.is_contiguous() && miss_rows.is_contiguous() && source_index.is_contiguous(),
        "gather inputs must be contiguous");
    const int64_t dimensions = cache_rows.size(1);
    auto output = torch::empty({source_index.numel(), dimensions}, cache_rows.options());
    const int64_t total = source_index.numel() * dimensions;
    if (total == 0) return output;
    c10::cuda::CUDAGuard guard(cache_rows.device());
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    gather_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        cache_rows.data_ptr<float>(), miss_rows.data_ptr<float>(), source_index.data_ptr<int32_t>(),
        output.data_ptr<float>(), source_index.numel(), dimensions, cache_rows.size(0), miss_rows.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


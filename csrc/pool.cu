#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <utility>
#include <vector>
#include "checks.h"

namespace {

__global__ void pool_kernel(
    const float* cache_rows,
    const float* miss_rows,
    const int32_t* source_index,
    const int64_t* unique_inverse,
    const int64_t* offsets,
    float* output,
    int64_t bags,
    int64_t dimensions,
    int64_t cache_capacity,
    int64_t miss_capacity,
    int64_t unique_count,
    int64_t token_count) {
    const int64_t bag = blockIdx.x;
    const int64_t dimension = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (bag >= bags || dimension >= dimensions) return;
    const int64_t begin = offsets[bag];
    const int64_t end = offsets[bag + 1];
    if (begin < 0 || end < begin || end > token_count) {
        output[bag * dimensions + dimension] = 0.0f;
        return;
    }
    float sum = 0.0f;
    for (int64_t token = begin; token < end; ++token) {
        const int64_t unique_index = unique_inverse[token];
        if (unique_index < 0 || unique_index >= unique_count) continue;
        const int32_t source = source_index[unique_index];
        const int64_t row = source >= 0 ? source : -static_cast<int64_t>(source) - 1;
        const float* rows = source >= 0 ? cache_rows : miss_rows;
        if ((source >= 0 && row >= cache_capacity) || (source < 0 && row >= miss_capacity)) continue;
        sum += rows[row * dimensions + dimension];
    }
    output[bag * dimensions + dimension] = end > begin ? sum / static_cast<float>(end - begin) : 0.0f;
}

}

torch::Tensor pool_mean_cuda(
    torch::Tensor cache_rows,
    torch::Tensor miss_rows,
    torch::Tensor source_index,
    torch::Tensor unique_inverse,
    torch::Tensor offsets) {
    for (const auto& entry : std::vector<std::pair<torch::Tensor, const char*>>{
             {cache_rows, "cache_rows"}, {miss_rows, "miss_rows"}, {source_index, "source_index"},
             {unique_inverse, "unique_inverse"}, {offsets, "offsets"}}) {
        check_cuda_tensor(entry.first, entry.second);
        check_same_device(cache_rows, entry.first, "pool inputs");
        TORCH_CHECK(entry.first.is_contiguous(), entry.second, " must be contiguous");
    }
    TORCH_CHECK(cache_rows.scalar_type() == torch::kFloat32 && miss_rows.scalar_type() == torch::kFloat32,
        "row buffers must be float32");
    TORCH_CHECK(source_index.scalar_type() == torch::kInt32, "source_index must be int32");
    TORCH_CHECK(unique_inverse.scalar_type() == torch::kInt64 && offsets.scalar_type() == torch::kInt64,
        "inverse indices and offsets must be int64");
    TORCH_CHECK(cache_rows.dim() == 2 && miss_rows.dim() == 2 && source_index.dim() == 1 &&
        unique_inverse.dim() == 1 && offsets.dim() == 1, "pool input dimensions are invalid");
    TORCH_CHECK(cache_rows.size(1) == miss_rows.size(1), "row buffer dimensions differ");
    TORCH_CHECK(offsets.numel() >= 1, "offsets must contain at least the initial zero");
    const int64_t bags = offsets.numel() - 1;
    const int64_t dimensions = cache_rows.size(1);
    auto output = torch::empty({bags, dimensions}, cache_rows.options());
    if (bags == 0 || dimensions == 0) return output;
    c10::cuda::CUDAGuard guard(cache_rows.device());
    constexpr int threads = 128;
    const dim3 block(threads);
    const dim3 grid(static_cast<unsigned int>(bags), static_cast<unsigned int>((dimensions + threads - 1) / threads));
    pool_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        cache_rows.data_ptr<float>(), miss_rows.data_ptr<float>(), source_index.data_ptr<int32_t>(),
        unique_inverse.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), output.data_ptr<float>(),
        bags, dimensions, cache_rows.size(0), miss_rows.size(0), source_index.numel(), unique_inverse.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


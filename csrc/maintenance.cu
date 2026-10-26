#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "checks.h"

namespace {

// One thread per update entry. Callers deduplicate keys and slots, so each map entry has
// at most one writer within a launch.
__global__ void mapping_kernel(
    int32_t* item_to_slot,
    const int64_t* old_keys,
    const int64_t* new_keys,
    const int32_t* slots,
    int64_t count,
    int64_t item_count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int64_t old_key = old_keys[index];
    const int64_t new_key = new_keys[index];
    const int32_t slot = slots[index];
    if (old_key >= 0 && old_key < item_count) item_to_slot[old_key] = -1;
    if (new_key >= 0 && new_key < item_count) item_to_slot[new_key] = slot;
}

// Copies packed source rows into reserved cache slots. Each (entry, dimension) output
// coordinate has exactly one writer; row offsets use 64-bit arithmetic.
__global__ void scatter_rows_kernel(
    float* cache_rows,
    const float* source_rows,
    const int64_t* source_positions,
    const int32_t* slots,
    int64_t count,
    int64_t dimensions,
    int64_t cache_capacity,
    int64_t source_count) {
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count * dimensions) return;
    const int64_t entry = flat / dimensions;
    const int64_t dimension = flat % dimensions;
    const int64_t slot = slots[entry];
    const int64_t source = source_positions[entry];
    if (slot < 0 || slot >= cache_capacity || source < 0 || source >= source_count) return;
    cache_rows[slot * dimensions + dimension] = source_rows[source * dimensions + dimension];
}

}

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void check_same_device(const torch::Tensor& left, const torch::Tensor& right, const char* name) {
    TORCH_CHECK(left.device() == right.device(), name, " must share a CUDA device");
}

void update_mapping_cuda(
    torch::Tensor item_to_slot,
    torch::Tensor old_keys,
    torch::Tensor new_keys,
    torch::Tensor slots) {
    check_cuda_tensor(item_to_slot, "item_to_slot");
    check_cuda_tensor(old_keys, "old_keys");
    check_cuda_tensor(new_keys, "new_keys");
    check_cuda_tensor(slots, "slots");
    check_same_device(item_to_slot, old_keys, "mapping inputs");
    check_same_device(item_to_slot, new_keys, "mapping inputs");
    check_same_device(item_to_slot, slots, "mapping inputs");
    TORCH_CHECK(item_to_slot.scalar_type() == torch::kInt32 && slots.scalar_type() == torch::kInt32,
        "mapping and slot tensors must be int32");
    TORCH_CHECK(old_keys.scalar_type() == torch::kInt64 && new_keys.scalar_type() == torch::kInt64,
        "mapping keys must be int64");
    TORCH_CHECK(item_to_slot.dim() == 1 && old_keys.dim() == 1 && new_keys.dim() == 1 && slots.dim() == 1,
        "mapping inputs must be one-dimensional");
    TORCH_CHECK(old_keys.numel() == new_keys.numel() && old_keys.numel() == slots.numel(),
        "mapping update lengths differ");
    TORCH_CHECK(item_to_slot.is_contiguous() && old_keys.is_contiguous() && new_keys.is_contiguous() && slots.is_contiguous(),
        "mapping inputs must be contiguous");
    if (slots.numel() == 0) return;
    c10::cuda::CUDAGuard guard(item_to_slot.device());
    constexpr int threads = 256;
    const int blocks = static_cast<int>((slots.numel() + threads - 1) / threads);
    mapping_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        item_to_slot.data_ptr<int32_t>(), old_keys.data_ptr<int64_t>(), new_keys.data_ptr<int64_t>(),
        slots.data_ptr<int32_t>(), slots.numel(), item_to_slot.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_rows_cuda(
    torch::Tensor cache_rows,
    torch::Tensor source_rows,
    torch::Tensor source_positions,
    torch::Tensor slots) {
    check_cuda_tensor(cache_rows, "cache_rows");
    check_cuda_tensor(source_rows, "source_rows");
    check_cuda_tensor(source_positions, "source_positions");
    check_cuda_tensor(slots, "slots");
    check_same_device(cache_rows, source_rows, "scatter inputs");
    check_same_device(cache_rows, source_positions, "scatter inputs");
    check_same_device(cache_rows, slots, "scatter inputs");
    TORCH_CHECK(cache_rows.scalar_type() == torch::kFloat32 && source_rows.scalar_type() == torch::kFloat32,
        "row buffers must be float32");
    TORCH_CHECK(source_positions.scalar_type() == torch::kInt64, "source_positions must be int64");
    TORCH_CHECK(slots.scalar_type() == torch::kInt32, "slots must be int32");
    TORCH_CHECK(cache_rows.dim() == 2 && source_rows.dim() == 2 && source_positions.dim() == 1 && slots.dim() == 1,
        "scatter input dimensions are invalid");
    TORCH_CHECK(cache_rows.size(1) == source_rows.size(1), "row buffer dimensions differ");
    TORCH_CHECK(source_positions.numel() == slots.numel(), "scatter update lengths differ");
    TORCH_CHECK(cache_rows.is_contiguous() && source_rows.is_contiguous() && source_positions.is_contiguous() && slots.is_contiguous(),
        "scatter inputs must be contiguous");
    const int64_t dimensions = cache_rows.size(1);
    const int64_t total = slots.numel() * dimensions;
    if (total == 0) return;
    c10::cuda::CUDAGuard guard(cache_rows.device());
    constexpr int threads = 256;
    const int64_t blocks = (total + threads - 1) / threads;
    TORCH_CHECK(blocks <= 2147483647LL, "scatter launch exceeds the supported grid size");
    scatter_rows_kernel<<<static_cast<unsigned int>(blocks), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        cache_rows.data_ptr<float>(), source_rows.data_ptr<float>(), source_positions.data_ptr<int64_t>(),
        slots.data_ptr<int32_t>(), slots.numel(), dimensions, cache_rows.size(0), source_rows.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

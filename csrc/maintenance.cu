#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "checks.h"

namespace {

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


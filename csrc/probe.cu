#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "checks.h"

namespace {

__global__ void probe_kernel(
    const int64_t* ids,
    const int32_t* item_to_slot,
    int32_t* slots,
    int64_t count,
    int64_t item_count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int64_t item = ids[index];
    slots[index] = (item >= 0 && item < item_count) ? item_to_slot[item] : -1;
}

}

torch::Tensor probe_slots_cuda(torch::Tensor ids, torch::Tensor item_to_slot) {
    check_cuda_tensor(ids, "ids");
    check_cuda_tensor(item_to_slot, "item_to_slot");
    check_same_device(ids, item_to_slot, "probe inputs");
    TORCH_CHECK(ids.scalar_type() == torch::kInt64, "ids must be int64");
    TORCH_CHECK(item_to_slot.scalar_type() == torch::kInt32, "item_to_slot must be int32");
    TORCH_CHECK(ids.dim() == 1 && item_to_slot.dim() == 1, "probe inputs must be one-dimensional");
    TORCH_CHECK(ids.is_contiguous() && item_to_slot.is_contiguous(), "probe inputs must be contiguous");
    auto slots = torch::empty(ids.sizes(), ids.options().dtype(torch::kInt32));
    if (ids.numel() == 0) return slots;
    c10::cuda::CUDAGuard guard(ids.device());
    constexpr int threads = 256;
    const int blocks = static_cast<int>((ids.numel() + threads - 1) / threads);
    probe_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int64_t>(), item_to_slot.data_ptr<int32_t>(), slots.data_ptr<int32_t>(),
        ids.numel(), item_to_slot.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return slots;
}


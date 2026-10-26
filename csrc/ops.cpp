#include "checks.h"

torch::Tensor probe_slots_cuda(torch::Tensor ids, torch::Tensor item_to_slot);
torch::Tensor gather_rows_cuda(
    torch::Tensor cache_rows,
    torch::Tensor miss_rows,
    torch::Tensor source_index);
torch::Tensor pool_mean_cuda(
    torch::Tensor cache_rows,
    torch::Tensor miss_rows,
    torch::Tensor source_index,
    torch::Tensor unique_inverse,
    torch::Tensor offsets);
void update_mapping_cuda(
    torch::Tensor item_to_slot,
    torch::Tensor old_keys,
    torch::Tensor new_keys,
    torch::Tensor slots);
void scatter_rows_cuda(
    torch::Tensor cache_rows,
    torch::Tensor source_rows,
    torch::Tensor source_positions,
    torch::Tensor slots);

// Inference-only operators: no autograd kernels are registered, so calls that require
// gradients fail instead of silently producing untracked results.
TORCH_LIBRARY(hotshift, library) {
    library.def("probe_slots(Tensor ids, Tensor item_to_slot) -> Tensor");
    library.def("gather_rows(Tensor cache_rows, Tensor miss_rows, Tensor source_index) -> Tensor");
    library.def(
        "pool_mean(Tensor cache_rows, Tensor miss_rows, Tensor source_index, Tensor unique_inverse, Tensor offsets) -> Tensor");
    library.def(
        "update_mapping_(Tensor(a!) item_to_slot, Tensor old_keys, Tensor new_keys, Tensor slots) -> ()");
    library.def(
        "scatter_rows_(Tensor(a!) cache_rows, Tensor source_rows, Tensor source_positions, Tensor slots) -> ()");
}

TORCH_LIBRARY_IMPL(hotshift, CUDA, library) {
    library.impl("probe_slots", TORCH_FN(probe_slots_cuda));
    library.impl("gather_rows", TORCH_FN(gather_rows_cuda));
    library.impl("pool_mean", TORCH_FN(pool_mean_cuda));
    library.impl("update_mapping_", TORCH_FN(update_mapping_cuda));
    library.impl("scatter_rows_", TORCH_FN(scatter_rows_cuda));
}

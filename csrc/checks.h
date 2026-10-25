#pragma once

#include <torch/extension.h>

void check_cuda_tensor(const torch::Tensor& tensor, const char* name);
void check_same_device(const torch::Tensor& left, const torch::Tensor& right, const char* name);


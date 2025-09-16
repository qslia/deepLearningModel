#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// PyTorch 2.6.0 compatibility fixes
#define TORCH_CHECK_ARG(cond, arg, msg) TORCH_CHECK(cond, msg)
#define TORCH_CHECK_TYPE(tensor, type) TORCH_CHECK(tensor.scalar_type() == type, "Expected " #type " tensor")

// Disable problematic half precision operations for PyTorch 2.6.0
#define __CUDA_NO_HALF_OPERATORS__
#define __CUDA_NO_HALF_CONVERSIONS__ 
#define __CUDA_NO_BFLOAT16_CONVERSIONS__
#define __CUDA_NO_HALF2_OPERATORS__

// Simple CUDA kernel loop macro
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// Simplified CUDA kernel for 2D convolution forward pass (PyTorch 2.6.0 compatible)
__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int output_height,
    int output_width) {
    
    CUDA_KERNEL_LOOP(idx, batch_size * out_channels * output_height * output_width) {
        const int w = idx % output_width;
        const int h = (idx / output_width) % output_height;
        const int oc = (idx / (output_width * output_height)) % out_channels;
        const int b = idx / (out_channels * output_height * output_width);
        
        float sum = 0.0f;
        
        // Convolution operation
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int input_h = h + kh;
                    const int input_w = w + kw;
                    
                    if (input_h < input_height && input_w < input_width) {
                        const int input_idx = b * in_channels * input_height * input_width +
                                             ic * input_height * input_width +
                                             input_h * input_width + input_w;
                        const int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                              ic * kernel_size * kernel_size +
                                              kh * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        output[idx] = sum + bias[oc];
    }
}

// CUDA kernel for 2D max pooling forward pass
__global__ void maxpool2d_forward_kernel(
    const float* input,
    float* output,
    int* indices,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int output_height,
    int output_width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * channels * output_height * output_width;
    
    if (idx < total_output_elements) {
        int w = idx % output_width;
        int h = (idx / output_width) % output_height;
        int c = (idx / (output_width * output_height)) % channels;
        int b = idx / (channels * output_height * output_width);
        
        float max_val = -FLT_MAX;
        int max_idx = -1;
        
        // Max pooling operation
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int input_h = h * stride + kh;
                int input_w = w * stride + kw;
                
                if (input_h < input_height && input_w < input_width) {
                    int input_idx = b * channels * input_height * input_width +
                                   c * input_height * input_width +
                                   input_h * input_width + input_w;
                    if (input[input_idx] > max_val) {
                        max_val = input[input_idx];
                        max_idx = input_idx;
                    }
                }
            }
        }
        
        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

// CUDA kernel for ReLU activation forward pass
__global__ void relu_forward_kernel(
    const float* input,
    float* output,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// CUDA kernel for linear layer forward pass
__global__ void linear_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_features,
    int output_features) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * output_features;
    
    if (idx < total_output_elements) {
        int feature_idx = idx % output_features;
        int batch_idx = idx / output_features;
        
        float sum = 0.0f;
        for (int i = 0; i < input_features; i++) {
            int input_idx = batch_idx * input_features + i;
            int weight_idx = feature_idx * input_features + i;
            sum += input[input_idx] * weight[weight_idx];
        }
        
        sum += bias[feature_idx];
        output[idx] = sum;
    }
}

// CUDA kernel for cross entropy loss forward pass
__global__ void cross_entropy_forward_kernel(
    const float* logits,
    const int* targets,
    float* losses,
    int batch_size,
    int num_classes) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        // Find max for numerical stability
        float max_logit = -FLT_MAX;
        for (int i = 0; i < num_classes; i++) {
            max_logit = fmaxf(max_logit, logits[idx * num_classes + i]);
        }
        
        // Compute log sum exp
        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            sum_exp += expf(logits[idx * num_classes + i] - max_logit);
        }
        float log_sum_exp = max_logit + logf(sum_exp);
        
        // Compute cross entropy loss
        int target = targets[idx];
        losses[idx] = log_sum_exp - logits[idx * num_classes + target];
    }
}

// CUDA kernel for softmax forward pass
__global__ void softmax_forward_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_classes) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        // Find max for numerical stability
        float max_val = -FLT_MAX;
        for (int i = 0; i < num_classes; i++) {
            max_val = fmaxf(max_val, input[idx * num_classes + i]);
        }
        
        // Compute softmax
        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            float exp_val = expf(input[idx * num_classes + i] - max_val);
            output[idx * num_classes + i] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        for (int i = 0; i < num_classes; i++) {
            output[idx * num_classes + i] /= sum_exp;
        }
    }
}

// Backward pass kernels

// CUDA kernel for convolution backward pass (input gradients)
__global__ void conv2d_backward_input_kernel(
    const float* grad_output,
    const float* weight,
    float* grad_input,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int output_height,
    int output_width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_input_elements = batch_size * in_channels * input_height * input_width;
    
    if (idx < total_input_elements) {
        int w = idx % input_width;
        int h = (idx / input_width) % input_height;
        int ic = (idx / (input_width * input_height)) % in_channels;
        int b = idx / (in_channels * input_height * input_width);
        
        float sum = 0.0f;
        
        // Compute gradient w.r.t input
        for (int oc = 0; oc < out_channels; oc++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int output_h = h - kh;
                    int output_w = w - kw;
                    
                    if (output_h >= 0 && output_h < output_height && 
                        output_w >= 0 && output_w < output_width) {
                        int grad_output_idx = b * out_channels * output_height * output_width +
                                             oc * output_height * output_width +
                                             output_h * output_width + output_w;
                        int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                        ic * kernel_size * kernel_size +
                                        kh * kernel_size + kw;
                        sum += grad_output[grad_output_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        grad_input[idx] = sum;
    }
}

// CUDA kernel for convolution backward pass (weight gradients)
__global__ void conv2d_backward_weight_kernel(
    const float* grad_output,
    const float* input,
    float* grad_weight,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int output_height,
    int output_width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weight_elements = out_channels * in_channels * kernel_size * kernel_size;
    
    if (idx < total_weight_elements) {
        int kw = idx % kernel_size;
        int kh = (idx / kernel_size) % kernel_size;
        int ic = (idx / (kernel_size * kernel_size)) % in_channels;
        int oc = idx / (in_channels * kernel_size * kernel_size);
        
        float sum = 0.0f;
        
        // Compute gradient w.r.t weight
        for (int b = 0; b < batch_size; b++) {
            for (int oh = 0; oh < output_height; oh++) {
                for (int ow = 0; ow < output_width; ow++) {
                    int input_h = oh + kh;
                    int input_w = ow + kw;
                    
                    if (input_h < input_height && input_w < input_width) {
                        int grad_output_idx = b * out_channels * output_height * output_width +
                                             oc * output_height * output_width +
                                             oh * output_width + ow;
                        int input_idx = b * in_channels * input_height * input_width +
                                       ic * input_height * input_width +
                                       input_h * input_width + input_w;
                        sum += grad_output[grad_output_idx] * input[input_idx];
                    }
                }
            }
        }
        
        grad_weight[idx] = sum;
    }
}

// CUDA kernel for ReLU backward pass
__global__ void relu_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

// CUDA kernel for linear layer backward pass (input gradients)
__global__ void linear_backward_input_kernel(
    const float* grad_output,
    const float* weight,
    float* grad_input,
    int batch_size,
    int input_features,
    int output_features) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_input_elements = batch_size * input_features;
    
    if (idx < total_input_elements) {
        int feature_idx = idx % input_features;
        int batch_idx = idx / input_features;
        
        float sum = 0.0f;
        for (int i = 0; i < output_features; i++) {
            int grad_output_idx = batch_idx * output_features + i;
            int weight_idx = i * input_features + feature_idx;
            sum += grad_output[grad_output_idx] * weight[weight_idx];
        }
        
        grad_input[idx] = sum;
    }
}

// CUDA kernel for linear layer backward pass (weight gradients)
__global__ void linear_backward_weight_kernel(
    const float* grad_output,
    const float* input,
    float* grad_weight,
    int batch_size,
    int input_features,
    int output_features) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weight_elements = output_features * input_features;
    
    if (idx < total_weight_elements) {
        int input_feature_idx = idx % input_features;
        int output_feature_idx = idx / input_features;
        
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            int grad_output_idx = b * output_features + output_feature_idx;
            int input_idx = b * input_features + input_feature_idx;
            sum += grad_output[grad_output_idx] * input[input_idx];
        }
        
        grad_weight[idx] = sum;
    }
}

// Host functions for forward pass
torch::Tensor conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto output_height = input_height - kernel_size + 1;
    auto output_width = input_width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());
    
    const int threads = 256;
    const int blocks = (output.numel() + threads - 1) / threads;
    
    conv2d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width, kernel_size,
        output_height, output_width);
    
    return output;
}

std::vector<torch::Tensor> maxpool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride) {
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    
    auto output_height = (input_height - kernel_size) / stride + 1;
    auto output_width = (input_width - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, output_height, output_width}, input.options());
    auto indices = torch::zeros({batch_size, channels, output_height, output_width}, torch::dtype(torch::kInt32).device(input.device()));
    
    const int threads = 256;
    const int blocks = (output.numel() + threads - 1) / threads;
    
    maxpool2d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        indices.data_ptr<int>(),
        batch_size, channels,
        input_height, input_width,
        kernel_size, stride,
        output_height, output_width);
    
    return {output, indices};
}

torch::Tensor relu_cuda_forward(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    relu_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel());
    
    return output;
}

torch::Tensor linear_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto input_features = input.size(1);
    auto output_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, output_features}, input.options());
    
    const int threads = 256;
    const int blocks = (output.numel() + threads - 1) / threads;
    
    linear_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, input_features, output_features);
    
    return output;
}

torch::Tensor cross_entropy_cuda_forward(
    torch::Tensor logits,
    torch::Tensor targets) {
    
    auto batch_size = logits.size(0);
    auto num_classes = logits.size(1);
    
    auto losses = torch::zeros({batch_size}, logits.options());
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    cross_entropy_forward_kernel<<<blocks, threads>>>(
        logits.data_ptr<float>(),
        targets.data_ptr<int>(),
        losses.data_ptr<float>(),
        batch_size, num_classes);
    
    return losses;
}

torch::Tensor softmax_cuda_forward(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto num_classes = input.size(1);
    
    auto output = torch::zeros_like(input);
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    softmax_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, num_classes);
    
    return output;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_forward", &conv2d_cuda_forward, "2D Convolution forward (CUDA)");
    m.def("maxpool2d_forward", &maxpool2d_cuda_forward, "2D Max Pooling forward (CUDA)");
    m.def("relu_forward", &relu_cuda_forward, "ReLU forward (CUDA)");
    m.def("linear_forward", &linear_cuda_forward, "Linear forward (CUDA)");
    m.def("cross_entropy_forward", &cross_entropy_cuda_forward, "Cross Entropy Loss forward (CUDA)");
    m.def("softmax_forward", &softmax_cuda_forward, "Softmax forward (CUDA)");
}

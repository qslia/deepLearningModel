#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CUDA kernel for global average pooling (squeeze operation)
__global__ void global_avg_pool_forward_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels;
    
    if (idx < total_elements) {
        int batch_idx = idx / channels;
        int channel_idx = idx % channels;
        
        float sum = 0.0f;
        int spatial_size = height * width;
        
        // Calculate the starting position for this batch and channel
        int start_pos = batch_idx * channels * spatial_size + channel_idx * spatial_size;
        
        // Sum all spatial locations for this channel
        for (int i = 0; i < spatial_size; i++) {
            sum += input[start_pos + i];
        }
        
        // Store the average
        output[idx] = sum / spatial_size;
    }
}

// CUDA kernel for global average pooling backward pass
__global__ void global_avg_pool_backward_kernel(
    const float* grad_output,
    float* grad_input,
    int batch_size,
    int channels,
    int height,
    int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int spatial_size = height * width;
        int channel_idx = (idx / spatial_size) % channels;
        int batch_idx = idx / (channels * spatial_size);
        
        int grad_output_idx = batch_idx * channels + channel_idx;
        grad_input[idx] = grad_output[grad_output_idx] / spatial_size;
    }
}

// CUDA kernel for element-wise multiplication (excitation operation)
__global__ void excitation_forward_kernel(
    const float* input,
    const float* weights,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int spatial_size = height * width;
        int channel_idx = (idx / spatial_size) % channels;
        int batch_idx = idx / (channels * spatial_size);
        
        int weight_idx = batch_idx * channels + channel_idx;
        output[idx] = input[idx] * weights[weight_idx];
    }
}

// CUDA kernel for excitation backward pass
__global__ void excitation_backward_kernel(
    const float* grad_output,
    const float* input,
    const float* weights,
    float* grad_input,
    float* grad_weights,
    int batch_size,
    int channels,
    int height,
    int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int spatial_size = height * width;
        int channel_idx = (idx / spatial_size) % channels;
        int batch_idx = idx / (channels * spatial_size);
        
        int weight_idx = batch_idx * channels + channel_idx;
        
        // Gradient w.r.t input
        grad_input[idx] = grad_output[idx] * weights[weight_idx];
        
        // Gradient w.r.t weights (accumulate across spatial dimensions)
        atomicAdd(&grad_weights[weight_idx], grad_output[idx] * input[idx]);
    }
}

// CUDA kernel for Swish activation forward pass
__global__ void swish_forward_kernel(
    const float* input,
    float* output,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));  // x * sigmoid(x)
    }
}

// CUDA kernel for Swish activation backward pass
__global__ void swish_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        float x = input[idx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float swish_derivative = sigmoid_x + x * sigmoid_x * (1.0f - sigmoid_x);
        grad_input[idx] = grad_output[idx] * swish_derivative;
    }
}

// CUDA kernel for sigmoid activation forward pass
__global__ void sigmoid_forward_kernel(
    const float* input,
    float* output,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// CUDA kernel for sigmoid activation backward pass
__global__ void sigmoid_backward_kernel(
    const float* grad_output,
    const float* output,
    float* grad_input,
    int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        float sigmoid_val = output[idx];
        grad_input[idx] = grad_output[idx] * sigmoid_val * (1.0f - sigmoid_val);
    }
}

// Host functions
torch::Tensor global_avg_pool_cuda_forward(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::zeros({batch_size, channels}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * channels + threads - 1) / threads;
    
    global_avg_pool_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width);
    
    return output;
}

torch::Tensor global_avg_pool_cuda_backward(torch::Tensor grad_output, std::vector<int64_t> input_shape) {
    auto batch_size = input_shape[0];
    auto channels = input_shape[1];
    auto height = input_shape[2];
    auto width = input_shape[3];
    
    auto grad_input = torch::zeros({batch_size, channels, height, width}, grad_output.options());
    
    const int threads = 256;
    const int blocks = (batch_size * channels * height * width + threads - 1) / threads;
    
    global_avg_pool_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        batch_size, channels, height, width);
    
    return grad_input;
}

torch::Tensor excitation_cuda_forward(torch::Tensor input, torch::Tensor weights) {
    auto output = torch::zeros_like(input);
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    excitation_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width);
    
    return output;
}

std::vector<torch::Tensor> excitation_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights) {
    
    auto grad_input = torch::zeros_like(input);
    auto grad_weights = torch::zeros_like(weights);
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    excitation_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        batch_size, channels, height, width);
    
    return {grad_input, grad_weights};
}

torch::Tensor swish_cuda_forward(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    swish_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel());
    
    return output;
}

torch::Tensor swish_cuda_backward(torch::Tensor grad_output, torch::Tensor input) {
    auto grad_input = torch::zeros_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    swish_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        input.numel());
    
    return grad_input;
}

torch::Tensor sigmoid_cuda_forward(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    sigmoid_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel());
    
    return output;
}

torch::Tensor sigmoid_cuda_backward(torch::Tensor grad_output, torch::Tensor output) {
    auto grad_input = torch::zeros_like(output);
    
    const int threads = 256;
    const int blocks = (output.numel() + threads - 1) / threads;
    
    sigmoid_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        output.numel());
    
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("global_avg_pool_forward", &global_avg_pool_cuda_forward, "Global Average Pool forward (CUDA)");
    m.def("global_avg_pool_backward", &global_avg_pool_cuda_backward, "Global Average Pool backward (CUDA)");
    m.def("excitation_forward", &excitation_cuda_forward, "Excitation forward (CUDA)");
    m.def("excitation_backward", &excitation_cuda_backward, "Excitation backward (CUDA)");
    m.def("swish_forward", &swish_cuda_forward, "Swish forward (CUDA)");
    m.def("swish_backward", &swish_cuda_backward, "Swish backward (CUDA)");
    m.def("sigmoid_forward", &sigmoid_cuda_forward, "Sigmoid forward (CUDA)");
    m.def("sigmoid_backward", &sigmoid_cuda_backward, "Sigmoid backward (CUDA)");
}

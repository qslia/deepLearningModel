#!/usr/bin/env python3
"""
Test script to compare CUDA SqueezeExcitation implementation with PyTorch version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from squeeze_excitation_cuda import (
    SqueezeExcitationCUDA,
    SwishCUDA,
    SigmoidCUDA,
    efficientnet_b4_cuda,
    CUDA_AVAILABLE,
)


# Original PyTorch implementations for comparison
class Swish(nn.Module):
    """Original Swish activation function: x * sigmoid(x)"""

    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """Original Squeeze-and-Excitation module"""

    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


def test_activation_functions():
    """Test CUDA activation functions against PyTorch versions"""
    print("=" * 60)
    print("Testing Activation Functions")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test data
    x = torch.randn(4, 64, 32, 32, device=device, requires_grad=True)

    # Test Swish
    print("\n1. Testing Swish Activation:")
    swish_pytorch = Swish().to(device)
    swish_cuda = SwishCUDA().to(device)

    # Forward pass
    with torch.no_grad():
        out_pytorch = swish_pytorch(x)
        out_cuda = swish_cuda(x)

    # Compare outputs
    diff = torch.abs(out_pytorch - out_cuda).max().item()
    print(f"   Max difference: {diff:.2e}")
    print(f"   Outputs match: {diff < 1e-5}")

    # Test Sigmoid
    print("\n2. Testing Sigmoid Activation:")
    sigmoid_pytorch = nn.Sigmoid().to(device)
    sigmoid_cuda = SigmoidCUDA().to(device)

    with torch.no_grad():
        out_pytorch = sigmoid_pytorch(x)
        out_cuda = sigmoid_cuda(x)

    diff = torch.abs(out_pytorch - out_cuda).max().item()
    print(f"   Max difference: {diff:.2e}")
    print(f"   Outputs match: {diff < 1e-5}")


def test_squeeze_excitation():
    """Test SqueezeExcitation module"""
    print("\n" + "=" * 60)
    print("Testing SqueezeExcitation Module")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    in_channels = 64
    reduced_dim = 16
    batch_size = 4
    height, width = 32, 32

    # Create modules
    se_pytorch = SqueezeExcitation(in_channels, reduced_dim).to(device)
    se_cuda = SqueezeExcitationCUDA(in_channels, reduced_dim, use_cuda_kernels=True).to(
        device
    )

    # Copy weights from PyTorch to CUDA version for fair comparison
    with torch.no_grad():
        # Copy conv1 weights
        se_cuda.conv1.weight.copy_(se_pytorch.se[1].weight)
        se_cuda.conv1.bias.copy_(se_pytorch.se[1].bias)

        # Copy conv2 weights
        se_cuda.conv2.weight.copy_(se_pytorch.se[3].weight)
        se_cuda.conv2.bias.copy_(se_pytorch.se[3].bias)

    # Test data
    x = torch.randn(batch_size, in_channels, height, width, device=device)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        out_pytorch = se_pytorch(x)
        out_cuda = se_cuda(x)

    print(f"PyTorch output shape: {out_pytorch.shape}")
    print(f"CUDA output shape: {out_cuda.shape}")

    # Compare outputs
    diff = torch.abs(out_pytorch - out_cuda).max().item()
    mean_diff = torch.abs(out_pytorch - out_cuda).mean().item()

    print(f"\nMax difference: {diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    print(f"Outputs match: {diff < 1e-4}")

    return diff < 1e-4


def benchmark_squeeze_excitation():
    """Benchmark SqueezeExcitation performance"""
    print("\n" + "=" * 60)
    print("Benchmarking SqueezeExcitation Performance")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    device = torch.device("cuda")

    # Parameters
    in_channels = 64
    reduced_dim = 16
    batch_size = 32
    height, width = 224, 224
    num_iterations = 100

    # Create modules
    se_pytorch = SqueezeExcitation(in_channels, reduced_dim).to(device)
    se_cuda = SqueezeExcitationCUDA(in_channels, reduced_dim, use_cuda_kernels=True).to(
        device
    )
    se_cuda_fallback = SqueezeExcitationCUDA(
        in_channels, reduced_dim, use_cuda_kernels=False
    ).to(device)

    # Test data
    x = torch.randn(batch_size, in_channels, height, width, device=device)

    print(f"Input shape: {x.shape}")
    print(f"Number of iterations: {num_iterations}")

    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = se_pytorch(x)
            _ = se_cuda(x)
            _ = se_cuda_fallback(x)

    torch.cuda.synchronize()

    # Benchmark PyTorch version
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = se_pytorch(x)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time

    # Benchmark CUDA version
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = se_cuda(x)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time

    # Benchmark CUDA fallback version
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = se_cuda_fallback(x)
    torch.cuda.synchronize()
    fallback_time = time.time() - start_time

    print(f"\nResults:")
    print(
        f"PyTorch SE:      {pytorch_time:.4f}s ({pytorch_time/num_iterations*1000:.2f}ms per iteration)"
    )
    print(
        f"CUDA SE:         {cuda_time:.4f}s ({cuda_time/num_iterations*1000:.2f}ms per iteration)"
    )
    print(
        f"CUDA Fallback:   {fallback_time:.4f}s ({fallback_time/num_iterations*1000:.2f}ms per iteration)"
    )

    if cuda_time > 0:
        speedup = pytorch_time / cuda_time
        print(f"\nSpeedup: {speedup:.2f}x")

    return pytorch_time, cuda_time, fallback_time


def test_gradient_computation():
    """Test gradient computation for CUDA implementation"""
    print("\n" + "=" * 60)
    print("Testing Gradient Computation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    in_channels = 32
    reduced_dim = 8
    batch_size = 2
    height, width = 16, 16

    # Create modules
    se_pytorch = SqueezeExcitation(in_channels, reduced_dim).to(device)
    se_cuda = SqueezeExcitationCUDA(in_channels, reduced_dim, use_cuda_kernels=True).to(
        device
    )

    # Copy weights for fair comparison
    with torch.no_grad():
        se_cuda.conv1.weight.copy_(se_pytorch.se[1].weight)
        se_cuda.conv1.bias.copy_(se_pytorch.se[1].bias)
        se_cuda.conv2.weight.copy_(se_pytorch.se[3].weight)
        se_cuda.conv2.bias.copy_(se_pytorch.se[3].bias)

    # Test data
    x_pytorch = torch.randn(
        batch_size, in_channels, height, width, device=device, requires_grad=True
    )
    x_cuda = x_pytorch.clone().detach().requires_grad_(True)

    # Forward pass
    out_pytorch = se_pytorch(x_pytorch)
    out_cuda = se_cuda(x_cuda)

    # Create dummy loss
    loss_pytorch = out_pytorch.sum()
    loss_cuda = out_cuda.sum()

    # Backward pass
    loss_pytorch.backward()
    loss_cuda.backward()

    # Compare gradients
    grad_diff = torch.abs(x_pytorch.grad - x_cuda.grad).max().item()
    grad_mean_diff = torch.abs(x_pytorch.grad - x_cuda.grad).mean().item()

    print(f"Max gradient difference: {grad_diff:.2e}")
    print(f"Mean gradient difference: {grad_mean_diff:.2e}")
    print(f"Gradients match: {grad_diff < 1e-4}")

    return grad_diff < 1e-4


def test_full_model():
    """Test full EfficientNet model with CUDA SE"""
    print("\n" + "=" * 60)
    print("Testing Full EfficientNet Model")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create models
    model_cuda = efficientnet_b4_cuda(num_classes=5, use_cuda_kernels=True).to(device)
    model_fallback = efficientnet_b4_cuda(num_classes=5, use_cuda_kernels=False).to(
        device
    )

    print(f"CUDA model parameters: {sum(p.numel() for p in model_cuda.parameters()):,}")
    print(
        f"Fallback model parameters: {sum(p.numel() for p in model_fallback.parameters()):,}"
    )

    # Test data
    x = torch.randn(2, 3, 224, 224, device=device)

    # Forward pass
    with torch.no_grad():
        out_cuda = model_cuda(x)
        out_fallback = model_fallback(x)

    print(f"\nInput shape: {x.shape}")
    print(f"CUDA output shape: {out_cuda.shape}")
    print(f"Fallback output shape: {out_fallback.shape}")

    print(
        f"CUDA output range: [{out_cuda.min().item():.4f}, {out_cuda.max().item():.4f}]"
    )
    print(
        f"Fallback output range: [{out_fallback.min().item():.4f}, {out_fallback.max().item():.4f}]"
    )


def main():
    """Main test function"""
    print("CUDA SqueezeExcitation Test Suite")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA extension available: {CUDA_AVAILABLE}")

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(
            f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Run tests
    try:
        test_activation_functions()

        se_test_passed = test_squeeze_excitation()

        if se_test_passed:
            print("\n✅ SqueezeExcitation test PASSED")
        else:
            print("\n❌ SqueezeExcitation test FAILED")

        grad_test_passed = test_gradient_computation()

        if grad_test_passed:
            print("✅ Gradient computation test PASSED")
        else:
            print("❌ Gradient computation test FAILED")

        if torch.cuda.is_available() and CUDA_AVAILABLE:
            benchmark_squeeze_excitation()

        test_full_model()

        print("\n" + "=" * 60)
        print("Test Summary:")
        print(f"SqueezeExcitation: {'✅ PASSED' if se_test_passed else '❌ FAILED'}")
        print(
            f"Gradient computation: {'✅ PASSED' if grad_test_passed else '❌ FAILED'}"
        )
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

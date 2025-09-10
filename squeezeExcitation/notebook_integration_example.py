"""
Example showing how to integrate CUDA SqueezeExcitation into your existing Cassava notebook
"""

# Add this cell to your notebook to use CUDA SqueezeExcitation

# Option 1: Direct replacement (minimal changes)
# Just replace your imports at the top of your notebook:

"""
# Replace this:
class SqueezeExcitation(nn.Module):
    # ... your original implementation

# With this:
"""

try:
    from squeeze_excitation_cuda import SqueezeExcitationCUDA as SqueezeExcitation

    print("✅ Using CUDA-accelerated SqueezeExcitation")
except ImportError:
    # Fallback to your original implementation
    class SqueezeExcitation(nn.Module):
        """Original Squeeze-and-Excitation module"""

        def __init__(self, in_channels, reduced_dim):
            super(SqueezeExcitation, self).__init__()
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, reduced_dim, 1),
                nn.SiLU(),  # Using SiLU instead of Swish for compatibility
                nn.Conv2d(reduced_dim, in_channels, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return x * self.se(x)

    print("⚠️  Using PyTorch SqueezeExcitation (CUDA version not available)")

# Option 2: Enhanced model creation (recommended)
# Replace your create_model function:

"""
# Replace this:
def create_model(model_name, pretrained=False, num_classes=1000):
    if 'efficientnet_b4' in model_name.lower():
        return efficientnet_b4(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported")

# With this:
"""


def create_model(model_name, pretrained=False, num_classes=1000):
    """Enhanced model creation with CUDA support"""
    try:
        from squeeze_excitation_cuda import create_model_cuda

        print("✅ Creating CUDA-accelerated model")
        return create_model_cuda(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            use_cuda_kernels=True,
        )
    except ImportError:
        # Fallback to your original implementation
        print("⚠️  Creating standard PyTorch model")
        if "efficientnet_b4" in model_name.lower():
            return efficientnet_b4(num_classes=num_classes, pretrained=pretrained)
        else:
            raise ValueError(f"Model {model_name} not supported")


# Option 3: Gradual integration (safest)
# Add this function to your notebook and use it instead of your original create_model:


def create_model_with_cuda_option(
    model_name, pretrained=False, num_classes=1000, use_cuda_se=True
):
    """
    Create model with optional CUDA SqueezeExcitation

    Args:
        model_name: Model name (e.g., 'tf_efficientnet_b4_ns')
        pretrained: Whether to load pretrained weights
        num_classes: Number of output classes
        use_cuda_se: Whether to use CUDA SqueezeExcitation (if available)
    """
    if use_cuda_se:
        try:
            from integrate_cuda_se import create_model_enhanced

            print("✅ Using CUDA-enhanced model")
            return create_model_enhanced(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                use_cuda=True,
            )
        except ImportError:
            print("⚠️  CUDA enhancement not available, using standard model")

    # Fallback to original implementation
    if "efficientnet_b4" in model_name.lower():
        return efficientnet_b4(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported")


# Example usage in your training code:
"""
# In your main training section, replace:
model = create_model("tf_efficientnet_b4_ns", pretrained=False, num_classes=5)

# With:
model = create_model_with_cuda_option("tf_efficientnet_b4_ns", 
                                     pretrained=False, 
                                     num_classes=5, 
                                     use_cuda_se=True)
"""


# Performance comparison function (optional)
def compare_models():
    """Compare performance between original and CUDA models"""
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create both models
    model_original = create_model_with_cuda_option(
        "tf_efficientnet_b4_ns", num_classes=5, use_cuda_se=False
    )
    model_cuda = create_model_with_cuda_option(
        "tf_efficientnet_b4_ns", num_classes=5, use_cuda_se=True
    )

    model_original = model_original.to(device)
    model_cuda = model_cuda.to(device)

    # Test data
    x = torch.randn(8, 3, 224, 224, device=device)

    # Warm up
    for _ in range(5):
        with torch.no_grad():
            _ = model_original(x)
            _ = model_cuda(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    num_iterations = 20

    # Original model
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model_original(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    original_time = time.time() - start_time

    # CUDA model
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model_cuda(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    cuda_time = time.time() - start_time

    print(f"\nPerformance Comparison ({num_iterations} iterations):")
    print(
        f"Original model: {original_time:.4f}s ({original_time/num_iterations*1000:.2f}ms per iteration)"
    )
    print(
        f"CUDA model:     {cuda_time:.4f}s ({cuda_time/num_iterations*1000:.2f}ms per iteration)"
    )

    if cuda_time > 0:
        speedup = original_time / cuda_time
        print(f"Speedup: {speedup:.2f}x")


# Test the integration
if __name__ == "__main__":
    print("Testing CUDA SqueezeExcitation integration...")

    # Test model creation
    model = create_model_with_cuda_option("tf_efficientnet_b4_ns", num_classes=5)
    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.randn(1, 3, 224, 224, device=device)

    with torch.no_grad():
        output = model(x)

    print(f"Forward pass successful: {x.shape} -> {output.shape}")

    # Compare performance if CUDA is available
    if torch.cuda.is_available():
        compare_models()

    print("\n✅ Integration test completed successfully!")
    print("\nTo use in your notebook:")
    print("1. Copy the create_model_with_cuda_option function")
    print("2. Replace your model creation call")
    print("3. Enjoy faster training with CUDA SqueezeExcitation!")

#!/usr/bin/env python3
"""
Integration script to use CUDA SqueezeExcitation with the existing Cassava model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try to import CUDA implementation, fallback to original if not available
try:
    from squeeze_excitation_cuda import (
        SqueezeExcitationCUDA,
        SwishCUDA,
        efficientnet_b4_cuda,
        CUDA_AVAILABLE,
    )

    print("✅ CUDA SqueezeExcitation available")
except ImportError:
    print("⚠️  CUDA SqueezeExcitation not available, using PyTorch fallback")
    CUDA_AVAILABLE = False


# Enhanced versions of your original classes with CUDA support
class SwishEnhanced(nn.Module):
    """Enhanced Swish activation with CUDA support"""

    def __init__(self, use_cuda=True):
        super(SwishEnhanced, self).__init__()
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        if self.use_cuda:
            self.swish = SwishCUDA()

    def forward(self, x):
        if self.use_cuda and x.is_cuda:
            return self.swish(x)
        else:
            return x * torch.sigmoid(x)


class SqueezeExcitationEnhanced(nn.Module):
    """Enhanced Squeeze-and-Excitation with CUDA support"""

    def __init__(self, in_channels, reduced_dim, use_cuda=True):
        super(SqueezeExcitationEnhanced, self).__init__()
        self.use_cuda = use_cuda and CUDA_AVAILABLE

        if self.use_cuda:
            self.se = SqueezeExcitationCUDA(
                in_channels, reduced_dim, use_cuda_kernels=True
            )
        else:
            # Fallback to original implementation
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, reduced_dim, 1),
                SwishEnhanced(use_cuda=False),
                nn.Conv2d(reduced_dim, in_channels, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        if self.use_cuda and x.is_cuda:
            return self.se(x)
        else:
            return x * self.se(x)


class MBConvBlockEnhanced(nn.Module):
    """Enhanced MBConv block with CUDA support"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_ratio,
        se_ratio=0.25,
        drop_rate=0.0,
        use_cuda=True,
    ):
        super(MBConvBlockEnhanced, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.use_residual = stride == 1 and in_channels == out_channels
        self.use_cuda = use_cuda and CUDA_AVAILABLE

        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = None
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                SwishEnhanced(use_cuda=self.use_cuda),
            )

        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size,
                stride,
                padding=kernel_size // 2,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            SwishEnhanced(use_cuda=self.use_cuda),
        )

        # Squeeze-and-Excitation
        self.se = None
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcitationEnhanced(
                expanded_channels, se_channels, use_cuda=self.use_cuda
            )

        # Output projection
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Dropout
        self.dropout = nn.Dropout2d(drop_rate) if drop_rate > 0 else None

    def forward(self, x):
        identity = x

        # Expansion
        if self.expand_conv is not None:
            x = self.expand_conv(x)

        # Depthwise convolution
        x = self.depthwise_conv(x)

        # Squeeze-and-Excitation
        if self.se is not None:
            x = self.se(x)

        # Output projection
        x = self.project_conv(x)

        # Dropout and residual connection
        if self.use_residual:
            if self.dropout is not None:
                x = self.dropout(x)
            x = x + identity

        return x


class EfficientNetEnhanced(nn.Module):
    """Enhanced EfficientNet with CUDA support"""

    def __init__(
        self,
        num_classes=1000,
        width_mult=1.0,
        depth_mult=1.0,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        use_cuda=True,
    ):
        super(EfficientNetEnhanced, self).__init__()
        self.use_cuda = use_cuda and CUDA_AVAILABLE

        # EfficientNet-B4 base configuration
        base_config = [
            [1, 16, 1, 1, 3],  # Stage 1
            [6, 24, 2, 2, 3],  # Stage 2
            [6, 40, 2, 2, 5],  # Stage 3
            [6, 80, 3, 2, 3],  # Stage 4
            [6, 112, 3, 1, 5],  # Stage 5
            [6, 192, 4, 2, 5],  # Stage 6
            [6, 320, 1, 1, 3],  # Stage 7
        ]

        # Apply width and depth multipliers
        def round_filters(filters, multiplier):
            if not multiplier:
                return filters
            filters *= multiplier
            new_filters = max(8, int(filters + 4) // 8 * 8)
            if new_filters < 0.9 * filters:
                new_filters += 8
            return int(new_filters)

        def round_repeats(repeats, multiplier):
            if not multiplier:
                return repeats
            return int(math.ceil(multiplier * repeats))

        # Stem
        stem_channels = round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            SwishEnhanced(use_cuda=self.use_cuda),
        )

        # Build blocks
        self.blocks = nn.ModuleList()
        in_channels = stem_channels
        total_blocks = sum(
            [round_repeats(config[2], depth_mult) for config in base_config]
        )
        block_idx = 0

        for expand_ratio, channels, repeats, stride, kernel_size in base_config:
            out_channels = round_filters(channels, width_mult)
            repeats = round_repeats(repeats, depth_mult)

            for i in range(repeats):
                # Drop connect rate increases linearly
                drop_rate = drop_connect_rate * block_idx / total_blocks

                self.blocks.append(
                    MBConvBlockEnhanced(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride if i == 0 else 1,
                        expand_ratio=expand_ratio,
                        se_ratio=0.25,
                        drop_rate=drop_rate,
                        use_cuda=self.use_cuda,
                    )
                )
                in_channels = out_channels
                block_idx += 1

        # Head
        head_channels = round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            SwishEnhanced(use_cuda=self.use_cuda),
        )

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(head_channels, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Blocks
        for block in self.blocks:
            x = block(x)

        # Head
        x = self.head(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


def create_model_enhanced(
    model_name, pretrained=False, num_classes=1000, use_cuda=True
):
    """
    Enhanced model creation with CUDA support
    """
    if "efficientnet_b4" in model_name.lower():
        if use_cuda and CUDA_AVAILABLE:
            # Use pure CUDA implementation if available
            return efficientnet_b4_cuda(
                num_classes=num_classes, pretrained=pretrained, use_cuda_kernels=True
            )
        else:
            # Use enhanced version with fallback
            return EfficientNetEnhanced(
                num_classes=num_classes,
                width_mult=1.4,  # EfficientNet-B4 parameters
                depth_mult=1.8,
                dropout_rate=0.4,
                drop_connect_rate=0.2,
                use_cuda=use_cuda,
            )
    else:
        raise ValueError(f"Model {model_name} not supported")


def benchmark_models():
    """Benchmark original vs CUDA-enhanced models"""
    if not torch.cuda.is_available():
        print("CUDA not available for benchmarking")
        return

    device = torch.device("cuda")

    # Create models
    print("Creating models...")
    model_original = EfficientNetEnhanced(num_classes=5, use_cuda=False).to(device)
    model_cuda = EfficientNetEnhanced(num_classes=5, use_cuda=True).to(device)

    # Test data
    x = torch.randn(4, 3, 224, 224, device=device)

    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model_original(x)
            _ = model_cuda(x)

    torch.cuda.synchronize()

    # Benchmark
    import time

    num_iterations = 50

    # Original model
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model_original(x)
    torch.cuda.synchronize()
    original_time = time.time() - start_time

    # CUDA model
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model_cuda(x)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time

    print(f"\nBenchmark Results ({num_iterations} iterations):")
    print(
        f"Original model: {original_time:.4f}s ({original_time/num_iterations*1000:.2f}ms per iteration)"
    )
    print(
        f"CUDA model:     {cuda_time:.4f}s ({cuda_time/num_iterations*1000:.2f}ms per iteration)"
    )

    if cuda_time > 0:
        speedup = original_time / cuda_time
        print(f"Speedup: {speedup:.2f}x")


def main():
    """Main integration example"""
    print("CUDA SqueezeExcitation Integration")
    print("=" * 50)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA SE available: {CUDA_AVAILABLE}")

    # Create enhanced model
    print("\nCreating enhanced model...")
    model = create_model_enhanced("tf_efficientnet_b4_ns", num_classes=5, use_cuda=True)

    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 3, 224, 224, device=device)

    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Benchmark if CUDA is available
    if torch.cuda.is_available() and CUDA_AVAILABLE:
        print("\nRunning benchmark...")
        benchmark_models()

    print("\n" + "=" * 50)
    print("Integration complete!")
    print("You can now use create_model_enhanced() in your training code")
    print("=" * 50)


if __name__ == "__main__":
    main()

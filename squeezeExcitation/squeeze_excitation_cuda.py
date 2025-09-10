import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

# Try to import the CUDA extension
try:
    import squeeze_excitation_cuda_ext

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print(
        "Warning: CUDA extension not available. Falling back to PyTorch implementation."
    )


class GlobalAvgPoolCUDAFunction(Function):
    @staticmethod
    def forward(ctx, input):
        if CUDA_AVAILABLE and input.is_cuda:
            ctx.input_shape = input.shape
            return squeeze_excitation_cuda_ext.global_avg_pool_forward(input)
        else:
            # Fallback to PyTorch
            return F.adaptive_avg_pool2d(input, 1).squeeze(-1).squeeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        if CUDA_AVAILABLE and grad_output.is_cuda:
            return squeeze_excitation_cuda_ext.global_avg_pool_backward(
                grad_output, list(ctx.input_shape)
            )
        else:
            # Fallback to PyTorch
            batch_size, channels, height, width = ctx.input_shape
            grad_input = (
                grad_output.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(batch_size, channels, height, width)
            )
            return grad_input / (height * width)


class ExcitationCUDAFunction(Function):
    @staticmethod
    def forward(ctx, input, weights):
        if CUDA_AVAILABLE and input.is_cuda:
            ctx.save_for_backward(input, weights)
            return squeeze_excitation_cuda_ext.excitation_forward(input, weights)
        else:
            # Fallback to PyTorch
            ctx.save_for_backward(input, weights)
            return input * weights.unsqueeze(-1).unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        if CUDA_AVAILABLE and grad_output.is_cuda:
            input, weights = ctx.saved_tensors
            grad_input, grad_weights = squeeze_excitation_cuda_ext.excitation_backward(
                grad_output, input, weights
            )
            return grad_input, grad_weights
        else:
            # Fallback to PyTorch
            input, weights = ctx.saved_tensors
            weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)
            grad_input = grad_output * weights_expanded
            grad_weights = (grad_output * input).sum(dim=[2, 3])
            return grad_input, grad_weights


class SwishCUDAFunction(Function):
    @staticmethod
    def forward(ctx, input):
        if CUDA_AVAILABLE and input.is_cuda:
            ctx.save_for_backward(input)
            return squeeze_excitation_cuda_ext.swish_forward(input)
        else:
            # Fallback to PyTorch
            ctx.save_for_backward(input)
            return input * torch.sigmoid(input)

    @staticmethod
    def backward(ctx, grad_output):
        if CUDA_AVAILABLE and grad_output.is_cuda:
            (input,) = ctx.saved_tensors
            return squeeze_excitation_cuda_ext.swish_backward(grad_output, input)
        else:
            # Fallback to PyTorch
            (input,) = ctx.saved_tensors
            sigmoid_x = torch.sigmoid(input)
            swish_derivative = sigmoid_x + input * sigmoid_x * (1 - sigmoid_x)
            return grad_output * swish_derivative


class SigmoidCUDAFunction(Function):
    @staticmethod
    def forward(ctx, input):
        if CUDA_AVAILABLE and input.is_cuda:
            output = squeeze_excitation_cuda_ext.sigmoid_forward(input)
            ctx.save_for_backward(output)
            return output
        else:
            # Fallback to PyTorch
            output = torch.sigmoid(input)
            ctx.save_for_backward(output)
            return output

    @staticmethod
    def backward(ctx, grad_output):
        if CUDA_AVAILABLE and grad_output.is_cuda:
            (output,) = ctx.saved_tensors
            return squeeze_excitation_cuda_ext.sigmoid_backward(grad_output, output)
        else:
            # Fallback to PyTorch
            (output,) = ctx.saved_tensors
            return grad_output * output * (1 - output)


class SwishCUDA(nn.Module):
    """CUDA-accelerated Swish activation function: x * sigmoid(x)"""

    def forward(self, x):
        return SwishCUDAFunction.apply(x)


class SigmoidCUDA(nn.Module):
    """CUDA-accelerated Sigmoid activation function"""

    def forward(self, x):
        return SigmoidCUDAFunction.apply(x)


class SqueezeExcitationCUDA(nn.Module):
    """CUDA-accelerated Squeeze-and-Excitation module"""

    def __init__(self, in_channels, reduced_dim, use_cuda_kernels=True):
        super(SqueezeExcitationCUDA, self).__init__()
        self.in_channels = in_channels
        self.reduced_dim = reduced_dim
        self.use_cuda_kernels = use_cuda_kernels and CUDA_AVAILABLE

        # First convolution (squeeze)
        self.conv1 = nn.Conv2d(in_channels, reduced_dim, 1, bias=True)
        # Second convolution (excitation)
        self.conv2 = nn.Conv2d(reduced_dim, in_channels, 1, bias=True)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        if self.use_cuda_kernels and x.is_cuda:
            # Use custom CUDA kernels
            # Global average pooling
            squeezed = GlobalAvgPoolCUDAFunction.apply(x)  # [B, C]

            # First convolution + Swish
            squeezed = squeezed.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            squeezed = self.conv1(squeezed)  # [B, reduced_dim, 1, 1]
            squeezed = SwishCUDAFunction.apply(squeezed)

            # Second convolution + Sigmoid
            excited = self.conv2(squeezed)  # [B, in_channels, 1, 1]
            excited = SigmoidCUDAFunction.apply(excited)

            # Element-wise multiplication
            excited = excited.squeeze(-1).squeeze(-1)  # [B, in_channels]
            output = ExcitationCUDAFunction.apply(x, excited)

        else:
            # Fallback to standard PyTorch implementation
            # Global average pooling
            squeezed = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]

            # First convolution + Swish
            squeezed = self.conv1(squeezed)  # [B, reduced_dim, 1, 1]
            squeezed = squeezed * torch.sigmoid(squeezed)  # Swish activation

            # Second convolution + Sigmoid
            excited = self.conv2(squeezed)  # [B, in_channels, 1, 1]
            excited = torch.sigmoid(excited)

            # Element-wise multiplication
            output = x * excited

        return output


class MBConvBlockCUDA(nn.Module):
    """CUDA-accelerated Mobile Inverted Bottleneck Convolution Block"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_ratio,
        se_ratio=0.25,
        drop_rate=0.0,
        use_cuda_kernels=True,
    ):
        super(MBConvBlockCUDA, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.use_residual = stride == 1 and in_channels == out_channels
        self.use_cuda_kernels = use_cuda_kernels

        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = None
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                SwishCUDA() if use_cuda_kernels else nn.SiLU(),
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
            SwishCUDA() if use_cuda_kernels else nn.SiLU(),
        )

        # Squeeze-and-Excitation
        self.se = None
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcitationCUDA(
                expanded_channels, se_channels, use_cuda_kernels
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


class EfficientNetCUDA(nn.Module):
    """CUDA-accelerated EfficientNet implementation"""

    def __init__(
        self,
        num_classes=1000,
        width_mult=1.0,
        depth_mult=1.0,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        use_cuda_kernels=True,
    ):
        super(EfficientNetCUDA, self).__init__()
        self.use_cuda_kernels = use_cuda_kernels

        # EfficientNet-B4 base configuration
        # [expand_ratio, channels, repeats, stride, kernel_size]
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
            SwishCUDA() if use_cuda_kernels else nn.SiLU(),
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
                    MBConvBlockCUDA(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride if i == 0 else 1,
                        expand_ratio=expand_ratio,
                        se_ratio=0.25,
                        drop_rate=drop_rate,
                        use_cuda_kernels=use_cuda_kernels,
                    )
                )
                in_channels = out_channels
                block_idx += 1

        # Head
        head_channels = round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            SwishCUDA() if use_cuda_kernels else nn.SiLU(),
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


def efficientnet_b4_cuda(num_classes=1000, pretrained=False, use_cuda_kernels=True):
    """
    CUDA-accelerated EfficientNet-B4 model

    Args:
        num_classes (int): Number of classification classes
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        use_cuda_kernels (bool): Whether to use custom CUDA kernels

    Returns:
        EfficientNetCUDA model
    """
    # EfficientNet-B4 scaling parameters
    model = EfficientNetCUDA(
        num_classes=num_classes,
        width_mult=1.4,  # phi = 3, width = 1.4
        depth_mult=1.8,  # phi = 3, depth = 1.8
        dropout_rate=0.4,
        drop_connect_rate=0.2,
        use_cuda_kernels=use_cuda_kernels,
    )

    if pretrained:
        # Note: You would need to implement loading pretrained weights here
        print("Warning: Pretrained weights not implemented for CUDA EfficientNet-B4")

    return model


def create_model_cuda(
    model_name, pretrained=False, num_classes=1000, use_cuda_kernels=True
):
    """
    Create CUDA-accelerated model function

    Args:
        model_name (str): Model name (e.g., 'tf_efficientnet_b4_ns')
        pretrained (bool): Whether to load pretrained weights
        num_classes (int): Number of output classes
        use_cuda_kernels (bool): Whether to use custom CUDA kernels

    Returns:
        Model instance
    """
    if "efficientnet_b4" in model_name.lower():
        return efficientnet_b4_cuda(
            num_classes=num_classes,
            pretrained=pretrained,
            use_cuda_kernels=use_cuda_kernels,
        )
    else:
        raise ValueError(f"Model {model_name} not supported")


if __name__ == "__main__":
    # Test the CUDA implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA kernels available: {CUDA_AVAILABLE}")

    # Test SqueezeExcitation module
    print("\nTesting SqueezeExcitation CUDA module...")
    se_cuda = SqueezeExcitationCUDA(64, 16).to(device)
    x = torch.randn(2, 64, 32, 32).to(device)

    # Forward pass
    with torch.no_grad():
        output = se_cuda(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test full model
    print("\nTesting EfficientNet CUDA model...")
    model = efficientnet_b4_cuda(num_classes=5, use_cuda_kernels=True).to(device)
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")

    print("\nCUDA SqueezeExcitation implementation ready!")

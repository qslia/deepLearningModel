import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.cpp_extension import load
from torch.autograd import Function
import os

# PyTorch 2.6.0 compatible CUDA extension loading
import warnings

# Suppress PyTorch 2.6.0 warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.utils.cpp_extension"
)

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"  # RTX 3060

try:
    # Try to import pre-compiled module first
    import fashion_mnist_cnn_cuda as cuda_module

    print("✓ Using pre-compiled CUDA module")
except ImportError:
    # Fallback to JIT compilation with PyTorch 2.6.0 compatibility
    print("⚠ Pre-compiled module not found, using JIT compilation...")
    try:
        cuda_module = load(
            name="fashion_mnist_cnn_cuda",
            sources=["fashion_mnist_cnn_cuda.cu"],
            verbose=False,  # Reduce verbose output
            extra_cflags=[
                "-O2",
                "-std=c++17",
                "-DWITH_CUDA",
                "-DTORCH_API_INCLUDE_EXTENSION_H",
                # CUDA 12.6 + PyTorch 2.6.0 compatibility
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
            extra_cuda_cflags=[
                "-O2",
                "--use_fast_math",
                "-gencode=arch=compute_86,code=sm_86",
                "--extended-lambda",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-std=c++17",
                # Critical CUDA 12.6 + PyTorch 2.6.0 compatibility
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
                "-DTORCH_API_INCLUDE_EXTENSION_H",
                "-DTORCH_EXTENSION_NAME=fashion_mnist_cnn_cuda",
                # Disable problematic CUDA features
                "-DCUDA_HAS_FP16=0",
                "-DCUDA_HAS_BF16=0",
                # Windows MSVC compatibility
                "-Xcompiler",
                "/wd4819,/wd4244,/wd4267,/wd4996,/wd4275,/wd4251,/EHsc",
                # Suppress CUDA compiler warnings
                "--diag-suppress",
                "767",  # pointer conversion
                "--diag-suppress",
                "3326",  # operator new
                "--diag-suppress",
                "3322",  # operator new
                "--diag-suppress",
                "20012",  # host device warnings
                "--diag-suppress",
                "20014",  # host device warnings
            ],
        )
        print("✓ JIT compilation successful")
    except Exception as e:
        print(f"✗ CUDA compilation failed: {e}")
        print("Falling back to CPU-only implementation...")
        cuda_module = None


# Autograd-compatible CUDA function wrappers
class Conv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, kernel_size):
        # Save tensors for backward pass
        ctx.save_for_backward(input, weight, bias)
        ctx.kernel_size = kernel_size

        if cuda_module is not None:
            # Use CUDA implementation
            return cuda_module.conv2d_forward(input, weight, bias, kernel_size)
        else:
            # Fallback to PyTorch
            return F.conv2d(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        # For now, use PyTorch's autograd for backward pass
        # This ensures gradients work while you learn CUDA forward passes
        input, weight, bias = ctx.saved_tensors

        # Use PyTorch's built-in backward computation
        # In a full CUDA implementation, you'd call your CUDA backward kernels here
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None


class ReLUFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        if cuda_module is not None:
            # Use CUDA implementation
            return cuda_module.relu_forward(input)
        else:
            # Fallback to PyTorch
            return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors

        if cuda_module is not None:
            # For learning: you could implement relu_backward in CUDA
            # For now, use PyTorch's computation
            grad_input = grad_output * (input > 0).float()
            return grad_input
        else:
            grad_input = grad_output * (input > 0).float()
            return grad_input


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)

        if cuda_module is not None:
            # Use CUDA implementation
            return cuda_module.linear_forward(input, weight, bias)
        else:
            # Fallback to PyTorch
            return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


# Convenience functions to call the autograd functions
def conv2d_cuda(input, weight, bias, kernel_size):
    return Conv2dFunction.apply(input, weight, bias, kernel_size)


def relu_cuda(input):
    return ReLUFunction.apply(input)


def linear_cuda(input, weight, bias):
    return LinearFunction.apply(input, weight, bias)


class Conv2dCUDA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(Conv2dCUDA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Initialize weights and bias as parameters
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        if cuda_module is None:
            print(
                f"⚠ Using PyTorch fallback for Conv2d({in_channels}, {out_channels}, {kernel_size})"
            )

    def forward(self, x):
        # Use CUDA implementation with autograd support!
        return conv2d_cuda(x, self.weight, self.bias, self.kernel_size)


class MaxPool2dCUDA(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(MaxPool2dCUDA, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

        # Fallback to PyTorch implementation
        self.pytorch_maxpool = nn.MaxPool2d(kernel_size, stride)
        if cuda_module is None:
            print(f"⚠ Using PyTorch fallback for MaxPool2d({kernel_size})")

    def forward(self, x):
        if cuda_module is not None:
            try:
                output, indices = cuda_module.maxpool2d_forward(
                    x, self.kernel_size, self.stride
                )
                return output
            except Exception as e:
                print(f"⚠ CUDA maxpool2d failed: {e}, using PyTorch fallback")
                return self.pytorch_maxpool(x)
        else:
            return self.pytorch_maxpool(x)


class ReLUCUDA(nn.Module):
    def __init__(self):
        super(ReLUCUDA, self).__init__()

        if cuda_module is None:
            print("⚠ Using PyTorch fallback for ReLU")

    def forward(self, x):
        # Use CUDA implementation with autograd support!
        return relu_cuda(x)


class LinearCUDA(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearCUDA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights and bias as parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        if cuda_module is None:
            print(f"⚠ Using PyTorch fallback for Linear({in_features}, {out_features})")

    def forward(self, x):
        # Reshape input if needed (for flattening after conv layers)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Use CUDA implementation with autograd support!
        return linear_cuda(x, self.weight, self.bias)


class FlattenCUDA(nn.Module):
    def __init__(self):
        super(FlattenCUDA, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class CrossEntropyLossCUDA(nn.Module):
    def __init__(self):
        super(CrossEntropyLossCUDA, self).__init__()
        # Fallback to PyTorch implementation to ensure autograd compatibility
        self.pytorch_ce = nn.CrossEntropyLoss()
        if cuda_module is None:
            print("⚠ Using PyTorch fallback for CrossEntropyLoss")

    def forward(self, logits, targets):
        # Convert targets to long if needed (CrossEntropyLoss expects LongTensor)
        if targets.dtype != torch.long:
            targets = targets.to(torch.long)

        if cuda_module is not None:
            try:
                # For now, use PyTorch's CrossEntropyLoss to ensure autograd compatibility
                # CUDA implementation can be added later with proper autograd integration
                return self.pytorch_ce(logits, targets)
            except Exception as e:
                print(f"⚠ CUDA cross_entropy failed: {e}, using PyTorch fallback")
                return self.pytorch_ce(logits, targets)
        else:
            return self.pytorch_ce(logits, targets)


class SoftmaxCUDA(nn.Module):
    def __init__(self, dim=-1):
        super(SoftmaxCUDA, self).__init__()
        self.dim = dim
        # Fallback to PyTorch implementation
        self.pytorch_softmax = nn.Softmax(dim=dim)
        if cuda_module is None:
            print("⚠ Using PyTorch fallback for Softmax")

    def forward(self, x):
        if cuda_module is not None:
            try:
                # For now, use PyTorch's Softmax to ensure autograd compatibility
                return self.pytorch_softmax(x)
            except Exception as e:
                print(f"⚠ CUDA softmax failed: {e}, using PyTorch fallback")
                return self.pytorch_softmax(x)
        else:
            return self.pytorch_softmax(x)


class FashionMNISTCNN(nn.Module):
    """
    CUDA implementation of the Fashion-MNIST CNN model:
    - Conv2d(1, 64, kernel_size=3) -> MaxPool2d(2) -> ReLU
    - Conv2d(64, 128, kernel_size=3) -> MaxPool2d(2) -> ReLU
    - Flatten -> Linear(3200, 256) -> ReLU -> Linear(256, 10)
    """

    def __init__(self, num_classes=10):
        super(FashionMNISTCNN, self).__init__()

        # Feature extraction layers
        self.conv1 = Conv2dCUDA(1, 64, kernel_size=3)
        self.pool1 = MaxPool2dCUDA(kernel_size=2, stride=2)
        self.relu1 = ReLUCUDA()

        self.conv2 = Conv2dCUDA(64, 128, kernel_size=3)
        self.pool2 = MaxPool2dCUDA(kernel_size=2, stride=2)
        self.relu2 = ReLUCUDA()

        # Classifier layers
        self.flatten = FlattenCUDA()
        self.fc1 = LinearCUDA(3200, 256)  # 128 * 5 * 5 = 3200
        self.relu3 = ReLUCUDA()
        self.fc2 = LinearCUDA(256, num_classes)

        self.softmax = SoftmaxCUDA()

    def forward(self, x):
        # First conv block: Conv -> MaxPool -> ReLU
        x = self.conv1(x)  # [N, 1, 28, 28] -> [N, 64, 26, 26]
        x = self.pool1(x)  # [N, 64, 26, 26] -> [N, 64, 13, 13]
        x = self.relu1(x)  # [N, 64, 13, 13]

        # Second conv block: Conv -> MaxPool -> ReLU
        x = self.conv2(x)  # [N, 64, 13, 13] -> [N, 128, 11, 11]
        x = self.pool2(x)  # [N, 128, 11, 11] -> [N, 128, 5, 5]
        x = self.relu2(x)  # [N, 128, 5, 5]

        # Classifier: Flatten -> Linear -> ReLU -> Linear
        x = self.flatten(x)  # [N, 128, 5, 5] -> [N, 3200]
        x = self.fc1(x)  # [N, 3200] -> [N, 256]
        x = self.relu3(x)  # [N, 256]
        x = self.fc2(x)  # [N, 256] -> [N, 10]

        return x

    def predict(self, x):
        """Get softmax probabilities for prediction"""
        logits = self.forward(x)
        return self.softmax(logits)


class FashionMNISTDataset:
    """Simple dataset class for Fashion-MNIST data"""

    def __init__(self, images, labels, device="cuda"):
        # Normalize to [0, 1] and add channel dimension
        self.images = (images.float() / 255.0).unsqueeze(1).to(device)
        self.labels = labels.to(device)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def get_batch(self, batch_size=32, shuffle=True):
        """Get a batch of data"""
        if shuffle:
            indices = torch.randperm(len(self))[:batch_size]
        else:
            indices = torch.arange(min(batch_size, len(self)))

        return self.images[indices], self.labels[indices]


def train_step(model, data_batch, target_batch, loss_fn, optimizer):
    """Single training step"""
    optimizer.zero_grad()

    # Forward pass
    logits = model(data_batch)
    loss = loss_fn(logits, target_batch)

    # Backward pass (using PyTorch's autograd for now)
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_accuracy(model, dataset, batch_size=1000):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            end_idx = min(i + batch_size, len(dataset))
            batch_images = dataset.images[i:end_idx]
            batch_labels = dataset.labels[i:end_idx]

            predictions = model.predict(batch_images)
            predicted_classes = predictions.argmax(dim=1)

            correct += (predicted_classes == batch_labels).sum().item()
            total += len(batch_labels)

    model.train()
    return correct / total


def create_model(device="cuda"):
    """Create and initialize the CUDA CNN model"""
    model = FashionMNISTCNN(num_classes=10).to(device)
    loss_fn = CrossEntropyLossCUDA()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    return model, loss_fn, optimizer


# Example usage and training function
def train_fashion_mnist_cuda(
    train_images,
    train_labels,
    val_images,
    val_labels,
    epochs=5,
    batch_size=32,
    device="cuda",
):
    """
    Complete training function for Fashion-MNIST with CUDA CNN

    Args:
        train_images: Training images tensor [N, 28, 28]
        train_labels: Training labels tensor [N]
        val_images: Validation images tensor [N, 28, 28]
        val_labels: Validation labels tensor [N]
        epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to run on ('cuda' or 'cpu')
    """

    # Create datasets
    train_dataset = FashionMNISTDataset(train_images, train_labels, device)
    val_dataset = FashionMNISTDataset(val_images, val_labels, device)

    # Create model, loss function, and optimizer
    model, loss_fn, optimizer = create_model(device)

    print(f"Training CUDA CNN on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        # Training
        num_batches = len(train_dataset) // batch_size
        for batch_idx in range(num_batches):
            # Get batch
            batch_images, batch_labels = train_dataset.get_batch(
                batch_size, shuffle=True
            )

            # Training step
            loss = train_step(model, batch_images, batch_labels, loss_fn, optimizer)
            epoch_losses.append(loss)

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}"
                )

        # Calculate metrics
        avg_train_loss = np.mean(epoch_losses)
        train_acc = evaluate_accuracy(model, train_dataset)
        val_acc = evaluate_accuracy(model, val_dataset)

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
        )

    return model, {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }


if __name__ == "__main__":
    # Test the model creation
    print("Creating CUDA CNN model...")
    try:
        model, loss_fn, optimizer = create_model()
        print("✓ Model created successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test with dummy data
        dummy_input = torch.randn(4, 1, 28, 28).cuda()
        dummy_target = torch.randint(0, 10, (4,)).cuda()

        print("Testing forward pass...")
        output = model(dummy_input)
        print(f"✓ Forward pass successful. Output shape: {output.shape}")

        print("Testing loss computation...")
        loss = loss_fn(output, dummy_target)
        print(f"✓ Loss computation successful. Loss: {loss.item():.4f}")

        print("Testing backward pass...")
        loss.backward()
        print("✓ Backward pass successful")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()

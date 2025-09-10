#!/usr/bin/env python3
"""
Test script for Fashion-MNIST CUDA CNN implementation
"""

import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from fashion_mnist_cnn_model import (
    FashionMNISTCNN,
    FashionMNISTDataset,
    train_fashion_mnist_cuda,
    create_model,
)


def download_fashion_mnist(data_folder="./data"):
    """Download Fashion-MNIST dataset"""
    print("Downloading Fashion-MNIST dataset...")

    # Download training data
    train_dataset = datasets.FashionMNIST(
        root=data_folder, train=True, download=True, transform=transforms.ToTensor()
    )

    # Download test data
    test_dataset = datasets.FashionMNIST(
        root=data_folder, train=False, download=True, transform=transforms.ToTensor()
    )

    # Extract tensors
    train_images = train_dataset.data
    train_labels = train_dataset.targets
    test_images = test_dataset.data
    test_labels = test_dataset.targets

    print(f"Training data: {train_images.shape}, {train_labels.shape}")
    print(f"Test data: {test_images.shape}, {test_labels.shape}")

    return train_images, train_labels, test_images, test_labels


def test_model_components():
    """Test individual model components"""
    print("\n" + "=" * 50)
    print("TESTING MODEL COMPONENTS")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Test model creation
        print("\n1. Testing model creation...")
        model, loss_fn, optimizer = create_model(device)
        print("✓ Model created successfully")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        # Test forward pass with dummy data
        print("\n2. Testing forward pass...")
        batch_size = 8
        dummy_input = torch.randn(batch_size, 1, 28, 28).to(device)
        dummy_target = torch.randint(0, 10, (batch_size,)).to(device)

        start_time = time.time()
        output = model(dummy_input)
        forward_time = time.time() - start_time

        print(f"✓ Forward pass successful")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Forward pass time: {forward_time*1000:.2f} ms")

        # Test loss computation
        print("\n3. Testing loss computation...")
        loss = loss_fn(output, dummy_target)
        print(f"✓ Loss computation successful")
        print(f"   Loss value: {loss.item():.4f}")

        # Test backward pass
        print("\n4. Testing backward pass...")
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        print(f"✓ Backward pass successful")
        print(f"   Backward pass time: {backward_time*1000:.2f} ms")

        # Test prediction
        print("\n5. Testing prediction...")
        model.eval()
        with torch.no_grad():
            predictions = model.predict(dummy_input)
            predicted_classes = predictions.argmax(dim=1)

        print(f"✓ Prediction successful")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Predicted classes: {predicted_classes.cpu().numpy()}")
        print(
            f"   Prediction probabilities sum: {predictions.sum(dim=1).cpu().numpy()}"
        )

        return True

    except Exception as e:
        print(f"✗ Error in component testing: {e}")
        import traceback

        traceback.print_exc()
        return False


def benchmark_performance(model, device="cuda", num_runs=100):
    """Benchmark model performance"""
    print("\n" + "=" * 50)
    print("PERFORMANCE BENCHMARK")
    print("=" * 50)

    model.eval()
    batch_sizes = [1, 8, 16, 32, 64]

    print(f"Running {num_runs} iterations for each batch size...")
    print(f"{'Batch Size':<12} {'Forward (ms)':<15} {'Throughput (img/s)':<20}")
    print("-" * 50)

    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, 1, 28, 28).to(device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)

        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)

        torch.cuda.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs
        avg_time_ms = avg_time * 1000
        throughput = batch_size / avg_time

        print(f"{batch_size:<12} {avg_time_ms:<15.2f} {throughput:<20.1f}")


def train_small_sample():
    """Train on a small sample to verify training works"""
    print("\n" + "=" * 50)
    print("TRAINING VERIFICATION")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create small dummy dataset
    print("Creating small dummy dataset...")
    train_images = torch.randint(0, 256, (1000, 28, 28)).float()
    train_labels = torch.randint(0, 10, (1000,))
    val_images = torch.randint(0, 256, (200, 28, 28)).float()
    val_labels = torch.randint(0, 10, (200,))

    print("Starting training...")
    try:
        model, history = train_fashion_mnist_cuda(
            train_images,
            train_labels,
            val_images,
            val_labels,
            epochs=2,
            batch_size=32,
            device=device,
        )

        print("✓ Training completed successfully")
        print(f"   Final train accuracy: {history['train_accuracies'][-1]:.4f}")
        print(f"   Final val accuracy: {history['val_accuracies'][-1]:.4f}")

        return True

    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def compare_with_pytorch():
    """Compare CUDA implementation with PyTorch implementation"""
    print("\n" + "=" * 50)
    print("PYTORCH COMPARISON")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create PyTorch reference model
    pytorch_model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=3),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, kernel_size=3),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(3200, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    ).to(device)

    # Create CUDA model
    cuda_model, _, _ = create_model(device)

    # Test input
    test_input = torch.randn(4, 1, 28, 28).to(device)

    print("Comparing output shapes...")
    pytorch_output = pytorch_model(test_input)
    cuda_output = cuda_model(test_input)

    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"CUDA output shape: {cuda_output.shape}")

    if pytorch_output.shape == cuda_output.shape:
        print("✓ Output shapes match")
    else:
        print("✗ Output shapes don't match")

    # Compare parameter counts
    pytorch_params = sum(p.numel() for p in pytorch_model.parameters())
    cuda_params = sum(p.numel() for p in cuda_model.parameters())

    print(f"PyTorch parameters: {pytorch_params:,}")
    print(f"CUDA parameters: {cuda_params:,}")

    if pytorch_params == cuda_params:
        print("✓ Parameter counts match")
    else:
        print("✗ Parameter counts don't match")


def visualize_sample_predictions(model, test_images, test_labels, num_samples=8):
    """Visualize sample predictions"""
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)

    # Fashion-MNIST class names
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    device = next(model.parameters()).device
    model.eval()

    # Get random samples
    indices = torch.randperm(len(test_images))[:num_samples]
    sample_images = test_images[indices]
    sample_labels = test_labels[indices]

    # Normalize and add channel dimension
    sample_images_norm = (sample_images.float() / 255.0).unsqueeze(1).to(device)

    # Get predictions
    with torch.no_grad():
        predictions = model.predict(sample_images_norm)
        predicted_classes = predictions.argmax(dim=1)

    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(num_samples):
        axes[i].imshow(sample_images[i].cpu().numpy(), cmap="gray")
        true_class = class_names[sample_labels[i]]
        pred_class = class_names[predicted_classes[i]]
        confidence = predictions[i].max().item()

        color = "green" if sample_labels[i] == predicted_classes[i] else "red"
        axes[i].set_title(
            f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}",
            color=color,
            fontsize=8,
        )
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("fashion_mnist_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Predictions visualized and saved to 'fashion_mnist_predictions.png'")


def main():
    """Main testing function"""
    print("Fashion-MNIST CUDA CNN Test Suite")
    print("=" * 50)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Running on CPU.")
    else:
        print(f"✓ CUDA available. GPU: {torch.cuda.get_device_name()}")

    # Test model components
    if not test_model_components():
        print("❌ Component testing failed. Exiting.")
        return

    # Run performance benchmark
    try:
        model, _, _ = create_model()
        benchmark_performance(model)
    except Exception as e:
        print(f"⚠️  Benchmark failed: {e}")

    # Test training
    if not train_small_sample():
        print("⚠️  Training verification failed.")

    # Compare with PyTorch
    try:
        compare_with_pytorch()
    except Exception as e:
        print(f"⚠️  PyTorch comparison failed: {e}")

    # Optional: Download real data and test
    try_real_data = input("\nDownload Fashion-MNIST and test with real data? (y/n): ")
    if try_real_data.lower() == "y":
        try:
            print("Downloading Fashion-MNIST...")
            train_images, train_labels, test_images, test_labels = (
                download_fashion_mnist()
            )

            print("Testing with real data...")
            model, _, _ = create_model()

            # Test on a small subset
            subset_size = 1000
            train_subset = train_images[:subset_size]
            train_labels_subset = train_labels[:subset_size]
            test_subset = test_images[:200]
            test_labels_subset = test_labels[:200]

            model, history = train_fashion_mnist_cuda(
                train_subset,
                train_labels_subset,
                test_subset,
                test_labels_subset,
                epochs=3,
                batch_size=32,
            )

            # Visualize predictions
            visualize_sample_predictions(model, test_images, test_labels)

            print("✓ Real data testing completed successfully")

        except Exception as e:
            print(f"⚠️  Real data testing failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 50)
    print("TEST SUITE COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()

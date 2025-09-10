#!/usr/bin/env python3
"""
Build script for CUDA SqueezeExcitation extension
"""

import os
import sys
import subprocess
import torch
from torch.utils.cpp_extension import load


def check_cuda_availability():
    """Check if CUDA is available and properly configured"""
    print("Checking CUDA availability...")

    if not torch.cuda.is_available():
        print("❌ CUDA is not available in PyTorch")
        return False

    print(f"✅ CUDA is available")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU device: {torch.cuda.get_device_name()}")

    return True


def build_extension():
    """Build the CUDA extension"""
    print("\nBuilding CUDA extension...")

    try:
        # Load and compile the extension
        squeeze_excitation_cuda = load(
            name="squeeze_excitation_cuda_ext",
            sources=["squeeze_excitation_cuda.cu"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode=arch=compute_60,code=sm_60",
                "-gencode=arch=compute_61,code=sm_61",
                "-gencode=arch=compute_70,code=sm_70",
                "-gencode=arch=compute_75,code=sm_75",
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_86,code=sm_86",
            ],
            verbose=True,
        )

        print("✅ CUDA extension built successfully!")
        return squeeze_excitation_cuda

    except Exception as e:
        print(f"❌ Failed to build CUDA extension: {e}")
        return None


def test_extension(extension):
    """Test the built extension"""
    print("\nTesting CUDA extension...")

    try:
        # Test basic functionality
        device = torch.device("cuda")
        x = torch.randn(2, 64, 32, 32, device=device)

        # Test global average pooling
        pooled = extension.global_avg_pool_forward(x)
        print(f"✅ Global average pooling: {x.shape} -> {pooled.shape}")

        # Test excitation
        weights = torch.randn(2, 64, device=device)
        excited = extension.excitation_forward(x, weights)
        print(f"✅ Excitation: {x.shape} -> {excited.shape}")

        # Test activations
        swish_out = extension.swish_forward(x)
        sigmoid_out = extension.sigmoid_forward(x)
        print(f"✅ Swish activation: {x.shape} -> {swish_out.shape}")
        print(f"✅ Sigmoid activation: {x.shape} -> {sigmoid_out.shape}")

        print("✅ All extension tests passed!")
        return True

    except Exception as e:
        print(f"❌ Extension test failed: {e}")
        return False


def install_with_pip():
    """Install using pip and setup.py"""
    print("\nInstalling with pip...")

    try:
        # Run pip install in development mode
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--verbose"],
            check=True,
            capture_output=True,
            text=True,
        )

        print("✅ Package installed successfully!")
        print(result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def main():
    """Main build function"""
    print("CUDA SqueezeExcitation Extension Builder")
    print("=" * 50)

    # Check CUDA availability
    if not check_cuda_availability():
        print("\nPlease ensure CUDA is properly installed and configured.")
        return 1

    # Check if source files exist
    if not os.path.exists("squeeze_excitation_cuda.cu"):
        print("❌ squeeze_excitation_cuda.cu not found!")
        return 1

    if not os.path.exists("setup.py"):
        print("❌ setup.py not found!")
        return 1

    print("✅ Source files found")

    # Choose build method
    print("\nBuild options:")
    print("1. Just-in-time compilation (JIT) - faster for development")
    print("2. Install with pip - better for production")

    choice = input("Choose build method (1 or 2): ").strip()

    if choice == "1":
        # JIT compilation
        extension = build_extension()
        if extension:
            test_extension(extension)
            print("\n✅ JIT compilation successful!")
            print("You can now import squeeze_excitation_cuda_ext in Python")
        else:
            return 1

    elif choice == "2":
        # Pip installation
        if install_with_pip():
            print("\n✅ Installation successful!")
            print("You can now import squeeze_excitation_cuda")
        else:
            return 1
    else:
        print("Invalid choice. Please run again and choose 1 or 2.")
        return 1

    print("\n" + "=" * 50)
    print("Build completed successfully!")
    print("Next steps:")
    print("1. Run: python test_cuda_se.py")
    print("2. Use squeeze_excitation_cuda.py in your projects")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())

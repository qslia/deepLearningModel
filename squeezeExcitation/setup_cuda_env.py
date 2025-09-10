#!/usr/bin/env python3
"""
Setup script to configure CUDA compilation environment on Windows
"""

import os
import sys
import subprocess
import torch


def find_visual_studio():
    """Find Visual Studio installation"""
    possible_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def find_msvc_compiler(vs_path):
    """Find MSVC compiler in Visual Studio installation"""
    if not vs_path:
        return None

    vc_path = os.path.join(vs_path, "VC", "Tools", "MSVC")
    if not os.path.exists(vc_path):
        return None

    # Find the latest MSVC version
    versions = [
        d for d in os.listdir(vc_path) if os.path.isdir(os.path.join(vc_path, d))
    ]
    if not versions:
        return None

    latest_version = sorted(versions)[-1]
    compiler_path = os.path.join(
        vc_path, latest_version, "bin", "Hostx64", "x64", "cl.exe"
    )

    if os.path.exists(compiler_path):
        return os.path.dirname(compiler_path)
    return None


def find_windows_sdk():
    """Find Windows SDK installation"""
    possible_paths = [
        r"C:\Program Files (x86)\Windows Kits\10",
        r"C:\Program Files\Windows Kits\10",
    ]

    for base_path in possible_paths:
        if os.path.exists(base_path):
            lib_path = os.path.join(base_path, "Lib")
            if os.path.exists(lib_path):
                # Find the latest SDK version
                versions = [
                    d
                    for d in os.listdir(lib_path)
                    if (
                        os.path.isdir(os.path.join(lib_path, d)) and d.startswith("10.")
                    )
                ]
                if versions:
                    latest_version = sorted(versions)[-1]
                    sdk_lib_path = os.path.join(lib_path, latest_version)
                    return sdk_lib_path
    return None


def setup_environment():
    """Setup environment variables for CUDA compilation"""
    print("Setting up CUDA compilation environment...")

    # Find Visual Studio
    vs_path = find_visual_studio()
    if not vs_path:
        print("❌ Visual Studio not found!")
        msg = "Please install Visual Studio 2019 or 2022 with C++ build tools."
        print(msg)
        return False

    print(f"✅ Found Visual Studio: {vs_path}")

    # Find MSVC compiler
    compiler_dir = find_msvc_compiler(vs_path)
    if not compiler_dir:
        print("❌ MSVC compiler not found!")
        print("Please install C++ build tools in Visual Studio.")
        return False

    print(f"✅ Found MSVC compiler: {compiler_dir}")

    # Find Windows SDK
    sdk_path = find_windows_sdk()
    if sdk_path:
        print(f"✅ Found Windows SDK: {sdk_path}")
    else:
        print("⚠️  Windows SDK not found, may cause linking issues")

    # Set environment variables
    current_path = os.environ.get("PATH", "")
    if compiler_dir not in current_path:
        os.environ["PATH"] = f"{compiler_dir};{current_path}"
        print("✅ Added compiler to PATH")

    # Set CUDA architecture for RTX 3060
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    print("✅ Set CUDA architecture for RTX 3060")

    # Set additional environment variables for proper linking
    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MSSdk"] = "1"

    # Set library paths if SDK found
    if sdk_path:
        ucrt_path = os.path.join(sdk_path, "ucrt", "x64")
        um_path = os.path.join(sdk_path, "um", "x64")
        if os.path.exists(ucrt_path):
            current_lib = os.environ.get("LIB", "")
            os.environ["LIB"] = f"{ucrt_path};{um_path};{current_lib}"

    return True


def check_requirements():
    """Check if all requirements are met"""
    print("\nChecking requirements...")

    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    if not torch.cuda.is_available():
        print("❌ CUDA not available in PyTorch")
        return False
    print(f"✅ CUDA available: {torch.version.cuda}")
    print(f"✅ GPU: {torch.cuda.get_device_name()}")

    # Check compiler
    try:
        result = subprocess.run(["cl"], capture_output=True, text=True)
        if "Microsoft" in result.stderr:
            print("✅ MSVC compiler available")
        else:
            print("❌ MSVC compiler not working")
            return False
    except FileNotFoundError:
        print("❌ MSVC compiler not found in PATH")
        return False

    # Check NVCC
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVCC compiler available")
        else:
            print("❌ NVCC compiler not working")
            return False
    except FileNotFoundError:
        print("❌ NVCC compiler not found")
        return False

    return True


def build_extension():
    """Build the CUDA extension"""
    print("\n" + "=" * 50)
    print("Building CUDA Extension")
    print("=" * 50)

    try:
        from torch.utils.cpp_extension import load

        print("Compiling CUDA extension...")

        # Get Visual Studio and MSVC paths for proper linking
        vs_path = find_visual_studio()
        sdk_path = find_windows_sdk()
        extra_ldflags = ["/MACHINE:X64"]

        if vs_path:
            msvc_path = find_msvc_compiler(vs_path)
            if msvc_path:
                # Get the MSVC version directory
                vc_tools_path = os.path.dirname(
                    os.path.dirname(os.path.dirname(msvc_path))
                )
                msvc_lib_path = os.path.join(vc_tools_path, "lib", "x64")

                extra_ldflags.append(f"/LIBPATH:{msvc_lib_path}")

        # Add Windows SDK library paths
        if sdk_path:
            ucrt_path = os.path.join(sdk_path, "ucrt", "x64")
            um_path = os.path.join(sdk_path, "um", "x64")
            if os.path.exists(ucrt_path):
                extra_ldflags.append(f"/LIBPATH:{ucrt_path}")
            if os.path.exists(um_path):
                extra_ldflags.append(f"/LIBPATH:{um_path}")

        # Add necessary system libraries
        extra_ldflags.extend(
            [
                "msvcrt.lib",
                "msvcprt.lib",
                "oldnames.lib",
                "kernel32.lib",
                "user32.lib",
                "ucrt.lib",
                "/NODEFAULTLIB:libcmt.lib",
            ]
        )

        squeeze_excitation_cuda = load(
            name="squeeze_excitation_cuda_ext",
            sources=["squeeze_excitation_cuda.cu"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode=arch=compute_86,code=sm_86",  # RTX 3060
                "--compiler-options",
                "/MD",  # Use dynamic runtime
            ],
            extra_ldflags=extra_ldflags,
            verbose=True,
        )

        print("✅ CUDA extension compiled successfully!")
        return squeeze_excitation_cuda

    except Exception as e:
        print(f"❌ Failed to compile CUDA extension: {e}")
        return None


def test_extension(extension):
    """Test the compiled extension"""
    print("\nTesting CUDA extension...")

    try:
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

        print("✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Extension test failed: {e}")
        return False


def main():
    """Main setup function"""
    print("CUDA SqueezeExcitation Setup for Windows")
    print("=" * 50)

    # Setup environment
    if not setup_environment():
        return 1

    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above.")
        return 1

    # Build extension
    extension = build_extension()
    if not extension:
        return 1

    # Test extension
    if not test_extension(extension):
        return 1

    print("\n" + "=" * 50)
    print("✅ CUDA SqueezeExcitation setup completed successfully!")
    print("You can now use the CUDA-accelerated SqueezeExcitation module.")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())

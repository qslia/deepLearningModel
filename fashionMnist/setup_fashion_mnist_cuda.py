from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Set environment variables for PyTorch 2.6.0 compatibility
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"  # RTX 3060 compute capability
os.environ["DISTUTILS_USE_SDK"] = "1"

# CUDA 12.6 + PyTorch 2.6.0 compatibility flags
extra_compile_args = {
    "cxx": [
        "-O2",
        "-std=c++17",
        "-DWITH_CUDA",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        # Disable problematic features
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ],
    "nvcc": [
        "-O2",
        "--use_fast_math",
        "-gencode=arch=compute_86,code=sm_86",  # RTX 3060 architecture
        "--extended-lambda",
        "-std=c++17",
        # Critical: CUDA 12.6 + PyTorch 2.6.0 compatibility
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DTORCH_EXTENSION_NAME=fashion_mnist_cnn_cuda",
        # Additional compatibility flags for CUDA 12.6
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        # Disable problematic CUDA features
        "-DCUDA_HAS_FP16=0",
        "-DCUDA_HAS_BF16=0",
        # Windows MSVC compatibility
        "-Xcompiler",
        "/wd4819",  # codepage warnings
        "-Xcompiler",
        "/wd4244",  # conversion warnings
        "-Xcompiler",
        "/wd4267",  # size_t warnings
        "-Xcompiler",
        "/wd4996",  # deprecated function warnings
        "-Xcompiler",
        "/wd4275",  # DLL interface warnings
        "-Xcompiler",
        "/wd4251",  # DLL interface warnings
        "-Xcompiler",
        "/EHsc",  # exception handling
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
}

# Setup for building the CUDA extension
setup(
    name="fashion_mnist_cnn_cuda",
    ext_modules=[
        CUDAExtension(
            name="fashion_mnist_cnn_cuda",
            sources=["fashion_mnist_cnn_cuda.cu"],
            extra_compile_args=extra_compile_args,
            include_dirs=[
                torch.utils.cpp_extension.include_paths()[0],  # Only main torch include
            ],
            define_macros=[
                ("WITH_CUDA", None),
                ("TORCH_API_INCLUDE_EXTENSION_H", None),
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

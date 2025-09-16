from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Set environment variables for PyTorch 2.6.0 compatibility
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"  # RTX 3060 compute capability
os.environ["DISTUTILS_USE_SDK"] = "1"

# PyTorch 2.6.0 specific compile flags
extra_compile_args = {
    "cxx": ["-O2", "-std=c++17", "-DWITH_CUDA", "-DTORCH_API_INCLUDE_EXTENSION_H"],
    "nvcc": [
        "-O2",
        "--use_fast_math",
        "-gencode=arch=compute_86,code=sm_86",  # RTX 3060 architecture
        "--extended-lambda",
        "-std=c++17",
        # PyTorch 2.6.0 compatibility flags
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DTORCH_EXTENSION_NAME=fashion_mnist_cnn_cuda",
        # Windows specific flags
        "-Xcompiler",
        "/wd4819,/wd4244,/wd4267,/wd4996,/wd4275,/wd4251",
        # Suppress problematic warnings
        "--diag-suppress",
        "767",  # pointer conversion warnings
        "--diag-suppress",
        "3326",  # operator new warnings
        "--diag-suppress",
        "3322",  # operator new warnings
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

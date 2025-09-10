from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Setup for building the CUDA extension
setup(
    name="fashion_mnist_cnn_cuda",
    ext_modules=[
        CUDAExtension(
            name="fashion_mnist_cnn_cuda",
            sources=[
                "fashion_mnist_cnn_cuda.cu",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

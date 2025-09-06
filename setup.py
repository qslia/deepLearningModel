from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from torch.utils import cpp_extension
import torch
import os

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    print("CUDA is available. Building CUDA extension...")

    # CUDA extension
    ext_modules = [
        cpp_extension.CUDAExtension(
            name="squeeze_excitation_cuda_ext",
            sources=[
                "squeeze_excitation_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_60,code=sm_60",
                    "-gencode=arch=compute_61,code=sm_61",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                ],
            },
            include_dirs=[
                # Path to pybind11 headers
                pybind11.get_include(),
            ],
        )
    ]

    cmdclass = {"build_ext": cpp_extension.BuildExtension}

else:
    print("CUDA is not available. Skipping CUDA extension build.")
    ext_modules = []
    cmdclass = {}

setup(
    name="squeeze_excitation_cuda",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="CUDA-accelerated Squeeze-and-Excitation implementation for PyTorch",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "numpy>=1.19.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="pytorch cuda squeeze-excitation deep-learning gpu-acceleration",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/squeeze-excitation-cuda/issues",
        "Source": "https://github.com/yourusername/squeeze-excitation-cuda",
    },
)

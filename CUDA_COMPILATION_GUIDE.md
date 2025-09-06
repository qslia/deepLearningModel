# CUDA Compilation Guide for Windows

This guide will help you compile CUDA extensions on Windows for the SqueezeExcitation implementation.

## Prerequisites

### 1. Visual Studio with C++ Build Tools
- **Visual Studio 2019 or 2022** (Community, Professional, or Enterprise)
- **C++ build tools** must be installed
- **Windows 10/11 SDK** (usually included)

### 2. CUDA Toolkit
- **CUDA 11.0 or later** (you have CUDA 12.1 ✅)
- **NVCC compiler** in PATH
- **Compatible GPU** (RTX 3060 ✅)

### 3. Python Environment
- **PyTorch with CUDA support** (you have 2.3.1+cu121 ✅)
- **Ninja build system** (installed ✅)

## Compilation Methods

### Method 1: Automated Setup (Recommended)

```bash
# Run the automated setup script
python setup_cuda_env.py
```

This script will:
- Find your Visual Studio installation
- Configure environment variables
- Set CUDA architecture for RTX 3060
- Compile and test the extension

### Method 2: Batch Script

```bash
# Run the Windows batch script
build_cuda_windows.bat
```

This will:
- Set up Visual Studio environment
- Check compiler availability
- Build the CUDA extension

### Method 3: Manual Setup

1. **Open Developer Command Prompt:**
   ```cmd
   "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
   ```

2. **Set CUDA Architecture:**
   ```cmd
   set TORCH_CUDA_ARCH_LIST=8.6
   ```

3. **Build Extension:**
   ```cmd
   python build_cuda_extension.py
   ```

### Method 4: Using setup.py

```bash
# Install with pip (production)
pip install -e .
```

## Common Issues and Solutions

### Issue 1: "cl.exe not found"
**Solution:**
- Ensure Visual Studio C++ build tools are installed
- Run from Developer Command Prompt
- Check PATH includes MSVC compiler directory

### Issue 2: "NVCC not found"
**Solution:**
- Install CUDA Toolkit
- Add CUDA bin directory to PATH
- Verify with: `nvcc --version`

### Issue 3: "Ninja is required"
**Solution:**
```bash
pip install ninja
```

### Issue 4: Architecture mismatch
**Solution:**
Set correct architecture for RTX 3060:
```bash
set TORCH_CUDA_ARCH_LIST=8.6
```

### Issue 5: Permission errors
**Solution:**
- Run as Administrator
- Check antivirus software
- Clear PyTorch extension cache

## Environment Variables

Set these for optimal compilation:

```cmd
set TORCH_CUDA_ARCH_LIST=8.6
set DISTUTILS_USE_SDK=1
set MSSdk=1
```

## Verification

After successful compilation, verify with:

```python
import torch
from squeeze_excitation_cuda import CUDA_AVAILABLE

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA extension available: {CUDA_AVAILABLE}")

# Test the extension
if CUDA_AVAILABLE:
    from squeeze_excitation_cuda import SqueezeExcitationCUDA
    se = SqueezeExcitationCUDA(64, 16).cuda()
    x = torch.randn(1, 64, 32, 32).cuda()
    output = se(x)
    print(f"Test successful: {x.shape} -> {output.shape}")
```

## Performance Tips

1. **Use Release Mode:**
   - Compile with `-O3` optimization
   - Use `--use_fast_math` for CUDA

2. **Set Correct Architecture:**
   - RTX 3060: `compute_86,code=sm_86`
   - Ensures optimal GPU utilization

3. **Clear Cache if Issues:**
   ```python
   import torch
   torch.utils.cpp_extension.clear_cache()
   ```

## Troubleshooting Commands

```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA toolkit
nvcc --version

# Check Visual Studio
where cl

# Check Ninja
ninja --version

# Clear extension cache
python -c "import torch; torch.utils.cpp_extension.clear_cache()"
```

## Next Steps

Once compilation is successful:

1. **Run tests:**
   ```bash
   python test_cuda_se.py
   ```

2. **Integrate with your model:**
   ```python
   from squeeze_excitation_cuda import create_model_cuda
   model = create_model_cuda("tf_efficientnet_b4_ns", num_classes=5)
   ```

3. **Benchmark performance:**
   ```bash
   python integrate_cuda_se.py
   ```

## Support

If you encounter issues:
1. Check this guide first
2. Verify all prerequisites
3. Try the automated setup script
4. Clear PyTorch extension cache
5. Restart your development environment

The CUDA SqueezeExcitation implementation will provide significant speedups for your Cassava leaf disease classification model!

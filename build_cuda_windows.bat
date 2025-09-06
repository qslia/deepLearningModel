nvcc@echo off
echo Setting up Visual Studio 2022 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo Checking compiler availability...
where cl
if %errorlevel% neq 0 (
    echo ERROR: MSVC compiler not found!
    echo Please ensure Visual Studio 2022 with C++ tools is installed.
    pause
    exit /b 1
)

echo.
echo Checking CUDA availability...
where nvcc
if %errorlevel% neq 0 (
    echo ERROR: NVCC compiler not found!
    echo Please ensure CUDA toolkit is installed and in PATH.
    pause
    exit /b 1
)

echo.
echo Setting CUDA architecture for RTX 3060 (compute capability 8.6)...
set TORCH_CUDA_ARCH_LIST=8.6

echo.
echo Building CUDA extension...
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
"

echo.
echo Starting build process...
python build_cuda_extension.py

pause

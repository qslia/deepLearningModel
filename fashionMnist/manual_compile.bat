@echo off
echo Manual CUDA Compilation for CUDA 12.6 + PyTorch 2.6.0
echo =====================================================

echo.
echo 1. Setting up Visual Studio environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo 2. Setting environment variables...
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.6
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set CUDA_PATH=%CUDA_HOME%

echo.
echo 3. Cleaning previous builds...
if exist build rmdir /s /q build
if exist fashion_mnist_cnn_cuda*.pyd del fashion_mnist_cnn_cuda*.pyd

echo.
echo 4. Checking tools...
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: MSVC compiler not found
    pause
    exit /b 1
)
echo Found MSVC compiler: 
where cl.exe

where nvcc.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: NVCC compiler not found
    pause
    exit /b 1
)
echo Found NVCC compiler:
where nvcc.exe

echo.
echo 5. Starting manual compilation...
echo This may take several minutes...
python setup_fashion_mnist_cuda.py build_ext --inplace --verbose

if errorlevel 1 (
    echo.
    echo Compilation failed. Trying alternative method...
    echo.
    echo Attempting direct NVCC compilation...
    
    REM Get Python include path
    for /f %%i in ('python -c "import sysconfig; print(sysconfig.get_path('include'))"') do set PYTHON_INCLUDE=%%i
    for /f %%i in ('python -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])"') do set TORCH_INCLUDE=%%i
    
    nvcc -shared -Xcompiler /MD -o fashion_mnist_cnn_cuda.pyd fashion_mnist_cnn_cuda.cu ^
        -I"%PYTHON_INCLUDE%" -I"%TORCH_INCLUDE%" ^
        -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include" ^
        -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64" -lcudart ^
        -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ ^
        -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ ^
        -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=fashion_mnist_cnn_cuda ^
        --expt-relaxed-constexpr --expt-extended-lambda ^
        -gencode=arch=compute_86,code=sm_86 -std=c++17
    
    if errorlevel 1 (
        echo ERROR: All compilation methods failed
        pause
        exit /b 1
    ) else (
        echo SUCCESS: Direct NVCC compilation worked!
    )
) else (
    echo SUCCESS: Setup.py compilation worked!
)

echo.
echo 6. Testing compiled module...
python -c "import fashion_mnist_cnn_cuda; print('SUCCESS: Module imports correctly')" 2>nul
if errorlevel 1 (
    echo Warning: Direct import failed, testing through model...
    python -c "from fashion_mnist_cnn_model import create_model; create_model(); print('SUCCESS: Model creation works')"
) else (
    echo Module imports successfully!
)

echo.
echo 7. Running test suite...
echo Press 'n' when prompted about Fashion-MNIST download
python test_fashion_mnist_cuda.py

echo.
echo =====================================================
echo Manual Compilation Complete!
echo Your CUDA extension is now compiled without JIT.
pause

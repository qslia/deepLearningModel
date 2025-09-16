# Manual CUDA Compilation Script for CUDA 12.6 + PyTorch 2.6.0
# This script compiles the CUDA extension without JIT compilation

Write-Host "Manual CUDA Extension Compilation" -ForegroundColor Green
Write-Host "=" * 50

# Step 1: Clean previous builds
Write-Host "`n1. Cleaning previous builds..." -ForegroundColor Yellow
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
Remove-Item fashion_mnist_cnn_cuda*.pyd -ErrorAction SilentlyContinue
Remove-Item fashion_mnist_cnn_cuda*.so -ErrorAction SilentlyContinue
Write-Host "✓ Cleaned build artifacts" -ForegroundColor Green

# Step 2: Setup Visual Studio environment
Write-Host "`n2. Setting up Visual Studio environment..." -ForegroundColor Yellow
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && set' | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)') {
        Set-Item "env:$($matches[1])" $matches[2]
    }
}

# Verify compiler is available
$clPath = Get-Command cl -ErrorAction SilentlyContinue
if ($clPath) {
    Write-Host "✓ MSVC compiler found: $($clPath.Source)" -ForegroundColor Green
} else {
    Write-Host "✗ MSVC compiler not found. Please install Visual Studio 2022." -ForegroundColor Red
    exit 1
}

# Step 3: Check CUDA
Write-Host "`n3. Checking CUDA availability..." -ForegroundColor Yellow
$nvccPath = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvccPath) {
    Write-Host "✓ NVCC found: $($nvccPath.Source)" -ForegroundColor Green
    $cudaVersion = nvcc --version | Select-String "release"
    Write-Host "  $cudaVersion" -ForegroundColor Cyan
} else {
    Write-Host "✗ NVCC not found. Please install CUDA Toolkit." -ForegroundColor Red
    exit 1
}

# Step 4: Set critical environment variables
Write-Host "`n4. Setting environment variables..." -ForegroundColor Yellow
$env:DISTUTILS_USE_SDK = "1"
$env:TORCH_CUDA_ARCH_LIST = "8.6"  # RTX 3060
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:CUDA_PATH = $env:CUDA_HOME

# Add required paths
$env:PATH = $env:PATH + ";C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
$env:INCLUDE = $env:INCLUDE + ";C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt"
$env:INCLUDE = $env:INCLUDE + ";C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared"
$env:INCLUDE = $env:INCLUDE + ";C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um"
$env:LIB = $env:LIB + ";C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64"
$env:LIB = $env:LIB + ";C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64"

Write-Host "✓ Environment variables set" -ForegroundColor Green

# Step 5: Disable JIT compilation in the model
Write-Host "`n5. Disabling JIT compilation..." -ForegroundColor Yellow
$modelFile = "fashion_mnist_cnn_model.py"
$content = Get-Content $modelFile -Raw
$newContent = $content -replace 'import fashion_mnist_cnn_cuda as cuda_module\s*print\("✓ Using pre-compiled CUDA module"\)', 'raise ImportError("Force manual compilation")'
Set-Content $modelFile $newContent
Write-Host "✓ JIT compilation disabled" -ForegroundColor Green

# Step 6: Manual compilation
Write-Host "`n6. Starting manual compilation..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Cyan

try {
    # Use Python to build the extension
    python setup_fashion_mnist_cuda.py build_ext --inplace --verbose
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Manual compilation successful!" -ForegroundColor Green
        
        # Check if .pyd file was created
        $pydFile = Get-ChildItem -Name "fashion_mnist_cnn_cuda*.pyd" -ErrorAction SilentlyContinue
        if ($pydFile) {
            Write-Host "✓ Generated: $pydFile" -ForegroundColor Green
        } else {
            Write-Host "⚠ No .pyd file found, checking build directory..." -ForegroundColor Yellow
            Get-ChildItem -Recurse -Name "*.pyd" | ForEach-Object { Write-Host "  Found: $_" -ForegroundColor Cyan }
        }
    } else {
        throw "Compilation failed with exit code $LASTEXITCODE"
    }
} catch {
    Write-Host "✗ Compilation failed: $_" -ForegroundColor Red
    Write-Host "`nTrying alternative compilation method..." -ForegroundColor Yellow
    
    # Alternative: Direct nvcc compilation
    Write-Host "Attempting direct NVCC compilation..." -ForegroundColor Cyan
    
    # Get Python and PyTorch paths
    $pythonInclude = python -c "import sysconfig; print(sysconfig.get_path('include'))"
    $torchInclude = python -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])"
    
    $nvccCmd = @"
nvcc -shared -Xcompiler /MD -o fashion_mnist_cnn_cuda.pyd fashion_mnist_cnn_cuda.cu
-I"$pythonInclude" -I"$torchInclude"
-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include"
-L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64" -lcudart
-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
-D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
-DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=fashion_mnist_cnn_cuda
--expt-relaxed-constexpr --expt-extended-lambda
-gencode=arch=compute_86,code=sm_86 -std=c++17
"@
    
    Invoke-Expression $nvccCmd
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Direct NVCC compilation successful!" -ForegroundColor Green
    } else {
        Write-Host "✗ All compilation methods failed" -ForegroundColor Red
        exit 1
    }
}

# Step 7: Restore original model file
Write-Host "`n7. Restoring original model file..." -ForegroundColor Yellow
$originalContent = $content -replace 'raise ImportError\("Force manual compilation"\)', 'import fashion_mnist_cnn_cuda as cuda_module\n    print("✓ Using pre-compiled CUDA module")'
Set-Content $modelFile $originalContent
Write-Host "✓ Model file restored" -ForegroundColor Green

# Step 8: Test the compiled module
Write-Host "`n8. Testing compiled module..." -ForegroundColor Yellow
try {
    python -c "import fashion_mnist_cnn_cuda; print('✓ Manual compilation test: SUCCESS')"
    Write-Host "✓ Module imports successfully" -ForegroundColor Green
} catch {
    Write-Host "⚠ Direct import failed, testing through model..." -ForegroundColor Yellow
    python -c "from fashion_mnist_cnn_model import create_model; create_model(); print('✓ Model creation test: SUCCESS')"
}

# Step 9: Run full test suite
Write-Host "`n9. Running test suite..." -ForegroundColor Yellow
Write-Host "Press 'n' when prompted about Fashion-MNIST download" -ForegroundColor Cyan
python test_fashion_mnist_cuda.py

Write-Host "`n" + "=" * 50
Write-Host "Manual Compilation Complete!" -ForegroundColor Green
Write-Host "Your CUDA extension has been compiled manually without JIT." -ForegroundColor Green

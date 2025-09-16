win+R cmd
cmd /k "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
where cl
set PATH=%PATH%;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
set DISTUTILS_USE_SDK=1
python setup_fashion_mnist_cuda.py build_ext --inplace
python test_fashion_mnist_cuda.py


Get-Command cl -ErrorAction SilentlyContinue
# Instead of cmd /k, use PowerShell to call the VS setup
& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

# Or better yet, use the PowerShell module approach:
Import-Module "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath "C:\Program Files\Microsoft Visual Studio\2022\Community" -SkipAutomaticLocation

# Check if cl.exe is available
Get-Command cl -ErrorAction SilentlyContinue

# Add to PATH if needed
$env:PATH += ";C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"

# 2. Check compiler availability
Get-Command cl

# 3. Set environment variables
$env:DISTUTILS_USE_SDK = "1"

python setup_fashion_mnist_cuda.py build_ext --inplace
python test_fashion_mnist_cuda.py

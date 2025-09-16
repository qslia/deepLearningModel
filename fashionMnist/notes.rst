win+R cmd
cmd /k "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
where cl
set PATH=%PATH%;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
set DISTUTILS_USE_SDK=1
python setup_fashion_mnist_cuda.py build_ext --inplace
python test_fashion_mnist_cuda.py

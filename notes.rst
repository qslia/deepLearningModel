win+R cmd
cmd /k "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
where cl
set PATH=%PATH%;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
python setup_cuda_env.py
==================================================
✅ CUDA SqueezeExcitation setup completed successfully!
You can now use the CUDA-accelerated SqueezeExcitation module.
==================================================


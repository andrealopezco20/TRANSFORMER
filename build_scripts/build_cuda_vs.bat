@echo off
echo ===============================================
echo  COMPILADOR CUDA PURO (Visual Studio)
echo ===============================================

REM Configurar entorno Visual Studio
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Limpiar compilaciones anteriores
if exist "obj_vs" rmdir /s /q "obj_vs"
if exist "TransformerCUDA_VS.exe" del "TransformerCUDA_VS.exe"
mkdir obj_vs

echo.
echo ✅ Compilando kernels CUDA...
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math -arch=sm_86 ^
     -Iinclude -c src/matrix_cuda.cu -o obj_vs/matrix_cuda.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando CUDA
    pause
    exit /b 1
)

echo ✅ CUDA compilado

echo.
echo Compilando archivos C++ con MSVC...
cl /O2 /EHsc /DUSE_CUDA /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/matrix.cpp /Fo:obj_vs/matrix.obj

cl /O2 /EHsc /DUSE_CUDA /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/mnist_loader.cpp /Fo:obj_vs/mnist_loader.obj

cl /O2 /EHsc /DUSE_CUDA /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /c src/transformer.cpp /Fo:obj_vs/transformer.obj

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando C++
    pause
    exit /b 1
)

echo ✅ C++ compilado

echo.
echo Enlazando con MSVC...
cl /O2 /EHsc /DUSE_CUDA /I"include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   main_fast.cpp obj_vs/matrix.obj obj_vs/mnist_loader.obj obj_vs/transformer.obj obj_vs/matrix_cuda.obj ^
   /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" ^
   cudart.lib cublas.lib /OUT:TransformerCUDA_VS.exe

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado
    pause
    exit /b 1
)

echo.
echo 🎉 ¡PROYECTO CUDA COMPILADO EXITOSAMENTE CON VISUAL STUDIO!
echo.
echo 📊 Tu Transformer ahora usa:
echo   - GPU: RTX 3050 Laptop
echo   - CUDA: 12.8
echo   - Compilador: Visual Studio 2022
echo   - Optimización máxima
echo.
echo Ejecutable: TransformerCUDA_VS.exe
echo.
echo Para ejecutar:
echo   .\TransformerCUDA_VS.exe
echo.
echo 🚀 ¡Listo para entrenar con aceleración GPU!
echo.
pause

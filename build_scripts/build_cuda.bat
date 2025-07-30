@echo off
echo ===================================
echo  Compilador CUDA Transformer v2.0
echo ===================================

REM Verificar que NVCC esté disponible
nvcc --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Error: NVCC no encontrado. Asegúrate de tener CUDA instalado y en el PATH.
    echo Verifica que CUDA Toolkit esté instalado correctamente.
    pause
    exit /b 1
)

echo ✅ NVCC encontrado
nvcc --version

REM Crear directorio obj si no existe
if not exist "obj" mkdir obj

echo.
echo Compilando operaciones CUDA...
nvcc -std=c++17 -O3 -DUSE_CUDA --use_fast_math -arch=sm_61 -Iinclude -c src/matrix_cuda.cu -o obj/matrix_cuda.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando CUDA
    pause
    exit /b 1
)

echo ✅ Operaciones CUDA compiladas

echo.
echo Compilando archivos C++...
g++ -std=c++17 -O3 -Wall -DUSE_CUDA -Iinclude -c src/matrix.cpp -o obj/matrix.o
g++ -std=c++17 -O3 -Wall -DUSE_CUDA -Iinclude -c src/mnist_loader.cpp -o obj/mnist_loader.o
g++ -std=c++17 -O3 -Wall -DUSE_CUDA -Iinclude -c src/transformer.cpp -o obj/transformer.o

if %ERRORLEVEL% neq 0 (
    echo ❌ Error compilando archivos C++
    pause
    exit /b 1
)

echo ✅ Archivos C++ compilados

echo.
echo Enlazando proyecto completo...
g++ -std=c++17 -O3 -Wall -DUSE_CUDA -Iinclude -o FashionMNISTTransformer_CUDA.exe main.cpp obj/matrix.o obj/mnist_loader.o obj/transformer.o obj/matrix_cuda.o -lcudart

if %ERRORLEVEL% neq 0 (
    echo ❌ Error en el enlazado
    pause
    exit /b 1
)

echo ✅ Proyecto compilado exitosamente: FashionMNISTTransformer_CUDA.exe

echo.
echo Compilando prueba CUDA...
g++ -std=c++17 -O3 -Wall -DUSE_CUDA -Iinclude -o test_cuda.exe test_cuda.cpp obj/matrix_cuda.o -lcudart

if %ERRORLEVEL% neq 0 (
    echo ⚠️  Advertencia: No se pudo compilar la prueba CUDA, pero el proyecto principal está listo
) else (
    echo ✅ Prueba CUDA compilada: test_cuda.exe
)

echo.
echo 🎉 ¡Compilación completada!
echo.
echo Archivos generados:
echo   - FashionMNISTTransformer_CUDA.exe (proyecto principal)
echo   - test_cuda.exe (prueba CUDA)
echo.
echo Para ejecutar:
echo   .\FashionMNISTTransformer_CUDA.exe
echo   .\test_cuda.exe
echo.
pause

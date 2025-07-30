#!/bin/bash

echo "================================================"
echo " DESCARGADOR DE DATASET FASHION-MNIST"
echo "================================================"
echo

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Verificar si wget está disponible
if ! command -v wget &> /dev/null; then
    echo -e "${RED}❌ wget no encontrado. Instalando...${NC}"
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y wget
    elif command -v yum &> /dev/null; then
        sudo yum install -y wget
    else
        echo -e "${RED}❌ No se pudo instalar wget automáticamente${NC}"
        echo "Por favor instala wget manualmente"
        exit 1
    fi
fi

echo -e "${BLUE}📥 Descargando Fashion-MNIST dataset...${NC}"
echo

# URLs del dataset
BASE_URL="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

# Función para descargar y verificar
download_file() {
    local filename=$1
    local expected_size=$2
    
    echo -e "${BLUE}⏬ Descargando ${filename}...${NC}"
    
    if wget -q --show-progress "${BASE_URL}${filename}.gz"; then
        echo -e "${GREEN}✅ ${filename}.gz descargado${NC}"
        
        echo -e "${BLUE}📦 Descomprimiendo ${filename}...${NC}"
        if gunzip -f "${filename}.gz"; then
            echo -e "${GREEN}✅ ${filename} descomprimido${NC}"
            
            # Verificar tamaño del archivo
            if [ -n "$expected_size" ]; then
                actual_size=$(stat -c%s "$filename" 2>/dev/null || stat -f%z "$filename" 2>/dev/null)
                if [ "$actual_size" -eq "$expected_size" ]; then
                    echo -e "${GREEN}✅ Tamaño verificado: $actual_size bytes${NC}"
                else
                    echo -e "${YELLOW}⚠️  Tamaño diferente: $actual_size bytes (esperado: $expected_size)${NC}"
                fi
            fi
        else
            echo -e "${RED}❌ Error descomprimiendo ${filename}${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Error descargando ${filename}${NC}"
        return 1
    fi
    echo
}

# Descargar archivos del dataset
echo -e "${YELLOW}🎯 Fashion-MNIST Dataset (70,000 imágenes 28x28)${NC}"
echo

download_file "train-images-idx3-ubyte" 47040016  # ~47 MB
download_file "train-labels-idx1-ubyte" 60008     # ~60 KB
download_file "t10k-images-idx3-ubyte" 7840016    # ~7.8 MB
download_file "t10k-labels-idx1-ubyte" 10008      # ~10 KB

# Verificar que todos los archivos están presentes
echo -e "${BLUE}🔍 Verificando archivos descargados...${NC}"
echo

all_files_ok=true

check_file() {
    local filename=$1
    if [ -f "$filename" ]; then
        local size=$(stat -c%s "$filename" 2>/dev/null || stat -f%z "$filename" 2>/dev/null)
        echo -e "${GREEN}✅ $filename ($size bytes)${NC}"
    else
        echo -e "${RED}❌ $filename no encontrado${NC}"
        all_files_ok=false
    fi
}

check_file "train-images-idx3-ubyte"
check_file "train-labels-idx1-ubyte"
check_file "t10k-images-idx3-ubyte"
check_file "t10k-labels-idx1-ubyte"

echo

if [ "$all_files_ok" = true ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN} ✅ DATASET DESCARGADO CORRECTAMENTE${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo
    echo -e "${BLUE}📊 Dataset Fashion-MNIST listo:${NC}"
    echo "   • 60,000 imágenes de entrenamiento"
    echo "   • 10,000 imágenes de prueba"
    echo "   • 10 clases de ropa"
    echo "   • Formato: 28x28 píxeles en escala de grises"
    echo
    echo -e "${BLUE}🚀 Ahora puedes ejecutar:${NC}"
    echo "   ./build_wsl.sh"
    echo "   o"
    echo "   mkdir build && cd build && cmake .. && make"
    echo
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED} ❌ ERROR EN LA DESCARGA${NC}"
    echo -e "${RED}========================================${NC}"
    echo
    echo -e "${YELLOW}💡 Puedes descargar manualmente desde:${NC}"
    echo "   http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    echo
fi
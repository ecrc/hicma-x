#!/bin/bash

# @copyright (c) 2023-2025     Saint Louis University (SLU)
# @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
# @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
# @copyright (c) 2023-2025     The University of Tennessee and The University of Tennessee Research Foundation
#                              All rights reserved.

# HICMA PARSEC Release Build Script
# This script helps build HICMA PARSEC for release

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build HICMA PARSEC for release

OPTIONS:
    -h, --help              Show this help message
    -b, --build-dir DIR     Build directory (default: build)
    -j, --jobs N            Number of parallel jobs (default: auto)
    -c, --clean             Clean build directory before building
    -t, --test              Run tests after building
    -i, --install           Install after building
    -p, --prefix DIR        Installation prefix (default: /usr/local)
    --cuda                  Enable CUDA support (default: off, CPU-only build)
    --hip                   Enable HIP support
    --fugaku                Enable Fugaku-specific optimizations
    --sanitizer             Enable AddressSanitizer
    --cutlass               Enable CUTLASS support (requires CUDA)
    --static                Build static libraries instead of shared
    --debug                 Build in debug mode
    --release               Build in release mode (default)

EXAMPLES:
    $0 --clean --test --install
    $0 --cuda --jobs 8 --prefix /opt/hicma
    $0 --hip --fugaku --static

EOF
}

# Default values
BUILD_DIR="build"
JOBS=$(nproc 2>/dev/null || echo 4)
CLEAN=false
RUN_TESTS=false
INSTALL=false
PREFIX="/usr/local"
ENABLE_CUDA=false
ENABLE_HIP=false
ENABLE_FUGAKU=false
ENABLE_SANITIZER=false
ENABLE_CUTLASS=false
BUILD_SHARED=true
BUILD_TYPE="RelWithDebInfo"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -t|--test)
            RUN_TESTS=true
            shift
            ;;
        -i|--install)
            INSTALL=true
            shift
            ;;
        -p|--prefix)
            PREFIX="$2"
            shift 2
            ;;
        --cuda)
            ENABLE_CUDA=true
            shift
            ;;
        --hip)
            ENABLE_HIP=true
            shift
            ;;
        --fugaku)
            ENABLE_FUGAKU=true
            shift
            ;;
        --sanitizer)
            ENABLE_SANITIZER=true
            shift
            ;;
        --cutlass)
            ENABLE_CUTLASS=true
            shift
            ;;
        --static)
            BUILD_SHARED=false
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate options
if [[ "$ENABLE_CUDA" == "true" && "$ENABLE_HIP" == "true" ]]; then
    print_error "CUDA and HIP cannot be enabled simultaneously"
    exit 1
fi

if [[ "$ENABLE_CUTLASS" == "true" && "$ENABLE_CUDA" == "false" ]]; then
    print_error "CUTLASS requires CUDA support"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

print_status "HICMA PARSEC Release Build Script"
print_status "Project directory: $PROJECT_DIR"
print_status "Build directory: $BUILD_DIR"
print_status "Installation prefix: $PREFIX"

# Change to project directory
cd "$PROJECT_DIR"

# Check if we're in the right directory
if [[ ! -f "CMakeLists.txt" ]]; then
    print_error "CMakeLists.txt not found. Are you in the correct directory?"
    exit 1
fi

# Check if submodules are present
if [[ ! -d "dplasma" ]] || [[ ! -d "stars-h" ]] || [[ ! -d "hcore" ]]; then
    print_error "Required submodules (dplasma, stars-h, hcore) not found. Please run: git submodule update --init --recursive"
    exit 1
fi

# Check if nested submodules are present
if [[ ! -d "stars-h/cmake_modules/ecrc/modules" ]]; then
    print_error "Nested submodule stars-h/cmake_modules/ecrc not found. Please run: git submodule update --init --recursive"
    exit 1
fi

if [[ ! -d "hcore/cmake_modules/ecrc/modules" ]]; then
    print_error "Nested submodule hcore/cmake_modules/ecrc not found. Please run: git submodule update --init --recursive"
    exit 1
fi

# Clean build directory if requested
if [[ "$CLEAN" == "true" ]]; then
    print_status "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake
print_status "Configuring CMake..."

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
    -DBUILD_SHARED_LIBS="$BUILD_SHARED"
    -DHICMA_PARSEC_HAVE_CUDA="$ENABLE_CUDA"
    -DHICMA_PARSEC_HAVE_HIP="$ENABLE_HIP"
    -DON_FUGAKU="$ENABLE_FUGAKU"
    -DENABLE_SANITIZER="$ENABLE_SANITIZER"
    -DENABLE_CUTLASS="$ENABLE_CUTLASS"
    -DPARSEC_MAX_DEP_OUT_COUNT=16
)

# Add CUDA-specific options
if [[ "$ENABLE_CUDA" == "true" ]]; then
    if [[ -n "$CUTLASS_ROOT" ]]; then
        CMAKE_ARGS+=(-DCUTLASS_ROOT="$CUTLASS_ROOT")
    fi
    if [[ -n "$CMAKE_CUDA_ARCHITECTURES" ]]; then
        CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES")
    fi
fi

# Add HIP-specific options
if [[ "$ENABLE_HIP" == "true" ]]; then
    if [[ -n "$ROCM_PATH" ]]; then
        CMAKE_ARGS+=(-DROCM_PATH="$ROCM_PATH")
    fi
fi

print_status "CMake arguments: ${CMAKE_ARGS[*]}"

cmake "${CMAKE_ARGS[@]}" ..

if [[ $? -ne 0 ]]; then
    print_error "CMake configuration failed"
    exit 1
fi

print_success "CMake configuration completed"

# Build
print_status "Building with $JOBS parallel jobs..."
make -j"$JOBS"

if [[ $? -ne 0 ]]; then
    print_error "Build failed"
    exit 1
fi

print_success "Build completed"

# Run tests if requested
if [[ "$RUN_TESTS" == "true" ]]; then
    print_status "Running tests..."
    # The GPU-labelled cases need a device; run them only for a GPU build
    CTEST_ARGS=(--output-on-failure -L hicma)
    if [[ "$ENABLE_CUDA" != "true" && "$ENABLE_HIP" != "true" ]]; then
        CTEST_ARGS+=(-LE gpu)
    fi
    ctest "${CTEST_ARGS[@]}"

    if [[ $? -ne 0 ]]; then
        print_error "Some tests failed"
        exit 1
    else
        print_success "All tests passed"
    fi
fi

# Install if requested
if [[ "$INSTALL" == "true" ]]; then
    print_status "Installing to $PREFIX..."
    make install
    
    if [[ $? -ne 0 ]]; then
        print_error "Installation failed"
        exit 1
    fi
    
    print_success "Installation completed"
fi

# Print summary
print_success "Build process completed successfully!"
print_status "Build configuration:"
print_status "  Build type: $BUILD_TYPE"
print_status "  Library type: $([ "$BUILD_SHARED" == "true" ] && echo "Shared" || echo "Static")"
print_status "  CUDA support: $([ "$ENABLE_CUDA" == "true" ] && echo "Enabled" || echo "Disabled")"
print_status "  HIP support: $([ "$ENABLE_HIP" == "true" ] && echo "Enabled" || echo "Disabled")"
print_status "  Fugaku mode: $([ "$ENABLE_FUGAKU" == "true" ] && echo "Enabled" || echo "Disabled")"
print_status "  AddressSanitizer: $([ "$ENABLE_SANITIZER" == "true" ] && echo "Enabled" || echo "Disabled")"
print_status "  CUTLASS support: $([ "$ENABLE_CUTLASS" == "true" ] && echo "Enabled" || echo "Disabled")"

if [[ "$INSTALL" == "true" ]]; then
    print_status "Installation location: $PREFIX"
fi 

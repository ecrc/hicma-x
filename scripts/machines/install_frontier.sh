#!/bin/bash

# Script to fix the build_test ninja build issue
# This script addresses the HIP include path problem

echo "Fixing build_test ninja build issue..."

# Clean and recreate build_test directory
echo "Cleaning build_test directory..."
rm -rf build_test
mkdir -p build_test
cd build_test

# Configure with proper HIP include paths and platform definitions
echo "Configuring build_test with proper HIP include paths and platform definitions..."
cmake ../ \
  -DCMAKE_INSTALL_PREFIX=`pwd`/installdir \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpic++ \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DHICMA_PARSEC_HAVE_HIP=ON \
  -DCMAKE_HIP_FLAGS="-I$CRAY_MPICH_DIR/include -D__HIP_PLATFORM_AMD__" \
  -DCMAKE_C_FLAGS="-I$OLCF_OPENBLAS_ROOT/include -I$ROCM_PATH/include -DLAPACKE_UTILS -D__HIP_PLATFORM_AMD__" \
  -DCMAKE_CXX_FLAGS="-I$OLCF_OPENBLAS_ROOT/include -I$ROCM_PATH/include -DLAPACKE_UTILS -D__HIP_PLATFORM_AMD__" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$ROCM_PATH/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$ROCM_PATH/lib" \
  -DROCM_PATH="$ROCM_PATH"

if [ $? -eq 0 ]; then
    echo "Configuration successful! Now building with ninja -j8..."
    ninja -j8
    if [ $? -eq 0 ]; then
        echo "Build successful!"
    else
        echo "Build failed. Check the error messages above."
    fi
else
    echo "Configuration failed. Check the error messages above."
fi

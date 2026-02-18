#/bin/bash

module list -l

home=$PWD

echo "update submodules"
#git submodule update --init --recursive

cd  $home && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE="Release" -DDPLASMA_PRECISIONS="s;d" -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DPARSEC_MAX_DEP_OUT_COUNT=16 -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran -DHICMA_PARSEC_HAVE_CUDA=ON

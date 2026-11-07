#/bin/bash

# env
#module load xl spectrum-mpi cuda lsf-tools essl pkgconf hwloc/2.5.0 cmake python/2.7.15 libxml2 gsl

ulimit -s 8388608

module list -l

#cd hicma-x-dev
home=$PWD

echo "update submodules"
#git submodule update --init --recursive

####################################################################################
echo "Install starsh"
cd stars-h && mkdir -p build && cd build
export LAPACKE_DIR=
make .. -DCMAKE_C_COMPILER=gcc -DCMAKE_INSTALL_PREFIX=$PWD/installdir -DMPI=OFF -DOPENMP=OFF -DSTARPU=ON -DEXAMPLES=OFF -DTESTING=OFF -DBUILD_SHARED_LIBS=ON -DBLAS_LIBRARIES="${LAPACKE_ROOT}/lib64/libblas.a" -DCBLAS_LIBRARIES="${LAPACKE_ROOT}/lib64/libcblas.a" -DGSL=OFF
make -j 8 install
echo "export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/env_hicma_parsec.sh
export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Install hcore"
cd $home/hcore
find . -type f -exec sed -i 's/zlacpy_/LAPACKE_zlacpy/g' {} +
find . -type f -exec sed -i 's/dlacpy_/LAPACKE_dlacpy/g' {} +
find . -type f -exec sed -i 's/slacpy_/LAPACKE_slacpy/g' {} +

find . -type f -exec sed -i 's/zlaset_/LAPACKE_zlaset/g' {} +
find . -type f -exec sed -i 's/dlaset_/LAPACKE_dlaset/g' {} +
find . -type f -exec sed -i 's/slaset_/LAPACKE_slaset/g' {} +

find . -type f -exec sed -i 's/zlange_/LAPACKE_zlange/g' {} +
find . -type f -exec sed -i 's/dlange_/LAPACKE_dlange/g' {} +
find . -type f -exec sed -i 's/slange_/LAPACKE_slange/g' {} +

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DCMAKE_C_COMPILER=gcc -DHCORE_TESTING=OFF
#cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DCMAKE_C_COMPILER=xlc_r -DBLAS_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so" -DCBLAS_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so" -DLAPACKE_DIR="${LAPACKE_ROOT}" -DLAPACKE_LIBRARIES="${LAPACKE_ROOT}/lib/liblapacke.a" -DLAPACK_LIBRARIES="${LAPACKE_ROOT}/lib/liblapack.a;${LAPACKE_ROOT}/lib/liblapacke.a" -DCMAKE_EXE_LINKER_FLAGS="-L ${LAPACKE_ROOT}/lib" -DHCORE_TESTING=OFF

make -j 8 install
echo "export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/env_hicma_parsec.sh
export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Install hicma_parsec"
cd $home && mkdir -p build && cd build


cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DCMAKE_BUILD_TYPE=Release -DPARSEC_GPU_WITH_CUDA=ON -DPARSEC_HAVE_DEV_CUDA_SUPPORT=ON -DPARSEC_GPU_WITH_HIP=OFF -DDPLASMA_PRECISIONS="s;d" -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_INSTALL_PREFIX=$PWD/installdir -DON_FUGAKU=OFF -DCMAKE_EXE_LINKER_FLAGS="-L ${CUDA_DIR}/lib64 -lgfortran" -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DPARSEC_MAX_DEP_OUT_COUNT=16 -G Ninja -DBUILD_SHARED_LIBS=OFF -DBLAS_LIBRARIES=${LAPACKE_ROOT}/lib64/libblas.a

# Debug
#cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DCMAKE_BUILD_TYPE=Debug -DPARSEC_GPU_WITH_CUDA=ON -DPARSEC_HAVE_DEV_CUDA_SUPPORT=ON -DPARSEC_GPU_WITH_HIP=OFF -DDPLASMA_PRECISIONS="s;d" -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_INSTALL_PREFIX=$PWD/installdir -DON_FUGAKU=OFF -DCMAKE_EXE_LINKER_FLAGS="-L ${CUDA_DIR}/lib64 -lgfortran" -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DPARSEC_MAX_DEP_OUT_COUNT=16 -G Ninja -DBUILD_SHARED_LIBS=OFF -DBLAS_LIBRARIES=${LAPACKE_ROOT}/lib64/libblas.a -DPARSEC_DEBUG_MEM_ADDR=ON

#cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DCMAKE_BUILD_TYPE=Release -DPARSEC_GPU_WITH_CUDA=ON -DPARSEC_HAVE_DEV_CUDA_SUPPORT=ON -DPARSEC_GPU_WITH_HIP=OFF -DDPLASMA_PRECISIONS="s;d" -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_INSTALL_PREFIX=$PWD/installdir -DON_FUGAKU=OFF -DCMAKE_EXE_LINKER_FLAGS="-L ${CUDA_DIR}/lib64" -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DPARSEC_MAX_DEP_OUT_COUNT=16 -G Ninja -DBUILD_SHARED_LIBS=OFF

#cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DCMAKE_BUILD_TYPE=Release -DPARSEC_GPU_WITH_CUDA=ON -DPARSEC_HAVE_DEV_CUDA_SUPPORT=ON -DPARSEC_GPU_WITH_HIP=OFF -DDPLASMA_PRECISIONS="s;d" -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_C_COMPILER=mpicc -DCMAKE_Fortran_COMPILER=mpif90 -DCMAKE_INSTALL_PREFIX=$PWD/installdir -DCMAKE_C_FLAGS_RELEASE="-O2 -qmaxmem=-1" -DLAPACKE_DIR="${LAPACKE_ROOT}" -DLAPACKE_LIBRARIES="-L ${LAPACKE_ROOT}/lib -llapacke" -DBLA_VENDOR=IBMESSL -DBLAS_LIBRARIES="-L ${OLCF_ESSL_ROOT}/lib64 -lessl" -DON_FUGAKU=OFF -DCMAKE_EXE_LINKER_FLAGS="-L ${CUDA_DIR}/lib64" -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DGSL=ON -DPARSEC_MAX_DEP_OUT_COUNT=16 -G Ninja -DBUILD_SHARED_LIBS=OFF
ninja
#make -j 8

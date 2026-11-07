#/bin/bash

module swap PrgEnv-cray PrgEnv-intel
module load cmake
module load gsl
module load python/2.7.18

module list -l

home=$PWD

echo "update submodules"
#git submodule update --init --recursive

echo "Install starsh"
cd stars-h
rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DMPI=OFF -DSTARPU=OFF -DCUDA=OFF -DOPENMP=OFF -DGSL=ON
make -j 8 install
echo "export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/env_hicma_parsec.sh
export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Install hcore"
cd $home/hcore
rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/installdir
make -j 8 install
echo "export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/env_hicma_parsec.sh
export export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Install hicma_parsec"
cd  $home && mkdir -p build && cd build
srun cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DCMAKE_Fortran_FLAGS="-nofor-main" -DDPLASMA_PRECISIONS="s;d" -DBUILD_SHARED_LIBS=ON -DPARSEC_WITH_HEADER_FILES=ON -DPARSEC_WITH_DEVEL_HEADERS=ON -DCMAKE_CXX_COMPILER=CC -DCMAKE_C_COMPILER=cc -DCMAKE_Fortran_COMPILER=ftn -DPARSEC_ATOMIC_USE_GCC_32_BUILTINS_EXITCODE=0 -DPARSEC_ATOMIC_USE_GCC_64_BUILTINS_EXITCODE=0 -DPARSEC_ATOMIC_USE_GCC_128_BUILTINS_EXITCODE=0 -DPARSEC_GPU_WITH_CUDA=OFF -DPARSEC_HAVE_DEV_CUDA_SUPPORT=OFF -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=Intel10_64lp_seq -DGSL=ON -DGSL_LIBRARY="$GSL_HOME/lib/libgsl.so" -DGSL_CBLAS_LIBRARY="$GSL_HOME/lib/libgslcblas.so" -G Ninja -DPARSEC_GPU_WITH_HIP=OFF -DPARSEC_HAVE_DEV_HIP_SUPPORT=OFF -DPARSEC_MAX_DEP_OUT_COUNT=16
make -j 8

#module load gcc/10.2.0
module load intel/2019
module load cmake/3.21.2
module load nlopt/2.7.0-intel-2019
module load gsl/2.6-intel-2019
module load cuda/11.6
#module load openmpi/4.1.0-gcc-10.2.0
#module load nlopt/2.7.0-gcc-10.2.0
module load mkl/2020.0.166
. /opt/ecrc/mkl/2020.0.166/mkl/bin/mklvars.sh ia32

export dir=$PWD
cd  $dir/exageostat-dev/hicma-x/stars-h
rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/installdir -DMPI=OFF -DSTARPU=OFF -DCUDA=OFF -DOPENMP=OFF -DGSL=ON -DBLAS_LIBRARIES="-Wl,--no-as-needed;-L${MKLROOT}/lib;-lmkl_intel_lp64;-lmkl_core;-lmkl_sequential;-lpthread;-lm;-ldl" -DBLAS_COMPILER_FLAGS="-m64;-I${MKLROOT}/include" -DLAPACK_LIBRARIES="-Wl,--no-as-needed;-L${MKLROOT}/lib;-lmkl_intel_lp64;-lmkl_core;-lmkl_sequential;-lpthread;-lm;-ldl" -DCBLAS_DIR="${MKLROOT}" -DLAPACKE_DIR="${MKLROOT}" -DTMG_DIR="${MKLROOT}"  -DEXAMPLES=OFF -DTESTING=OFF
make -j
make -j && make install
export PKG_CONFIG_PATH=$dir/exageostat-dev/hicma-x/stars-h/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH
export CPATH=$dir/exageostat-dev/hicma-x/stars-h/build/installdir/include:$CPATH

cd  $dir/exageostat-dev/hicma-x/hcore
rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/installdir 
make -j && make install
export PKG_CONFIG_PATH=$dir/exageostat-dev/hicma-x/hcore/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH
export CPATH=$dir/exageostat-dev/hicma-x/hcore/build/installdir/include:$CPATH

module load gcc/10.2.0
module load openmpi/4.1.0-gcc-10.2.0

cd $dir/exageostat-dev/hicma-x
rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc -DCMAKE_Fortran_COMPILER=ifort -DCMAKE_Fortran_FLAGS="-nofor-main" -DCMAKE_INSTALL_PREFIX=`pwd`/installdir  -DDPLASMA_PRECISIONS="s;d" -DBUILD_SHARED_LIBS=ON -DPARSEC_WITH_HEADER_FILES=ON -DPARSEC_WITH_DEVEL_HEADERS=ON  -DPARSEC_ATOMIC_USE_GCC_32_BUILTINS_EXITCODE=0 -DPARSEC_ATOMIC_USE_GCC_64_BUILTINS_EXITCODE=0 -DPARSEC_ATOMIC_USE_GCC_128_BUILTINS_EXITCODE=0 -DPARSEC_GPU_WITH_CUDA=ON -DPARSEC_HAVE_DEV_CUDA_SUPPORT=ON -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=Intel10_64lp_seq -DGSL=ON  -DPARSEC_GPU_WITH_HIP=OFF -DPARSEC_HAVE_DEV_HIP_SUPPORT=OFF -DPARSEC_MAX_DEP_OUT_COUNT=16 
make -j && make install
export PKG_CONFIG_PATH=$dir/exageostat-dev/hicma-x/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$dir/exageostat-dev/hicma-x/build/installdir/lib:$LD_LIBRARY_PATH
export PaRSEC_ROOT=$dir/exageostat-dev/hicma-x/build/installdir


cd $dir/exageostat-dev
rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc -DCMAKE_Fortran_COMPILER=ifort -DCMAKE_Fortran_FLAGS="-nofor-main"  -DEXAGEOSTAT_USE_MPI=OFF -DEXAGEOSTAT_USE_NETCDF=OFF -DEXAGEOSTAT_USE_DPLASMA=OFF -DEXAGEOSTAT_USE_HICMAX=ON -DEXAGEOSTAT_SCHED_QUARK=OFF -DEXAGEOSTAT_SCHED_STARPU=OFF -DEXAGEOSTAT_USE_CHAMELEON=OFF -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DDPLASMA_PRECISIONS="s;d" -DBUILD_SHARED_LIBS=OFF -DPARSEC_GPU_WITH_CUDA=ON -DEXAGEOSTAT_USE_DPLASMA=ON -DPARSEC_HAVE_DEV_CUDA_SUPPORT=ON  -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DPARSEC_GPU_WITH_HIP=OFF -DPARSEC_HAVE_DEV_HIP_SUPPORT=OFF  -DPARSEC_MAX_DEP_OUT_COUNT=16 -DCMAKE_BUILD_TYPE="Release" -DEXAGEOSTAT_SCHED_PARSEC=ON
make -j && make install 



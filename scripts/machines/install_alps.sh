#/bin/bash

# env

ulimit -s 8388608

module purge
module load cray
module load PrgEnv-gnu
module load cudatoolkit

#module load nvidia/23.9

# LAPACKE
export LAPACKE_ROOT=/users/qcao/software/lapack/build/installdir
#export LAPACKE_DIR=$LAPACKE_ROOT
echo $LAPACKE_ROOT
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=$LAPACKE_ROOT/lib:$LD_LIBRARY_PATH
export CPATH=$LAPACKE_ROOT/include:$CPATH

# CMAKE
export PATH=/users/qcao/software/cmake-3.30.1/installdir/bin:$PATH

module list -l


#cd hicma-x-dev
export home=$PWD

echo "update submodules"
#git submodule update --init --recursive

####################################################################################
echo "Install starsh"
cd $home/stars-h && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/installdir -DMPI=OFF -DOPENMP=OFF -DSTARPU=ON -DEXAMPLES=OFF -DTESTING=OFF -DBUILD_SHARED_LIBS=ON -DGSL=OFF -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran
make -j 8 install
echo "export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/env_hicma_parsec.sh
export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Install hcore"
cd $home/hcore && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran
make -j 8 install
echo "export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/env_hicma_parsec.sh
export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Install hicma_parsec"
cd $home && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE="Release" -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DDPLASMA_PRECISIONS="s;d" -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DPARSEC_MAX_DEP_OUT_COUNT=16 -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran -DHICMA_PARSEC_HAVE_CUDA=ON -DCMAKE_EXE_LINKER_FLAGS="-L /opt/nvidia/hpc_sdk/Linux_aarch64/23.9/math_libs/lib64"
make -j 10

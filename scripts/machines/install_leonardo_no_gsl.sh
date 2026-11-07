#/bin/bash

# env

ulimit -s 8388608

module purge

# CUDA
module load cuda

# MPI
module load openmpi

#CMAKE
module load cmake

# MKL on Intel
module load intel-oneapi-mkl

# PYTHON
module load python

module list -l


#cd hicma-x-dev
home=$PWD

echo "update submodules"
#git submodule update --init --recursive

####################################################################################
echo "Install starsh"
cd $home/stars-h && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/installdir -DMPI=OFF -DOPENMP=OFF -DSTARPU=ON -DEXAMPLES=OFF -DTESTING=OFF -DBUILD_SHARED_LIBS=ON -DGSL=OFF
make -j 8 install
echo "export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/env_hicma_parsec.sh
export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Install hcore"
cd $home/hcore && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=`pwd`/installdir 
make -j 8 install
echo "export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/env_hicma_parsec.sh
export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Install hicma_parsec"
cd $home && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE="Release" -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DDPLASMA_PRECISIONS="s;d" -DBUILD_SHARED_LIBS=ON -DPARSEC_GPU_WITH_CUDA=ON -DPARSEC_HAVE_DEV_CUDA_SUPPORT=ON -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -G Ninja -DPARSEC_GPU_WITH_HIP=OFF -DPARSEC_HAVE_DEV_HIP_SUPPORT=OFF  -DPARSEC_MAX_DEP_OUT_COUNT=16
make -j 10

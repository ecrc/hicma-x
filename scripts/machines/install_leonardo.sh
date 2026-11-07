#/bin/bash

# env

ulimit -s 8388608

module purge

module load cuda
module load openmpi
module load cmake
module load ninja
module load intel-oneapi-mkl
module load python
module load gsl

module list -l


#cd hicma-x-dev
home=$PWD

echo "update submodules"
#git submodule update --init --recursive

####################################################################################
echo "Install starsh"
cd $home/stars-h && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/installdir -DMPI=OFF -DOPENMP=OFF -DSTARPU=ON -DEXAMPLES=OFF -DTESTING=OFF -DBUILD_SHARED_LIBS=ON -DGSL=ON
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
cmake .. -DCMAKE_BUILD_TYPE="Release" -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DDPLASMA_PRECISIONS="s;d" -DBUILD_SHARED_LIBS=ON -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -G Ninja -DPARSEC_MAX_DEP_OUT_COUNT=16 -DHICMA_PARSEC_HAVE_CUDA=ON
ninja

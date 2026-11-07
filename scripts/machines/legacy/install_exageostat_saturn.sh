#/bin/bash

module list -l

home=$PWD/hicma-x

echo "update submodules"
#git submodule update --init --recursive

echo "Install starsh"
cd $home/stars-h && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DMPI=OFF -DSTARPU=OFF -DCUDA=OFF -DOPENMP=OFF -DGSL=ON
make -j 8 install
echo "export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/../env_exageostat_parsec.sh
export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Install hcore"
cd $home/hcore && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/installdir
make -j 8 install
echo "export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/../env_exageostat_parsec.sh
export export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Install ExaGeoStat"
cd  $home/.. && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE="Release" -DEXAGEOSTAT_USE_MPI=OFF -DEXAGEOSTAT_USE_NETCDF=OFF -DEXAGEOSTAT_USE_PARSEC=ON -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DDPLASMA_PRECISIONS="s;d" -DBUILD_SHARED_LIBS=ON -DPARSEC_GPU_WITH_CUDA=ON -DPARSEC_HAVE_DEV_CUDA_SUPPORT=ON -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -G Ninja -DPARSEC_GPU_WITH_HIP=OFF -DPARSEC_HAVE_DEV_HIP_SUPPORT=OFF  -DPARSEC_MAX_DEP_OUT_COUNT=16 -DCMAKE_C_COMPILER=mpicc -DEXAGEOSTAT_USE_CUDA=ON
#cmake .. -DCMAKE_BUILD_TYPE="Release" -DEXAGEOSTAT_USE_MPI=OFF -DEXAGEOSTAT_USE_NETCDF=OFF -DEXAGEOSTAT_USE_PARSEC=ON -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DDPLASMA_PRECISIONS="s;d" -DBUILD_SHARED_LIBS=ON -DPARSEC_GPU_WITH_CUDA=ON -DPARSEC_HAVE_DEV_CUDA_SUPPORT=ON -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -G Ninja -DPARSEC_GPU_WITH_HIP=OFF -DPARSEC_HAVE_DEV_HIP_SUPPORT=OFF  -DPARSEC_MAX_DEP_OUT_COUNT=16
#cmake .. -DCMAKE_BUILD_TYPE="Release" -DDPLASMA_PRECISIONS="s;d" -DBUILD_SHARED_LIBS=ON -DPARSEC_GPU_WITH_CUDA=ON -DPARSEC_HAVE_DEV_CUDA_SUPPORT=ON -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -G Ninja -DPARSEC_PROF_TRACE=ON -DPARSEC_PROF_GRAPHER=ON
ninja

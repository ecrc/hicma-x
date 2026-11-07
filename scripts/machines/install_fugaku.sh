#/bin/bash

# env
source /vol0004/apps/oss/spack/share/spack/setup-env.sh
export MODULEPATH=/vol0004/apps/arm-lib/modulefiles:$MODULEPATH
export MODULEPATH=/vol0004/apps/oss/spack/share/spack/modules/linux-rhel8-a64fx:$MODULEPATH

module load gsl
module load ninja
module load cmake

export PLE_MPI_STD_EMPTYFILE=off
export FLIB_SCCR_CNTL=FALSE
export FLIB_PTHREAD=1

ulimit -s 8388608

module list -l

home=$PWD

echo "update submodules"
#git submodule update --init --recursive

cmake ../ -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DDPLASMA_PRECISIONS="s;d" -DCMAKE_C_COMPILER=mpifcc -DCMAKE_CXX_COMPILER=mpiFCC -DCMAKE_Fortran_COMPILER=mpifrt -DBLAS_LIBRARIES="-L $FJSVXTCLANGA/lib64 -lfjlapacksve -SSL2 -SSL2MPI" -DSUPPORT_CXX=OFF -DSUPPORT_FORTRAN=ON -DCMAKE_C_FLAGS="-Nclang -fPIC -v -Ofast" -DBLAS_INCLUDE_DIRS=$FJSVXTCLANGA/include -DCMAKE_BUILD_TYPE="Release" -DCMAKE_C_FLAGS_RELEASE="-DNDEBUG" -DON_FUGAKU=ON -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DHWLOC_DIR=/lib64 -DBUILD_SHARED_LIBS=OFF -DPARSEC_MAX_DEP_OUT_COUNT=16 -G Ninja

ninja

####################################################################################
# On compute node below
# pjsub --interact -L "node=1,freq=2200" -L "rscgrp=int" -L "elapse=1:00:00" --sparam "wait-time=600"

#echo "Install starsh"
#cd $home/stars-h && mkdir -p build && cd build
#cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DMPI=OFF -DSTARPU=OFF -DCUDA=OFF -DOPENMP=OFF -DGSL=ON -DCMAKE_C_COMPILER=fcc -DCMAKE_CXX_COMPILER=FCC -DCMAKE_Fortran_COMPILER=frt -DCMAKE_C_FLAGS="-Nclang -fPIC -v -Ofast" -DBLAS_LIBRARIES="-L $FJSVXTCLANGA/lib64 -lfjlapacksve -SSL2"  -DCBLAS_DIR="$FJSVXTCLANGA/lib64" -DCBLAS_LIBRARIES="-L $FJSVXTCLANGA/lib64 -lfjlapacksve -SSL2"
make -j 8 install
#echo "export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/env_hicma_parsec.sh
#export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

#echo "Install hcore"
#cd $home/hcore && mkdir -p build && cd build
#cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DCMAKE_C_COMPILER=fcc -DCMAKE_CXX_COMPILER=FCC -DCMAKE_Fortran_COMPILER=frt -DCMAKE_C_FLAGS="-Nclang -fPIC -Ofast" -DBLAS_LIBRARIES="-L $FJSVXTCLANGA/lib64 -lfjlapacksve -SSL2"  -DCBLAS_DIR="$FJSVXTCLANGA/lib64" -DCBLAS_INCLUDE_DIRS="$FJSVXTCLANGA/include" -DCBLAS_LIBRARIES="-L $FJSVXTCLANGA/lib64 -lfjlapacksve -SSL2" -DCBLAS_cblas.h_DIRS="$FJSVXTCLANGA/include" -DHCORE_TESTING=OFF
make -j 8 install
#echo "export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/env_hicma_parsec.sh
#export export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

#echo "Install hicma_parsec"
#cd  $home && mkdir -p build && cd build
#cmake ../ -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DDPLASMA_PRECISIONS="s;d" -DCMAKE_C_COMPILER=mpifcc -DCMAKE_CXX_COMPILER=mpiFCC -DCMAKE_Fortran_COMPILER=mpifrt -DBLAS_LIBRARIES="-L $FJSVXTCLANGA/lib64 -lfjlapacksve -SSL2 -SSL2MPI" -DSUPPORT_CXX=OFF -DSUPPORT_FORTRAN=ON -DCMAKE_C_FLAGS="-Nclang -fPIC -v -Ofast" -DBLAS_INCLUDE_DIRS=$FJSVXTCLANGA/include -DCMAKE_BUILD_TYPE="Release" -DCMAKE_C_FLAGS_RELEASE="-DNDEBUG" -DON_FUGAKU=ON -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DHWLOC_DIR=/lib64 -DBUILD_SHARED_LIBS=OFF -DPARSEC_MAX_DEP_OUT_COUNT=16 -G Ninja -DPARSEC_GPU_WITH_HIP=OFF -DPARSEC_GPU_WITH_CUDA=OFF
#ninja

#cmake ../ -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DDPLASMA_PRECISIONS="s;d" -DCMAKE_C_COMPILER=mpifcc -DCMAKE_CXX_COMPILER=mpiFCC -DCMAKE_Fortran_COMPILER=mpifrt -DBLAS_LIBRARIES="-L $FJSVXTCLANGA/lib64 -lfjlapacksve -SSL2 -SSL2MPI" -DSUPPORT_CXX=OFF -DSUPPORT_FORTRAN=ON -DCMAKE_C_FLAGS="-Nclang -fPIC -v -Ofast" -DBLAS_INCLUDE_DIRS=$FJSVXTCLANGA/include -DCMAKE_BUILD_TYPE="Release" -DCMAKE_C_FLAGS_RELEASE="-DNDEBUG" -DON_FUGAKU=ON -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DHWLOC_DIR=/lib64 -DPARSEC_PROF_TRACE=ON -DPAPI_INCLUDE_DIR=/usr/include -DPAPI_LIBRARY=/lib64/libpapi.so.5

#/opt/FJSVxtclanga/tcsds-1.2.35/bin/mpifrt -O3 -DNDEBUG CMakeFiles/testing_dpotrf_tlr.dir/testing_dpotrf.c.o -o testing_dpotrf_tlr   -L/vol0004/share/ra010009/lei/hicma-x-dev-static/stars-h/build/installdir/lib  -L/vol0004/share/ra010009/lei/hicma-x-dev-static/hcore/build/installdir/lib  -L/opt/FJSVxtclanga/tcsds-1.2.35/clang-comp/lib64 -L/opt/FJSVxtclanga/tcsds-1.2.35/lib64 -lfjlapacksve -SSL2 -SSL2MPI libhicma_parsec.a /vol0004/share/ra010009/lei/hicma-x-dev-static/stars-h/build/installdir/lib/libstarsh.a /vol0004/apps/oss/spack-v0.17.0/opt/spack/linux-rhel8-a64fx/fj-4.7.0/gsl-2.7-fm4fuduogrkyoqft5vpbzv7ylz5ejlrc/lib/libgsl.a /vol0004/apps/oss/spack-v0.17.0/opt/spack/linux-rhel8-a64fx/fj-4.7.0/gsl-2.7-fm4fuduogrkyoqft5vpbzv7ylz5ejlrc/lib/libgslcblas.a /vol0004/share/ra010009/lei/hicma-x-dev-static/hcore/build/installdir/lib/libhcore.a ../dplasma/src/libdplasma.a ../dplasma/parsec/parsec/libparsec.a -ldl /usr/lib64/libhwloc.so /lib64/libpapi.so.5 -L/opt/FJSVxtclanga/tcsds-1.2.35/lib64 -lfjlapacksve -SSL2 -SSL2MPI -lm -lstdc++ -lfjprofmpi -lmpi_cxx -lfjcrt -lz

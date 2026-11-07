# Builds the Cholesky driver against an *installed* HiCMA-X, from outside the
# source tree, to check that the installed headers and the exported target are
# usable:
#
#   cmake -S tests/install_consumer -B build-consumer -DCMAKE_PREFIX_PATH=<prefix>
#   cmake --build build-consumer
#   ./build-consumer/hicma_consumer --N 512 --NB 128 --kind_of_problem 2 --check 1
cmake_minimum_required(VERSION 3.21)
project(hicma_consumer LANGUAGES C CXX Fortran)

find_package(HICMAParsec REQUIRED)

add_executable(hicma_consumer ${CMAKE_CURRENT_SOURCE_DIR}/../testing_potrf.c)
target_link_libraries(hicma_consumer PRIVATE HICMAParsec::hicma_parsec)
if(CMAKE_Fortran_COMPILER_WORKS)
    set_target_properties(hicma_consumer PROPERTIES LINKER_LANGUAGE Fortran)
endif()

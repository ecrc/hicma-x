# @copyright (c) 2023-2025     Saint Louis University (SLU)
# @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
# Copyright (c) 2023-2025     Nvidia Corporation
# @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
# @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
#                              All rights reserved.

# A driver linked against the library (which carries BLAS/LAPACKE, MPI and the
# GPU libraries). HICMA_PARSEC_BUILDING gives the in-tree drivers the same view
# of the headers as the library; installed consumers see the public subset only.
function(hicma_add_driver name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE hicma_parsec)
    target_compile_definitions(${name} PRIVATE HICMA_PARSEC_BUILDING)
    if(CMAKE_Fortran_COMPILER_WORKS)
        set_target_properties(${name} PROPERTIES LINKER_LANGUAGE Fortran)
    endif()
endfunction()

set(HICMA_GPU HICMA_PARSEC_HAVE_CUDA OR HICMA_PARSEC_HAVE_HIP)

# ---- Cholesky and TRMM drivers ----
hicma_add_driver(testing_potrf_tlr testing_potrf.c)
hicma_add_driver(testing_trmm_mp testing_trmm.c $<TARGET_OBJECTS:dplasma_tests_common>)
hicma_add_driver(testing_trmm_performance testing_trmm_performance.c $<TARGET_OBJECTS:dplasma_tests_common>)
hicma_add_driver(testing_trmm_performance_static testing_trmm_performance_static.c $<TARGET_OBJECTS:dplasma_tests_common>)
hicma_add_driver(testing_dgemm_tlr testing_dgemm_tlr.c)
target_compile_definitions(testing_dgemm_tlr PRIVATE HICMA_PARSEC_HAVE_TLR=1)
hicma_add_driver(testing_climate_emulator testing_climate_emulator.c)
hicma_add_driver(testing_dist_read testing_dist_read.c)

# CBLAS micro-benchmark (no HiCMA)
add_executable(testing_dgemm_cpu testing_dgemm.c)
target_link_libraries(testing_dgemm_cpu PRIVATE ${LD_LIB_HICMA_PARSEC})

# ---- Genomics (kernel ridge regression): GPU-only GEMM kernels, GENOMICS=1 ----
if(HICMA_PARSEC_GENOMICS)
    hicma_add_driver(testing_syrk testing_syrk.c)
    hicma_add_driver(testing_Kmatrix testing_Kmatrix.c)
    hicma_add_driver(testing_KRR testing_KRR.c)
    hicma_add_driver(testing_KRR_file testing_KRR_file.c)
    hicma_add_driver(testing_igemm_summa testing_igemm_summa.c)
endif()

# ---- GPU-only drivers and micro-benchmarks ----
if(HICMA_PARSEC_HAVE_CUDA OR HICMA_PARSEC_HAVE_HIP)
    hicma_add_driver(testing_hamming testing_hamming.c)
    if(ENABLE_CUTLASS)
        target_link_directories(testing_hamming PRIVATE ${CUTLASS_ROOT}/lib64)
        target_link_libraries(testing_hamming PRIVATE cutlass)
    endif()
endif()

if(HICMA_PARSEC_HAVE_CUDA)
    foreach(t testing_dgemv_gpu:testing_dgemv.cu testing_dgemm_gpu:testing_dgemm.cu
              testing_gemm_gpu:testing_gemm_gpu.cu testing_gpu_bandwidth:testing_gpu_bandwidth.cpp)
        string(REPLACE ":" ";" t ${t})
        list(GET t 0 name)
        list(GET t 1 src)
        add_executable(${name} ${src})
        target_link_libraries(${name} ${LD_LIB_CUDA_TARGETS})
    endforeach()
endif()

if(HICMA_PARSEC_HAVE_HIP)
    foreach(t testing_dgemm_gpu_hip:testing_dgemm_hip.cu testing_gemm_gpu_hip:testing_gemm_gpu_hip.cu
              testing_gemm_gpu_hip_dp:testing_gemm_gpu_hip_dp.cu)
        string(REPLACE ":" ";" t ${t})
        list(GET t 0 name)
        list(GET t 1 src)
        set_source_files_properties(${src} PROPERTIES LANGUAGE HIP COMPILE_FLAGS "-x hip")
        add_executable(${name} ${src})
        target_link_libraries(${name} ${LD_LIB_HIP})
        add_dependencies(${name} parsec)
    endforeach()
endif()

if(ON_FUGAKU)
    add_executable(testing_hgemm_cpu testing_hgemm.c)
    target_link_libraries(testing_hgemm_cpu PRIVATE ${LD_LIB_HICMA_PARSEC})
    set_target_properties(testing_hgemm_cpu PROPERTIES LINK_FLAGS "-SSL2")
endif()

# ---- CTest: every case runs with its correctness check and exits non-zero on
# a wrong result. Labels: hicma (all), mpi (needs MPIEXEC_EXECUTABLE), gpu.
# scripts/test_CI.sh runs the same cases outside CTest. ----
set(HICMA_TEST_LABELS hicma)
function(hicma_add_test name nproc)
    set(labels hicma)
    set(launch)
    if(nproc GREATER 1)
        if(NOT MPIEXEC_EXECUTABLE)
            return()
        endif()
        set(launch ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nproc} ${MPIEXEC_PREFLAGS})
        list(APPEND labels mpi)
    endif()
    if(HICMA_TEST_GPU)
        list(APPEND labels gpu)
    endif()
    add_test(NAME ${name} COMMAND ${launch} ${ARGN})
    set_tests_properties(${name} PROPERTIES TIMEOUT 900 LABELS "${labels}")
endfunction()

set(P $<TARGET_FILE:testing_potrf_tlr>)
set(T $<TARGET_FILE:testing_trmm_mp>)
set(COMMON --kind_of_problem 2 --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --check 1)

hicma_add_test(potrf_dense_fp64        1 ${P} ${COMMON})
hicma_add_test(potrf_tlr_dp_band1      1 ${P} ${COMMON} --kind_of_cholesky 2 --band_dense 1)
hicma_add_test(potrf_tlr_dp_band2      1 ${P} ${COMMON} --kind_of_cholesky 2 --band_dense 2)
hicma_add_test(potrf_tlr_mp_band1      1 ${P} ${COMMON} --kind_of_cholesky 1 --band_dense 1 --adaptive_decision 0)
hicma_add_test(potrf_tlr_mp_adaptive   1 ${P} ${COMMON} --kind_of_cholesky 1 --band_dense 1 --adaptive_decision 1 --check_tol 1e-7)
hicma_add_test(potrf_mp_dense_adaptive 1 ${P} ${COMMON} --kind_of_cholesky 1 --band_dense 1000 --adaptive_decision 1)
hicma_add_test(potrf_mp_dense_admem    1 ${P} ${COMMON} --kind_of_cholesky 1 --band_dense 1000 --adaptive_decision 1 --adaptive_memory 1)
hicma_add_test(potrf_band_dp_sp        1 ${P} ${COMMON} --kind_of_cholesky 3 --band_dense_dp 2 --band_dense_sp 4 --check_tol 1e-6)
hicma_add_test(potrf_runtime_decision  1 ${P} ${COMMON} --adaptive_decision_runtime 1 --check_tol 1e-7)
hicma_add_test(potrf_sparse_general    1 ${P} ${COMMON} --sparse 1)
hicma_add_test(potrf_sparse_balance    1 ${P} ${COMMON} --sparse 2)
hicma_add_test(trmm_fp64               1 ${T} --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 0)
hicma_add_test(trmm_adaptive           1 ${T} --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 1 --fixedacc 1e-4)
hicma_add_test(trmm_adaptive_sr        1 ${T} --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 1 --fixedacc 1e-4 --enable_stochastic_rounding 1)
# two ranks
hicma_add_test(potrf_dense_fp64_np2    2 ${P} ${COMMON})
hicma_add_test(potrf_tlr_dp_band2_np2  2 ${P} ${COMMON} --kind_of_cholesky 2 --band_dense 2)
hicma_add_test(potrf_runtime_np2       2 ${P} ${COMMON} --adaptive_decision_runtime 1 --check_tol 1e-7)
hicma_add_test(potrf_sparse_np2        2 ${P} ${COMMON} --sparse 1)
hicma_add_test(trmm_adaptive_np2       2 ${T} --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 1 --fixedacc 1e-4)
# the virus mesh shipped with stars-h
set(HICMA_MESH ${PROJECT_SOURCE_DIR}/stars-h/SARS-CoV-2-meshes/singleviursdata/SortVirus10370.txt)
if(EXISTS ${HICMA_MESH})
    hicma_add_test(potrf_mesh_sparse   1 ${P} --N 10370 --NB 1037 --fixedacc 1e-4 --maxrank 50 --adddiag 0.1 --kind_of_problem 6
        --band_dense 1 --send_full_tile 0 --lookahead 1 --auto_band 0 --mesh_file ${HICMA_MESH} --numobj 1 --rbf_kernel 0
        --radius 4.63e-04 --order 2 --density -1 --sparse 1 --check 1)
endif()
# GPU (one device)
if(HICMA_PARSEC_HAVE_CUDA OR HICMA_PARSEC_HAVE_HIP)
    set(HICMA_TEST_GPU ON)
    set(G --kind_of_problem 2 --N 8192 --NB 1024 --gpus 1 --check 1)
    hicma_add_test(gpu_dense_fp64        1 ${P} ${G})
    hicma_add_test(gpu_dense_adaptive    1 ${P} ${G} --adaptive_decision 1)
    hicma_add_test(gpu_dense_band2       1 ${P} ${G} --band_dense 2 --maxrank 512)
    hicma_add_test(gpu_fp8_static        1 ${P} ${G} --kind_of_cholesky 8 --adaptive_decision 1)
    hicma_add_test(gpu_fp8_sp            1 ${P} ${G} --kind_of_cholesky 9 --adaptive_decision 1 --check_tol 1e-6)
    hicma_add_test(gpu_runtime_decision  1 ${P} ${G} --adaptive_decision_runtime 1 --check_tol 1e-7)
    hicma_add_test(gpu_runtime_admem     1 ${P} ${G} --adaptive_decision_runtime 1 --adaptive_memory 1 --check_tol 1e-7)
    hicma_add_test(gpu_trmm_adaptive     1 ${T} --N 2000 --K 1500 --NB 256 --gpus 1 --check 1 --adaptive_decision 1 --fixedacc 1e-4)
    if(HICMA_PARSEC_HAVE_CUDA)
        # kinds 1 and 2 have CUDA bodies only
        hicma_add_test(gpu_tlr_mp        1 ${P} ${COMMON} --gpus 1 --kind_of_cholesky 1 --band_dense 1000 --adaptive_decision 1)
        hicma_add_test(gpu_tlr_dp_band2  1 ${P} ${COMMON} --gpus 1 --kind_of_cholesky 2 --band_dense 2)
    endif()
endif()

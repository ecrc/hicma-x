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
hicma_add_driver(testing_lapack_tile testing_lapack_tile.c)
hicma_add_driver(testing_matrix_sum testing_matrix_sum.c)

# CBLAS micro-benchmark (no HiCMA)
add_executable(testing_dgemm_cpu testing_dgemm.c)
target_link_libraries(testing_dgemm_cpu PRIVATE LAPACKE::LAPACKE)

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
    target_link_libraries(testing_hgemm_cpu PRIVATE LAPACKE::LAPACKE)
    set_target_properties(testing_hgemm_cpu PROPERTIES LINK_FLAGS "-SSL2")
endif()

# ---- CTest: every case runs with its correctness check and exits non-zero on
# a wrong result. Labels: hicma (all), mpi (needs MPIEXEC_EXECUTABLE), gpu.
# scripts/test_CI.sh runs the same cases outside CTest. ----
set(HICMA_TEST_LABELS hicma)
function(hicma_add_test name nproc)
    # testing_potrf refuses to run when the library is built with GENOMICS=1
    list(GET ARGN 0 exe)
    if(HICMA_PARSEC_GENOMICS AND "${exe}" STREQUAL "${P}")
        return()
    endif()
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
    # Every case calls the BLAS from several PaRSEC worker threads at once.
    # DPLASMA's dplasma_blas_threadsafety checks that the BLAS survives that;
    # as a fixture, ctest runs it first (also with -L hicma) and skips these
    # cases if it fails, instead of reporting wrong residuals everywhere.
    set_tests_properties(${name} PROPERTIES TIMEOUT 900 LABELS "${labels}"
                         FIXTURES_REQUIRED DPLASMA_BLAS_THREADSAFE)
endfunction()

set(P $<TARGET_FILE:testing_potrf_tlr>)
set(T $<TARGET_FILE:testing_trmm_mp>)
set(COMMON --kind_of_problem 2 --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --check 1)

hicma_add_test(potrf_dense_fp64        1 ${P} ${COMMON})
hicma_add_test(potrf_tlr_dp_band1      1 ${P} ${COMMON} --kind_of_cholesky 2 --band_dense 1)
hicma_add_test(potrf_tlr_dp_band2      1 ${P} ${COMMON} --kind_of_cholesky 2 --band_dense 2)
hicma_add_test(potrf_tlr_mp_band1      1 ${P} ${COMMON} --kind_of_cholesky 1 --band_dense 1 --adaptive_decision 0)
hicma_add_test(potrf_tlr_mp_adaptive   1 ${P} ${COMMON} --kind_of_cholesky 1 --band_dense 1 --adaptive_decision 1 --check_tol 1e-7)
# Mixed precision (about half of the GEMMs in FP32): the residual is bounded by
# FP32 arithmetic, 7e-9 to 1.1e-8 depending on the BLAS library, not by the
# 1e-8 compression accuracy; the bound is the one of potrf_tlr_mp_adaptive.
hicma_add_test(potrf_mp_dense_adaptive 1 ${P} ${COMMON} --kind_of_cholesky 1 --band_dense 1000 --adaptive_decision 1 --check_tol 1e-7)
hicma_add_test(potrf_mp_dense_admem    1 ${P} ${COMMON} --kind_of_cholesky 1 --band_dense 1000 --adaptive_decision 1 --adaptive_memory 1 --check_tol 1e-7)
hicma_add_test(potrf_band_dp_sp        1 ${P} ${COMMON} --kind_of_cholesky 3 --band_dense_dp 2 --band_dense_sp 4 --check_tol 1e-6)
hicma_add_test(potrf_band_dp_sp_sender 1 ${P} ${COMMON} --kind_of_cholesky 3 --band_dense_dp 2 --band_dense_sp 4 --datatype_convert 1 --check_tol 1e-6)
hicma_add_test(potrf_runtime_decision  1 ${P} ${COMMON} --adaptive_decision_runtime 1 --check_tol 1e-7)
hicma_add_test(potrf_sparse_general    1 ${P} ${COMMON} --sparse 1)
hicma_add_test(potrf_sparse_balance    1 ${P} ${COMMON} --sparse 2)
# band auto-tuning on a 3-D problem (kind 4)
hicma_add_test(potrf_autoband_3d       1 ${P} --N 5400 --NB 540 --fixedacc 1e-8 --maxrank 540 --kind_of_problem 4 --auto_band 1 --check 1
               -- -mca runtime_comm_coll_bcast 0)
hicma_add_test(trmm_fp64               1 ${T} --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 0)
hicma_add_test(trmm_adaptive           1 ${T} --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 1 --fixedacc 1e-4)
hicma_add_test(trmm_adaptive_sr        1 ${T} --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 1 --fixedacc 1e-4 --enable_stochastic_rounding 1)
hicma_add_test(lapack_tile_convert     1 $<TARGET_FILE:testing_lapack_tile>)
hicma_add_test(matrix_sum              1 $<TARGET_FILE:testing_matrix_sum>)
# --tensor_gemm with two FP16 GEMM variants is refused at start-up (it gave
# wrong results in kind 3 and aborted mid-run in kind 5)
hicma_add_test(tensor_gemm_invalid     1 ${P} ${COMMON} --tensor_gemm 0x18)
if(TEST tensor_gemm_invalid)
    set_tests_properties(tensor_gemm_invalid PROPERTIES PASS_REGULAR_EXPRESSION "give exactly one FP16 GEMM variant")
endif()
# two ranks
hicma_add_test(potrf_dense_fp64_np2    2 ${P} ${COMMON})
hicma_add_test(potrf_tlr_dp_band2_np2  2 ${P} ${COMMON} --kind_of_cholesky 2 --band_dense 2)
hicma_add_test(potrf_runtime_np2       2 ${P} ${COMMON} --adaptive_decision_runtime 1 --check_tol 1e-7)
hicma_add_test(potrf_sparse_np2        2 ${P} ${COMMON} --sparse 1)
hicma_add_test(trmm_adaptive_np2       2 ${T} --N 2000 --K 1500 --NB 256 --check 1 --adaptive_decision 1 --fixedacc 1e-4)
hicma_add_test(lapack_tile_convert_np2 2 $<TARGET_FILE:testing_lapack_tile>)
hicma_add_test(matrix_sum_np2          2 $<TARGET_FILE:testing_matrix_sum>)
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
    # AMD gfx90a flushes FP16 subnormals in its FP16 GEMMs, which made the
    # adaptive FP16 tiles miss --fixedacc now and then (1.9e-6, 2.8e-5 for 1e-8):
    # on HIP their GEMMs run in BF16 (--tensor_gemm_hp 0x20, dense layout)
    set(HP_VARIANT)
    if(HICMA_PARSEC_HAVE_HIP)
        set(HP_VARIANT --tensor_gemm_hp 0x20)
    endif()
    hicma_add_test(gpu_dense_fp64        1 ${P} ${G})
    hicma_add_test(gpu_dense_adaptive    1 ${P} ${G} --adaptive_decision 1 ${HP_VARIANT})
    hicma_add_test(gpu_dense_band2       1 ${P} ${G} --band_dense 2 --maxrank 512)
    hicma_add_test(gpu_fp8_static        1 ${P} ${G} --kind_of_cholesky 8 --adaptive_decision 1 ${HP_VARIANT})
    hicma_add_test(gpu_fp8_sp            1 ${P} ${G} --kind_of_cholesky 9 --adaptive_decision 1 --check_tol 1e-6 ${HP_VARIANT})
    hicma_add_test(gpu_runtime_decision  1 ${P} ${G} --adaptive_decision_runtime 1 --check_tol 1e-7)
    hicma_add_test(gpu_runtime_admem     1 ${P} ${G} --adaptive_decision_runtime 1 --adaptive_memory 1 --check_tol 1e-7)
    hicma_add_test(gpu_trmm_adaptive     1 ${T} --N 2000 --K 1500 --NB 256 --gpus 1 --check 1 --adaptive_decision 1 --fixedacc 1e-4)
    if(HICMA_PARSEC_HAVE_CUDA)
        # kinds 1 and 2 have CUDA bodies only
        hicma_add_test(gpu_tlr_mp        1 ${P} ${COMMON} --gpus 1 --kind_of_cholesky 1 --band_dense 1000 --adaptive_decision 1 --check_tol 1e-7)
        hicma_add_test(gpu_tlr_dp_band2  1 ${P} ${COMMON} --gpus 1 --kind_of_cholesky 2 --band_dense 2)
        # kinds 3 and 4 with a half-precision band (GPU only), and kind 3 with
        # the conversion done by the sender; residuals 8.5e-7 (HP band) and 5e-8
        hicma_add_test(gpu_band_dp_sp_hp     1 ${P} ${G} --kind_of_cholesky 3 --band_dense_dp 2 --band_dense_sp 4 --check_tol 1e-5)
        hicma_add_test(gpu_band_dp_sp_hp_sender 1 ${P} ${G} --kind_of_cholesky 3 --band_dense_dp 2 --band_dense_sp 4 --datatype_convert 1 --check_tol 1e-5)
        hicma_add_test(gpu_band_sp_hp        1 ${P} ${G} --kind_of_cholesky 4 --band_dense_sp 4 --check_tol 1e-5)
        # FP16 accumulation (--tensor_gemm 0x10): residual 2.3e-5
        hicma_add_test(gpu_band_hp_fp16acc   1 ${P} ${G} --kind_of_cholesky 3 --band_dense_dp 2 --band_dense_sp 4 --tensor_gemm 0x10 --check_tol 1e-4)
        # Stochastic rounding in the GPU Cholesky (kind 5; kinds 8 to 10 refuse it)
        hicma_add_test(gpu_dense_adaptive_sr 1 ${P} ${G} --adaptive_decision 1 --enable_stochastic_rounding 1)
        # kind 1 version of gpu_tlr_hp_band (static FP16 band with low-rank tiles)
        hicma_add_test(gpu_tlr_mp_hp_band    1 ${P} --kind_of_problem 2 --N 8192 --NB 512 --gpus 1 --check 1 --kind_of_cholesky 1
                       --band_dense 4 --band_dense_dp 1 --band_dense_sp 2 --fixedacc 1e-4 --maxrank 256 --check_tol 3e-4)
    endif()
    # kinds 5 and 8, CUDA and HIP (on HIP --tensor_gemm_sp 0x4 runs in FP32)
    # GEMM variants of the dense GEMMs (kind 5), through --tensor_gemm_hp,
    # --tensor_gemm_sp and the combined --tensor_gemm: BF16 operands with
    # FP32 C (residual 6.7e-6) and with BF16 C (1.0e-4); TF32 for the FP32
    # band (5.1e-6, against 5.1e-8 in FP32)
    hicma_add_test(gpu_dense_bf16        1 ${P} ${G} --band_dense_dp 2 --band_dense_sp 4 --band_dense_hp 8 --tensor_gemm_hp 0x20 --check_tol 1e-4)
    hicma_add_test(gpu_dense_bf16_c16    1 ${P} ${G} --band_dense_dp 2 --band_dense_sp 4 --band_dense_hp 8 --tensor_gemm 0x40 --check_tol 1e-3)
    hicma_add_test(gpu_dense_tf32        1 ${P} ${G} --band_dense_dp 2 --band_dense_sp 8 --tensor_gemm_sp 0x4 --check_tol 1e-4)
    # Adaptive TLR with FP16 C (0x10): the dense and the low-rank GEMMs of a
    # tile disagreed on its type (NaNs, then a crash); residual 2.2e-5
    hicma_add_test(gpu_tlr_adaptive_fp16acc 1 ${P} --kind_of_problem 2 --N 8192 --NB 512 --gpus 1 --check 1 --band_dense 4
                   --fixedacc 1e-4 --maxrank 256 --adaptive_decision 1 --tensor_gemm_hp 0x10 --check_tol 1e-4)
    # Static FP16 band with low-rank tiles, so FP16 tiles get low-rank x
    # low-rank updates (their low-rank B was converted as a dense tile:
    # residual 22 to 59, or a segfault); residuals 4.3e-5 (kinds 5 and 1)
    hicma_add_test(gpu_tlr_hp_band       1 ${P} --kind_of_problem 2 --N 8192 --NB 512 --gpus 1 --check 1 --band_dense 4
                   --band_dense_dp 1 --band_dense_sp 2 --fixedacc 1e-4 --maxrank 256 --check_tol 3e-4)
    # Kind 8 with an FP8 band (FP16 where the GPUs lack FP8) and low-rank
    # tiles: FP8 operands, stored as FP32, were read as FP64 by the
    # low-rank GEMMs (residual 1.5e-2 on an H100, 1.6e-5 fixed). --adddiag
    # keeps the matrix positive definite under FP8 so close to the diagonal
    hicma_add_test(gpu_tlr_fp8_band      1 ${P} --kind_of_problem 2 --N 8192 --NB 512 --gpus 1 --check 1 --kind_of_cholesky 8
                   --band_dense 6 --band_dense_dp 1 --band_dense_sp 2 --band_dense_hp 3 --adddiag 100 --fixedacc 1e-4 --maxrank 256
                   --check_tol 3e-4)
endif()

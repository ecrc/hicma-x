# @copyright (c) 2023-2025     Saint Louis University (SLU)
# @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
# Copyright (c) 2023-2025     Nvidia Corporation
# @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
# @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
#                              All rights reserved.

# Executable 
if( HICMA_PARSEC_HAVE_CUDA )
    add_executable(testing_dgemv_gpu testing_dgemv.cu)
    target_link_libraries(testing_dgemv_gpu ${LD_LIB_CUDA_TARGETS})

    add_executable(testing_dgemm_gpu testing_dgemm.cu)
    target_link_libraries(testing_dgemm_gpu ${LD_LIB_CUDA_TARGETS})

    add_executable(testing_gemm_gpu testing_gemm_gpu.cu)
    target_link_libraries(testing_gemm_gpu ${LD_LIB_CUDA_TARGETS})
   
    # Executable testing_gpu_bandwidth
    add_executable(testing_gpu_bandwidth testing_gpu_bandwidth.cpp)
    target_link_libraries(testing_gpu_bandwidth ${LD_LIB_CUDA_TARGETS})
endif( HICMA_PARSEC_HAVE_CUDA )

# Executable 
if( HICMA_PARSEC_HAVE_HIP )
    set_source_files_properties( testing_dgemm_hip.cu PROPERTIES LANGUAGE HIP)
    set_source_files_properties( testing_dgemm_hip.cu PROPERTIES COMPILE_FLAGS "-x hip")
    add_executable(testing_dgemm_gpu_hip testing_dgemm_hip.cu)
    target_link_libraries(testing_dgemm_gpu_hip ${LD_LIB_HIP})
    add_dependencies(testing_dgemm_gpu_hip parsec)

    set_source_files_properties( testing_gemm_gpu_hip.cu PROPERTIES LANGUAGE HIP)
    set_source_files_properties( testing_gemm_gpu_hip.cu PROPERTIES COMPILE_FLAGS "-x hip")
    add_executable(testing_gemm_gpu_hip testing_gemm_gpu_hip.cu)
    target_link_libraries(testing_gemm_gpu_hip ${LD_LIB_HIP})
    add_dependencies(testing_gemm_gpu_hip parsec)
    
    # Executable testing_gemm_gpu_hip_dp
    set_source_files_properties( testing_gemm_gpu_hip_dp.cu PROPERTIES LANGUAGE HIP)
    set_source_files_properties( testing_gemm_gpu_hip_dp.cu PROPERTIES COMPILE_FLAGS "-x hip")
    add_executable(testing_gemm_gpu_hip_dp testing_gemm_gpu_hip_dp.cu)
    target_link_libraries(testing_gemm_gpu_hip_dp ${LD_LIB_HIP})
    add_dependencies(testing_gemm_gpu_hip_dp parsec)
endif( HICMA_PARSEC_HAVE_HIP )

add_executable(testing_potrf_tlr testing_potrf.c)
target_link_libraries(testing_potrf_tlr PRIVATE ${LD_LIB_HICMA_PARSEC} hicma_parsec ${HCORE_STARSH_LIB})

if( CMAKE_Fortran_COMPILER_WORKS )
    set_target_properties(testing_potrf_tlr PROPERTIES
                          LINKER_LANGUAGE Fortran)
endif()

add_executable(testing_syrk testing_syrk.c)
target_link_libraries(testing_syrk PRIVATE ${LD_LIB_HICMA_PARSEC} hicma_parsec ${HCORE_STARSH_LIB})

if( CMAKE_Fortran_COMPILER_WORKS )
    set_target_properties(testing_syrk PROPERTIES
                          LINKER_LANGUAGE Fortran)
endif()

add_executable(testing_Kmatrix testing_Kmatrix.c)
target_link_libraries(testing_Kmatrix PRIVATE ${LD_LIB_HICMA_PARSEC} hicma_parsec ${HCORE_STARSH_LIB})

if( CMAKE_Fortran_COMPILER_WORKS )
    set_target_properties(testing_Kmatrix PROPERTIES
                          LINKER_LANGUAGE Fortran)
endif()

if( HICMA_PARSEC_HAVE_CUDA OR HICMA_PARSEC_HAVE_HIP )
    add_executable(testing_hamming testing_hamming.c)
    set_property(TARGET testing_hamming PROPERTY LINKER_LANGUAGE CXX)
    if(ENABLE_CUTLASS)
        # Use target_link_directories instead of hardcoded -L flag
        target_link_directories(testing_hamming PRIVATE ${CUTLASS_ROOT}/lib64)
        target_link_libraries(testing_hamming PRIVATE ${LD_LIB_HICMA_PARSEC} hicma_parsec ${HCORE_STARSH_LIB} cutlass)
    else(ENABLE_CUTLASS)
        target_link_libraries(testing_hamming PRIVATE ${LD_LIB_HICMA_PARSEC} hicma_parsec ${HCORE_STARSH_LIB} ${LD_LIB_CUDA_TARGETS})
    endif(ENABLE_CUTLASS)

    if( CMAKE_Fortran_COMPILER_WORKS )
        set_target_properties(testing_hamming PROPERTIES
            LINKER_LANGUAGE Fortran)
    endif()
endif()

add_executable(testing_igemm_summa testing_igemm_summa.c)
target_link_libraries(testing_igemm_summa PRIVATE ${LD_LIB_HICMA_PARSEC} hicma_parsec ${HCORE_STARSH_LIB})

if( CMAKE_Fortran_COMPILER_WORKS )
    set_target_properties(testing_igemm_summa PROPERTIES
                          LINKER_LANGUAGE Fortran)
endif()


add_executable(testing_dist_read testing_dist_read.c)
target_link_libraries(testing_dist_read PRIVATE ${LD_LIB_HICMA_PARSEC} hicma_parsec ${HCORE_STARSH_LIB})

if( CMAKE_Fortran_COMPILER_WORKS )
    set_target_properties(testing_dist_read PROPERTIES
                          LINKER_LANGUAGE Fortran)
endif()


# Executable testing_dgemm_cpu 
add_executable(testing_dgemm_cpu testing_dgemm.c)
target_link_libraries(testing_dgemm_cpu PRIVATE ${LD_LIB_HICMA_PARSEC})

# Executable testing_dgemm_tlr (TLR version) - Protected with HCORE availability
if( HCORE_STARSH_LIB )
    add_executable(testing_dgemm_tlr testing_dgemm_tlr.c)
    target_link_libraries(testing_dgemm_tlr PRIVATE ${LD_LIB_HICMA_PARSEC} hicma_parsec ${HCORE_STARSH_LIB})
    
    if( CMAKE_Fortran_COMPILER_WORKS )
        set_target_properties(testing_dgemm_tlr PROPERTIES
                              LINKER_LANGUAGE Fortran)
    endif()
    
    # Add TLR-specific compile definitions
    target_compile_definitions(testing_dgemm_tlr PRIVATE HICMA_PARSEC_HAVE_TLR=1)
    
    message(STATUS "HCORE support enabled - Building TLR DGEMM executable")
else( HCORE_STARSH_LIB )
    message(STATUS "HCORE support disabled - Skipping TLR DGEMM executable")
endif( HCORE_STARSH_LIB )

add_executable(testing_KRR testing_KRR.c)
target_link_libraries(testing_KRR PRIVATE ${LD_LIB_HICMA_PARSEC} hicma_parsec ${HCORE_STARSH_LIB})

if( CMAKE_Fortran_COMPILER_WORKS )
    set_target_properties(testing_KRR PROPERTIES
                          LINKER_LANGUAGE Fortran)
endif()

add_executable(testing_KRR_file testing_KRR_file.c)
target_link_libraries(testing_KRR_file PRIVATE ${LD_LIB_HICMA_PARSEC} hicma_parsec ${HCORE_STARSH_LIB})

if( CMAKE_Fortran_COMPILER_WORKS )
    set_target_properties(testing_KRR_file PROPERTIES
                          LINKER_LANGUAGE Fortran)
endif()

# Executable testing_hgemm_cpu 
if( ON_FUGAKU )
    add_executable(testing_hgemm_cpu testing_hgemm.c)
    target_link_libraries(testing_hgemm_cpu PRIVATE ${LD_LIB_HICMA_PARSEC})
    set_target_properties( testing_hgemm_cpu PROPERTIES LINK_FLAGS "-SSL2" )
endif()


# climate_emulator
add_executable(testing_climate_emulator testing_climate_emulator.c)
target_link_libraries(testing_climate_emulator PRIVATE ${LD_LIB_HICMA_PARSEC} hicma_parsec ${HCORE_STARSH_LIB})

if( CMAKE_Fortran_COMPILER_WORKS )
    set_target_properties(testing_climate_emulator PROPERTIES
                          LINKER_LANGUAGE Fortran)
endif()


# TRMM
add_executable(testing_trmm_mp testing_trmm.c)
target_link_libraries(testing_trmm_mp PRIVATE ${LD_LIB_HICMA_PARSEC} hicma_parsec ${HCORE_STARSH_LIB})
target_sources(testing_trmm_mp PRIVATE $<TARGET_OBJECTS:dplasma_tests_common>)

if( CMAKE_Fortran_COMPILER_WORKS )
    set_target_properties(testing_trmm_mp PROPERTIES
                          LINKER_LANGUAGE Fortran)
endif()

add_executable(testing_trmm_performance testing_trmm_performance.c)
target_link_libraries(testing_trmm_performance PRIVATE ${LD_LIB_HICMA_PARSEC} hicma_parsec ${HCORE_STARSH_LIB})
target_sources(testing_trmm_performance PRIVATE $<TARGET_OBJECTS:dplasma_tests_common>)

if( CMAKE_Fortran_COMPILER_WORKS )
    set_target_properties(testing_trmm_performance PROPERTIES
                          LINKER_LANGUAGE Fortran)
endif()

add_executable(testing_trmm_performance_static testing_trmm_performance_static.c)
target_link_libraries(testing_trmm_performance_static PRIVATE ${LD_LIB_HICMA_PARSEC} hicma_parsec ${HCORE_STARSH_LIB})
target_sources(testing_trmm_performance_static PRIVATE $<TARGET_OBJECTS:dplasma_tests_common>)

if( CMAKE_Fortran_COMPILER_WORKS )
    set_target_properties(testing_trmm_performance_static PROPERTIES
                          LINKER_LANGUAGE Fortran)
endif()

# Header and library installation is handled by the top-level CMakeLists.txt

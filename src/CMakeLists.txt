# @copyright (c) 2023-2025     Saint Louis University (SLU)
# @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
# Copyright (c) 2023-2025     Nvidia Corporation
# @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
# @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
#                              All rights reserved.

# Source files for the main library
set(SRC
    # Core HICMA PARSEC files
    core/hicma_parsec.c
    core/hicma_parsec_internal.c
    core/hicma_parsec_core.c
    core/hicma_parsec_gpu.c
    kernels/hicma_kernel_time.c
    
    # Wrapper files
    kernels/potrf_L_dense_tlr_dp_wrapper.c
    kernels/potrf_L_sparse_tlr_dp_general_wrapper.c
    kernels/potrf_L_sparse_tlr_dp_balance_wrapper.c
    kernels/potrf_L_dense_tlr_mp_wrapper.c
    kernels/potrf_L_dense_mp_gpu_wrapper.c
    kernels/potrf_L_dense_sp_hp_band_wrapper.c
    kernels/potrf_L_dense_mp_band_wrapper.c
    kernels/potrf_L_dense_mp_gpu_fp8_wrapper.c
    kernels/potrf_L_dense_mp_gpu_fp8_sp_wrapper.c
    kernels/potrf_L_dense_mp_gpu_fp8_adaptive_wrapper.c
    kernels/isyrk_wrapper.c
    kernels/gemm_wrapper.c
    kernels/gemmex_wrapper.c
    kernels/hamming_binary_wrapper.c
    
    # Utility files
    core/hicma_parsec_rank_statistics.c
    core/hicma_parsec_sparse_analysis.c
    core/hicma_parsec_check.c
    core/hicma_parsec_distribution.c
    core/hicma_parsec_decision.c
    core/hicma_parsec_flat_file.c
    core/hicma_parsec_datatype_convert.c
    core/hicma_parsec_io.c
    
    # Climate emulator files
    climate_emulator/climate_emulator.c
    climate_emulator/climate_emulator_gpu.c
    climate_emulator/climate_emulator_read_write.c
    
    # C++ files
    core/hicma_parsec.cpp
)

# Fugaku-specific files
if(ON_FUGAKU)
    list(APPEND SRC
        core/fortran_files/dgesdd.f
        core/fortran_files/droundup_lwork.f
        core/fortran_files/dorgqr.f
        core/fortran_files/dgeqrf.f
    )
    message(STATUS "Adding Fugaku-specific Fortran files")
endif(ON_FUGAKU)

# JDF (Job Description Files) for PARSEC
set(JDF
    # Core JDF files
    kernels/potrf_L_warmup.jdf
    kernels/band_free_memory.jdf
    kernels/band_regenerate.jdf
    kernels/band_size_dense_auto_tuning.jdf
    kernels/diagonal_correction_calculation.jdf
    
    # POTRF JDF files
    kernels/potrf_L_dense_tlr_dp.jdf
    kernels/potrf_L_sparse_tlr_dp_general.jdf
    kernels/potrf_L_sparse_tlr_dp_balance.jdf
    kernels/potrf_L_dense_tlr_mp.jdf
    kernels/potrf_L_dense_mp_gpu.jdf
    kernels/potrf_L_dense_sp_hp_band.jdf
    kernels/potrf_L_dense_mp_band.jdf
    kernels/potrf_L_dense_mp_gpu_fp8.jdf
    kernels/potrf_L_dense_mp_gpu_fp8_sp.jdf
    kernels/potrf_L_dense_mp_gpu_fp8_adaptive.jdf
    
    # GEMM and related JDF files
    kernels/isyrk_LT.jdf
    #kernels/misyrk_LT.jdf
    kernels/gemm_TN_summa.jdf
    kernels/gemmex_TN_summa.jdf
    kernels/gemm_TN.jdf
    #kernels/gemm_TN_batched.jdf
    kernels/gemmex_TN.jdf
    kernels/reorder_gemm.jdf
    
    # Utility JDF files
    kernels/rank_gather.jdf
    kernels/rank_check.jdf
    kernels/rank_print.jdf
    kernels/matrix_ssum.jdf
    kernels/matrix_uncompress.jdf
    kernels/matrix_compress.jdf
    kernels/decision_make_comp.jdf
    kernels/decision_make_send.jdf
    kernels/decision_count.jdf
    
    # Conversion JDF files
    kernels/convert_d2s.jdf
    kernels/convert_s2d.jdf
    kernels/datatype_convert_dense_static.jdf
    kernels/memory_allocation_dense_static.jdf
    kernels/matrix_dsum.jdf
    kernels/matrix_check_norm_diff.jdf
    kernels/matrix_det.jdf
    
    # LAPACK to Tile conversion JDF files
    kernels/Lapack_to_Tile.jdf
    kernels/Tile_to_Lapack.jdf
    kernels/Tile_to_Lapack_single.jdf
    kernels/Lapack_to_Tile_Single.jdf
    kernels/Lapack_to_Tile_sym.jdf
    kernels/Tile_to_Lapack_sym.jdf
    kernels/Tile_to_Lapack_sym_single.jdf
    kernels/Tile_to_Lapack_int.jdf
    kernels/Tile_to_Lapack_int8.jdf
    
    # Other utility JDF files
    kernels/replicate_to_rows.jdf
    kernels/sqrt_kernel.jdf
    kernels/genotype_generator.jdf
    kernels/dist_allocate.jdf
    kernels/dist_allocate_sym.jdf
    kernels/convert_d2i.jdf
    kernels/convert_i2d.jdf
    kernels/convert_s2i.jdf
    kernels/convert_i2s.jdf
    kernels/sqr_sum_vec_cyclic.jdf
    kernels/copy_vector_bcdd.jdf
    kernels/matrix_add.jdf
    kernels/cast_double_diag.jdf
    kernels/memory_free_tile.jdf
    kernels/dist_read.jdf
    kernels/matrix_sum_vec.jdf
    kernels/matrix_norm_get.jdf
    
    # Climate emulator JDF files
    climate_emulator/climate_emulator_read_csv_complex.jdf
    climate_emulator/climate_emulator_read_csv_double.jdf
    climate_emulator/climate_emulator_read_csv_double_timeslot.jdf
    climate_emulator/climate_emulator_read_csv_double2complex.jdf
    climate_emulator/climate_emulator_read_csv_double2complex_timeslot.jdf
    climate_emulator/climate_emulator_geqsht_forward_pre_computed_version.jdf
    climate_emulator/climate_emulator_diff_double.jdf
    climate_emulator/climate_emulator_geqsht_inverse_pre_computed_version.jdf
    climate_emulator/climate_emulator_mse.jdf
    climate_emulator/climate_emulator_print_double.jdf
    climate_emulator/climate_emulator_geqsht_forward_reshape.jdf
    climate_emulator/climate_emulator_matrix_norm_get.jdf
    
    # Binary and Hamming JDF files
    kernels/hamming_binary.jdf
    kernels/dist_allocate_hamming.jdf
    kernels/hamming_binary_generator.jdf
    kernels/matrix_unique_element.jdf
    kernels/check_read_data.jdf
)

# CUDA support
if(HICMA_PARSEC_HAVE_CUDA)
    if(ENABLE_CUTLASS)
        list(APPEND SRC 
            core/hicma_parsec.cu 
            core/hamming_distance.cu
        )
        message(STATUS "Adding CUDA files with CUTLASS support")
    else(ENABLE_CUTLASS)
        list(APPEND SRC core/hicma_parsec.cu)
        message(STATUS "Adding CUDA files")
    endif(ENABLE_CUTLASS)
endif(HICMA_PARSEC_HAVE_CUDA)

# HIP support
if(HICMA_PARSEC_HAVE_HIP)
    set_source_files_properties(core/hicma_parsec_hip.cu PROPERTIES LANGUAGE HIP)
    set_source_files_properties(core/hicma_parsec_hip.cu PROPERTIES COMPILE_FLAGS "-x hip")
    list(APPEND SRC core/hicma_parsec_hip.cu)
    message(STATUS "Adding HIP files")
endif(HICMA_PARSEC_HAVE_HIP)

# Create the library target
if(BUILD_SHARED_LIBS)
    add_library(hicma_parsec SHARED ${SRC})
    message(STATUS "Building shared library: libhicma_parsec.so")
else(BUILD_SHARED_LIBS)
    add_library(hicma_parsec STATIC ${SRC})
    message(STATUS "Building static library: libhicma_parsec.a")
endif(BUILD_SHARED_LIBS)

# Set target properties
set_target_properties(hicma_parsec PROPERTIES
    VERSION ${HICMA_PARSEC_VERSION}
    SOVERSION ${HICMA_PARSEC_VERSION_MAJOR}
    OUTPUT_NAME hicma_parsec
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
)

# Add PARSEC PTG sources
target_ptg_sources(hicma_parsec PRIVATE ${JDF})

# GPU-specific target properties
if(HICMA_PARSEC_HAVE_CUDA)
    set_target_properties(hicma_parsec PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    message(STATUS "CUDA separable compilation enabled")
endif(HICMA_PARSEC_HAVE_CUDA)

if(HICMA_PARSEC_HAVE_HIP)
    set_target_properties(hicma_parsec PROPERTIES HIP_SEPARABLE_COMPILATION ON)
    message(STATUS "HIP separable compilation enabled")
endif(HICMA_PARSEC_HAVE_HIP)

# Link libraries
target_link_libraries(hicma_parsec ${LD_LIB_BASIC} starsh)

# Add MPI C++ libraries if available
if(MPI_CXX_FOUND)
    target_link_libraries(hicma_parsec ${MPI_CXX_LIBRARIES})
    message(STATUS "Added MPI C++ libraries to hicma_parsec: ${MPI_CXX_LIBRARIES}")
endif()

# Add dependencies for HIP device headers
if(HICMA_PARSEC_HAVE_HIP)
    add_dependencies(hicma_parsec parsec)
endif(HICMA_PARSEC_HAVE_HIP)

# Set include directories for the target
target_include_directories(hicma_parsec PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Add compile definitions
target_compile_definitions(hicma_parsec PRIVATE
    $<$<BOOL:${HICMA_PARSEC_HAVE_CUDA}>:HICMA_PARSEC_HAVE_CUDA>
    $<$<BOOL:${HICMA_PARSEC_HAVE_HIP}>:HICMA_PARSEC_HAVE_HIP>
    $<$<BOOL:${ON_FUGAKU}>:ON_FUGAKU>
    $<$<BOOL:${ENABLE_SANITIZER}>:ENABLE_SANITIZER>
)

# Print build information
message(STATUS "=== HICMA PARSEC Source Configuration ===")
message(STATUS "Source files: ${SRC}")
message(STATUS "JDF files: ${JDF}")
message(STATUS "Library type: $<IF:$<BOOL:${BUILD_SHARED_LIBS}>,shared,static>")
message(STATUS "=========================================")

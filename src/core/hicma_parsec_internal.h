/**
 * @file hicma_parsec_internal.h
 * @brief HiCMA PaRSEC internal header file
 * 
 * This header contains internal definitions, structures, and function declarations
 * for the HiCMA (Hierarchical Computations on Manycore Architectures) library
 * using PaRSEC (Parallel Runtime Scheduling and Execution Controller).
 * 
 * HiCMA is a library for hierarchical matrix computations that leverages
 * low-rank approximations to reduce computational complexity and memory usage
 * for dense linear algebra operations. This internal header provides the core
 * infrastructure for:
 * 
 * - Mixed-precision computations (FP64, FP32, FP16, FP8)
 * - Tile Low-Rank (TLR) matrix representations
 * - GPU acceleration with CUDA and HIP support
 * - Sparse matrix analysis and optimization
 * - Adaptive precision selection and memory management
 * - Performance profiling and benchmarking
 * 
 * The library supports various scientific computing applications including
 * climate modeling, genomics, and electrodynamics simulations.
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 * 
 * @version 1.0.0
 */

#ifndef HICMA_PARSEC_INTERNAL_H
#define HICMA_PARSEC_INTERNAL_H

/* ============================================================================
 * Feature test macros for additional functions
 * ============================================================================ */
#define _GNU_SOURCE        /**< Enable GNU extensions for enhanced functionality */
#define _POSIX_C_SOURCE 200809L  /**< Enable POSIX.1-2008 features for modern system calls */

/* ============================================================================
 * System includes
 * ============================================================================ */
#include <math.h>      /**< Mathematical functions and constants */
#include <limits.h>    /**< System limits and ranges for integer types */
#include <stdbool.h>   /**< Boolean type support */
#include <stdint.h>    /**< Fixed-width integer types */
#include <sys/types.h> /**< System data types */
#include <sys/stat.h>  /**< File status information */
#include <unistd.h>    /**< POSIX operating system API */
#include <stdlib.h>    /**< Standard library functions */
#include <stdio.h>     /**< Standard I/O functions */
#include <string.h>    /**< String manipulation functions */

/* ============================================================================
 * System definitions
 * ============================================================================ */

/* Define PATH_MAX if not already defined */
#ifndef PATH_MAX
#define PATH_MAX 4096  /**< Maximum path length for file operations */
#endif

/* Type definitions for compatibility */
#ifndef uint
typedef unsigned int uint;  /**< Unsigned integer type for compatibility */
#endif

/* ============================================================================
 * PaRSEC includes
 * ============================================================================ */
#include "dplasma/parsec/parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic_band.h"  /**< Symmetric band matrix distribution */
#include "dplasma/parsec/parsec/utils/mca_param.h"  /**< Parameter management utilities */
#include "dplasma/parsec/parsec/runtime.h"  /**< PaRSEC runtime system */
#include "dplasma/parsec/parsec/include/parsec/execution_stream.h"  /**< Execution stream management */
#include "dplasma/tests/common_timing.h"  /**< Common timing utilities for testing */
#include "dplasma/parsec/parsec/data_dist/matrix/matrix_internal.h"  /**< Internal matrix data structures */

/* ============================================================================
 * DPLASMA includes
 * ============================================================================ */
#include "dplasma/src/dplasmajdf.h"  /**< DPLASMA Job Data Flow definitions */

/* ============================================================================
 * Starsh includes
 * ============================================================================ */
#include <starsh-randtlr.h>        /**< Random Tile Low-Rank matrix generation */
#include <starsh-electrodynamics.h> /**< Electrodynamics kernel functions */
#include <starsh-spatial.h>        /**< Spatial kernel functions */
#include <starsh-rbf.h>            /**< Radial Basis Function kernels */

/* ============================================================================
 * HCORE includes
 * ============================================================================ */
#include "hcore.h"     /**< Core HCORE functionality */
#include "hcore_d.h"   /**< Double precision HCORE operations */
#include "hcore_s.h"   /**< Single precision HCORE operations */

/* ============================================================================
 * Conditional includes
 * ============================================================================ */

/* Recursive support - enables hierarchical task decomposition */
#if defined(PARSEC_HAVE_DEV_RECURSIVE_SUPPORT)
#include "dplasma/parsec/parsec/data_dist/matrix/subtile.h"  /**< Sub-tile matrix operations */
#include "dplasma/parsec/parsec/recursive.h"  /**< Recursive task scheduling */
#endif

/* CUDA support - NVIDIA GPU acceleration */
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
#include "parsec/mca/device/cuda/device_cuda.h"  /**< CUDA device management */
#include "parsec/mca/device/cuda/device_cuda_internal.h"  /**< Internal CUDA device functions */
#include "parsec/utils/zone_malloc.h"  /**< Zone-based memory allocation */
#include <cuda_runtime.h>  /**< CUDA runtime API */
#include <cuda_fp16.h>     /**< Half precision floating point support */
#include <cublasLt.h>      /**< cuBLASLt library for tensor operations */
#include <cuda_fp8.h>      /**< FP8 precision support */
#include <cuda_fp8.hpp>    /**< C++ FP8 precision support */

#undef CUBLAS_H_  /**< Undefine to avoid conflicts */
#include <cublas_v2.h>  /**< cuBLAS v2 API for linear algebra */
#include <cusolverDn.h>  /**< cuSOLVER dense linear algebra */
#endif /* PARSEC_HAVE_DEV_CUDA_SUPPORT */

/* HIP support - AMD GPU acceleration */
#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
#include "parsec/utils/zone_malloc.h"  /**< Zone-based memory allocation */
#include "hicma_parsec_hip_cuda.h"     /**< HIP-CUDA compatibility layer */
#include "../dplasma/parsec/parsec/mca/device/hip/device_hip.h"  /**< HIP device management */
#endif

/* ============================================================================
 * C++ compatibility
 * ============================================================================ */
#ifdef __cplusplus
extern "C" {  /**< Enable C linkage for C++ compatibility */
#endif

/* ============================================================================
 * Feature flags and configuration
 * ============================================================================ */

/* Datatypes - Unique identifiers for different numeric precision types */
/* 64-bit types */
#define HICMA_PARSEC_FP64      1   /**< 64-bit double precision floating point */
#define HICMA_PARSEC_INT64     2   /**< 64-bit signed integer */

/* 32-bit types */
#define HICMA_PARSEC_FP32      3   /**< 32-bit single precision floating point */
#define HICMA_PARSEC_INT32     4   /**< 32-bit signed integer */
#define HICMA_PARSEC_UINT32    5   /**< 32-bit unsigned integer */

/* 16-bit types */
#define HICMA_PARSEC_FP16      6   /**< 16-bit half precision floating point */
#define HICMA_PARSEC_BF16      7   /**< 16-bit bfloat16 floating point format */
#define HICMA_PARSEC_INT16     8   /**< 16-bit signed integer */
#define HICMA_PARSEC_UINT16    9   /**< 16-bit unsigned integer */

/* 8-bit types */
#define HICMA_PARSEC_FP8_E4M3  10  /**< 8-bit floating point (4 exponent, 3 mantissa bits) */
#define HICMA_PARSEC_FP8_E5M2  11  /**< 8-bit floating point (5 exponent, 2 mantissa bits) */
#define HICMA_PARSEC_INT8      12  /**< 8-bit signed integer */
#define HICMA_PARSEC_UINT8     13  /**< 8-bit unsigned integer */

/* 4-bit types */
#define HICMA_PARSEC_INT4      14  /**< 4-bit signed integer */
#define HICMA_PARSEC_UINT4     15  /**< 4-bit unsigned integer */

/* 1-bit type */
#define HICMA_PARSEC_ONE_BIT   16  /**< Single bit data type */ 

/* Tile Formats */
#define HICMA_PARSEC_DENSE      1  
#define HICMA_PARSEC_LOW_RANK   2
#define HICMA_PARSEC_SPARSE     3

/* Application-specific flags - control specialized computation modes */
#define FOR_CLIMATE_EMULATOR    0  /**< Enable climate emulator mode for atmospheric modeling */
#define GENOMICS                    0  /**< Enable genomics mode for genetic analysis */
#define GENOMICS_ALLOCATE_SP    0  /**< Allocate in single precision for genomics (memory optimization) */
#define GENOMICS_ALLOCATE_INT   0  /**< Use integer allocation for genomics (genotype data) */
#define CHOLESKY_CPU_ONLY       0  /**< Force CPU-only Cholesky decomposition (disable GPU) */
#define GEMM_SUMMA              0  /**< Enable SUMMA GEMM algorithm for matrix multiplication */
#define DISTFILE                0  /**< Enable distributed file I/O for large datasets */
#define SINGLEFILE              0  /**< Enable single file I/O for simplified data handling */
#define DATATYPE                double  /**< Double precision for general applications */
#define DATATYPE_USE            HICMA_PARSEC_FP64 

/* Data type selection - choose precision based on application */
#if GENOMICS
#undef DATATYPE
#undef DATATYPE_USE 
#define DATATYPE                float   /**< Single precision for genomics applications */
#define DATATYPE_USE            HICMA_PARSEC_FP32 
#endif

/* ============================================================================
 * Matrix format and precision definitions
 * ============================================================================ */

/* Matrix format types - define storage formats for different precision levels */
#define NB_DECISIONS 6
#define DENSE_DP    1  /**< Dense double precision (64-bit floating point) */
#define DENSE_SP    2  /**< Dense single precision (32-bit floating point) */
#define LOW_RANK_DP 3  /**< Low-rank double precision (compressed representation) */
#define LOW_RANK_SP 4  /**< Low-rank single precision (compressed representation) */
#define DENSE_HP    5  /**< Dense half precision (16-bit floating point) */
#define DENSE_FP8   6  /**< Dense FP8 precision (8-bit floating point) */

/* Matrix format checking macros - determine tile storage format */
#define IS_DENSE(m, n) (                                 \
       DENSE_DP == params_tlr->decisions[n*params_tlr->NT+m] \
    || DENSE_SP == params_tlr->decisions[n*params_tlr->NT+m] \
    || DENSE_HP == params_tlr->decisions[n*params_tlr->NT+m] \
    || DENSE_FP8 == params_tlr->decisions[n*params_tlr->NT+m] \
    )  /**< Check if tile (m,n) is stored in dense format */

#define IS_ALLOCATE_DP(m, n) (                           \
       !GENERATE_RANDOM_DATA                             \
    || params_tlr->auto_band                             \
    || params_tlr->adaptive_decision                     \
    || DENSE_DP == params_tlr->decisions[n*descA->lmt+m] \
    )  /**< Check if tile (m,n) should be allocated in double precision */

/* ============================================================================
 * Precision masks for GEMM operations
 * ============================================================================ */
#define MASK_FP64                  0x1   /**< Double precision mask (64-bit) */
#define MASK_FP32                  0x2   /**< Single precision mask (32-bit) */
#define MASK_TF32                  0x4   /**< TensorFloat-32 mask (NVIDIA tensor cores) */
#define MASK_TF16_A16_B16_C32_OP32 0x8   /**< TF16 A16 B16 C32 OP32 mask (mixed precision) */
#define MASK_TF16_A16_B16_C16_OP16 0x10  /**< TF16 A16 B16 C16 OP16 mask (full TF16) */
#define MASK_BF16_A16_B16_C32_OP32 0x20  /**< BF16 A16 B16 C32 OP32 mask (Brain Float 16) */
#define MASK_BF16_A16_B16_C16_OP16 0x40  /**< BF16 A16 B16 C16 OP16 mask (full BF16) */
#define MASK_TF16_A16_B16_C16_OP32 0x80  /**< TF16 A16 B16 C16 OP32 mask (TF16 with FP32 ops) */
#define MASK_ONLY_FP16             0xFFFF8 /**< FP16-only options mask (exclude FP32/FP64) */

/* ============================================================================
 * GPU architecture definitions
 * ============================================================================ */
#define GPU_ARCH_NVIDIA_V100 1  /**< NVIDIA V100 GPU (Volta architecture) */
#define GPU_ARCH_NVIDIA_A100 2  /**< NVIDIA A100 GPU (Ampere architecture) */
#define GPU_ARCH_NVIDIA_H100 3  /**< NVIDIA H100 GPU (Hopper architecture) */

/* ============================================================================
 * Feature enablement flags
 * ============================================================================ */
#define ENABLE_TF16_A16_B16_C16_OP16   0  /**< Enable TF16 A16 B16 C16 OP16 (tensor core operations) */
#define HAVE_HP                        1  /**< Enable half precision support (FP16) */
#define DEBUG_INFO                     0  /**< Enable debug information and verbose output */
#define ENABLE_PROFILING               0  /**< Enable GPU profiling and performance monitoring */
#define GENERATE_RANDOM_DATA           0  /**< Generate random data for testing */

/* Conditional feature flags - enable features based on available hardware */
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
#define HAVE_FP8 1  /**< Enable FP8 precision on CUDA (8-bit floating point) */
#define HAVE_I8  1  /**< Enable INT8 precision on CUDA (8-bit integer) */
#else
#define HAVE_FP8 0  /**< FP8 precision not available (no CUDA support) */
#define HAVE_I8  0  /**< INT8 precision not available (no CUDA support) */
#endif

#if FOR_CLIMATE_EMULATOR
#undef GENERATE_RANDOM_DATA
#define GENERATE_RANDOM_DATA 0  /**< Disable random data generation for climate emulator */
#endif

/* ============================================================================
 * Platform-specific definitions
 * ============================================================================ */

/* Fugaku supercomputer specific definitions - handle AArch64 architecture */
#if defined(ON_FUGAKU)
#define main MAIN__  /**< Fortran-style main function for Fugaku */
#define HAVE_HP_CPU 1  /**< Enable half precision CPU support on Fugaku */
typedef __int64_t hicma_parsec_int64_t;  /**< 64-bit integer type for Fugaku */
#else
#define HAVE_HP_CPU 0  /**< Disable half precision CPU support on other platforms */
typedef int64_t hicma_parsec_int64_t;  /**< Standard 64-bit integer type */
#endif

/* ============================================================================
 * Utility macros
 * ============================================================================ */

/* String conversion macros - convert macro values to strings */
#define xstr(a) str(a)  /**< Expand macro then convert to string */
#define str(a) #a       /**< Convert macro argument to string literal */

/* ============================================================================
 * Debug and timing print macros
 * ============================================================================ */

/* Synchronized time printing macro - print timing information from rank 0 */
#undef SYNC_TIME_PRINT
#define SYNC_TIME_PRINT(rank, print) do {                          \
        SYNC_TIME_STOP();                                           \
        if(0 == rank) {                                             \
            printf("[****] TIME(s) %12.5f : ", sync_time_elapsed);  \
            printf print;                                           \
            fflush(stdout);                                         \
            fflush(stderr);                                         \
        }                                                           \
  } while(0)  /**< Print synchronized timing information with custom message */

/* Verbose printing macro - print debug information when verbose mode is enabled */
#undef VERBOSE_PRINT
#define VERBOSE_PRINT(rank, verbose, print) do {                               \
        if(0 == rank && verbose) {                                             \
            printf print;                                           \
            fflush(stdout);                                         \
            fflush(stderr);                                         \
        }                                                           \
  } while(0)  /**< Print verbose debug information from rank 0 */ 

/* ============================================================================
 * Color codes for terminal output
 * ============================================================================ */
#define RED   "\x1B[31m"  /**< Red color for error messages */
#define GRN   "\x1B[32m"  /**< Green color for success messages */
#define YEL   "\x1B[33m"  /**< Yellow color for warning messages */
#define BLU   "\x1B[34m"  /**< Blue color for information messages */
#define MAG   "\x1B[35m"  /**< Magenta color for special output */
#define CYN   "\x1B[36m"  /**< Cyan color for debug information */
#define WHT   "\x1B[37m"  /**< White color for normal text */
#define PUR   "\033[0;35m" /**< Purple color for headers */
#define RESET "\x1B[0m"   /**< Reset color to default */

/* ============================================================================
 * Algorithm configuration constants
 * ============================================================================ */

#define NB_PROBLEM 16  /**< Number of problem types supported by the library */

/* Data type conversion strategies - control where precision conversion occurs */
#define RECEIVER_CONVERT 0  /**< Receiver converts data (reduce sender overhead) */
#define SENDER_CONVERT   1  /**< Sender converts data (reduce receiver overhead) */
#define ADAPTIVE_CONVERT 2  /**< Adaptive conversion strategy (dynamic selection) */

/* Sparse matrix handling modes - control sparse matrix processing */
#define NOT_SPARSE     0  /**< Not sparse (dense matrix processing) */
#define SPARSE_NORMAL  1  /**< Normal sparse mode (standard sparse algorithms) */
#define SPARSE_BALANCE 2  /**< Balanced sparse mode (workload-balanced sparse algorithms) */

/* Cholesky decomposition variants - different algorithm implementations */
#define DENSE_TLR_MP          1  /**< Dense TLR mixed precision (dense band + low-rank off-band) */
#define DENSE_TLR_DP          2  /**< Dense TLR double precision (full double precision) */
#define DENSE_MP_BAND         3  /**< Dense mixed precision band (mixed precision in band) */
#define DENSE_SP_HP_BAND      4  /**< Dense single/half precision band (SP/HP in band) */
#define DENSE_MP_GPU          5  /**< Dense mixed precision GPU (GPU-accelerated mixed precision) */
#define SPARSE_TLR_DP_GENERAL 6  /**< Sparse TLR double precision general (sparse + low-rank) */
#define SPARSE_TLR_DP_BALANCE 7  /**< Sparse TLR double precision balanced (workload-balanced sparse) */
#define DENSE_MP_GPU_FP8      8  /**< Dense mixed precision GPU FP8 (FP8 precision on GPU) */
#define DENSE_MP_GPU_FP8_SP   9  /**< Dense mixed precision GPU FP8 single (FP8 + SP on GPU) */
#define DENSE_MP_GPU_FP8_ADAPTIVE   10  /**< Dense mixed precision GPU FP8 single (FP8 + SP on GPU) */

/* TLR boundary conditions - control off-band computation */
#define TLR_BOUND_WITHOUT_OFFBAND_GEMM      0  /**< Skip off-band GEMM operations */
#define TLR_BOUND_WITHOUT_OFFBAND_GEMM_TRSM 0  /**< Skip off-band GEMM and TRSM operations */

/* Analysis data type - data type for matrix analysis operations */
#define DATATYPE_ANALYSIS uint16_t  /**< 16-bit unsigned integer for matrix analysis */

/* Auto-tuning parameters - parameters for automatic performance tuning */
#define FLUCTUATION 0.66666667  /**< Fluctuation factor for auto-band tuning (2/3) */

/* ============================================================================
 * Debugging and profiling flags
 * ============================================================================ */

/* Rank statistics - control rank information output */
#define PRINT_RANK 0  /**< Enable rank statistics printing (debugging) */

/* Rank mapping configuration - rank analysis parameters */
#define RANK_MAP_BUFF 4  /**< Buffer size for rank mapping operations */
#define RANK_MAP_TYPE 5  /**< Number of rank mapping types supported */

/* Critical path analysis - performance analysis flags */
#ifndef PRINT_CRITICAL_PATH_TIME
#define PRINT_CRITICAL_PATH_TIME 0  /**< Print critical path timing information */
#endif
#ifndef PRINT_KERNEL_TIME
#define PRINT_KERNEL_TIME 0  /**< Print per-kernel/task timing information */
#endif

/* ============================================================================
 * Memory and performance configuration
 * ============================================================================ */

#define BAND_MEMORY_CONTIGUOUS 0  /**< Use contiguous memory for band (memory layout optimization) */
#define THRESHOLD_MEMORY_PER_NODE INT_MAX  /**< Memory threshold per node (unlimited) */
#define WORKLOAD_BALANCE 1  /**< Workload balancing strategy (enable load balancing) */
#define FREE_BAND_MEMORY    0  /**< Free band memory flag (keep band memory allocated) */
#define FREE_OFFBAND_MEMORY 1  /**< Free off-band memory flag (free off-band memory) */
#define PERFORMANCE_BAND_BOUND 0  /**< Performance testing band bound (disable performance bounds) */
#define PORTION_NORM 1  /**< Portion of band for global norm calculation (full band) */

/* ============================================================================
 * Application-specific configuration
 * ============================================================================ */

#if GENOMICS
/* Genomics-specific flags - specialized settings for genetic analysis */
#define WRITE_A 0      /**< Write kernel matrix to file (genomics output) */
#define MPIIO 0        /**< Use MPI I/O for parallel file operations */
#define CHECKSOLVE 0   /**< Check solution correctness (validation) */
#define PREDICTION 0   /**< Enable prediction mode (genetic prediction) */
#else
/* General application flags - standard settings for general applications */
#define WRITE_A 0      /**< Write kernel matrix to file (general output) */
#define MPIIO 0        /**< Use MPI I/O for parallel file operations */
#define CHECKSOLVE 0   /**< Check solution correctness (validation) */
#define PREDICTION 0   /**< Enable prediction mode (general prediction) */
#endif

/* ============================================================================
 * Mathematical constants and FLOPS calculations
 * ============================================================================ */

/* SYRK operation counts - symmetric rank-k update operation counts */
/* Note: FMULS_SYRK, FADDS_SYRK, FLOPS_DSYRK are already defined in dplasma/src/flops.h */

/* GEMM operation counts - general matrix-matrix multiplication operation counts */
/* Note: FMULS_GEMM, FADDS_GEMM, FLOPS_DGEMM, FLOPS_SGEMM are already defined in dplasma/src/flops.h */
#define FLOPS_GEMM(__m, __n, __k) (FMULS_GEMM((__m), (__k), (__n)) + FADDS_GEMM((__m), (__k), (__n)))  /**< Calculate total FLOPS for GEMM operation */

/* Machine epsilon values - precision limits for different floating point formats */
#define EPS_DP     1.11e-16  /**< Double precision epsilon (IEEE 754) */
#define EPS_SP     5.96e-08  /**< Single precision epsilon (IEEE 754) */
#define EPS_HP     4.88e-04  /**< Half precision epsilon (IEEE 754) */
#define EPS_HP_GPU 8.88e-01  /**< Half precision GPU epsilon (relaxed precision) */
#define EPS_FP8    0.0625    /**< FP8 precision epsilon (custom 8-bit format) */

/* ============================================================================
 * JDF (Job Data Flow) identifiers
 * ============================================================================ */

#define HICMA_GEMM_NN        1  /**< GEMM No-Transpose No-Transpose (C = A * B) */
#define HICMA_GEMM_NN_SUMMA  2  /**< GEMM NN with SUMMA algorithm (scalable matrix multiplication) */
#define HICMA_GEMM_NT        3  /**< GEMM No-Transpose Transpose (C = A * B^T) */
#define HICMA_GEMM_NT_SUMMA  4  /**< GEMM NT with SUMMA algorithm (scalable with transpose) */
#define HICMA_GEMM_TN        5  /**< GEMM Transpose No-Transpose (C = A^T * B) */
#define HICMA_GEMM_TN_SUMMA  6  /**< GEMM TN with SUMMA algorithm (scalable with transpose) */
#define HICMA_GEMM_TT        7  /**< GEMM Transpose Transpose (C = A^T * B^T) */
#define HICMA_GEMM_TT_SUMMA  8  /**< GEMM TT with SUMMA algorithm (scalable with both transposed) */

/* GPU buffer allocation strategy - memory management for GPU operations */
#define GPU_BUFFER_ONCE 1  /**< Allocate GPU buffer once and reuse (memory efficiency) */

/* ============================================================================
 * Utility macros
 * ============================================================================ */

/* FLOPS calculation macro - compute floating point operations */
#define PASTE_CODE_FLOPS( FORMULA, PARAMS, flops ) \
    flops = FORMULA PARAMS;  /**< Calculate FLOPS using specified formula and parameters */

typedef uint16_t hicma_parsec_decision_enum_t; 

/* ============================================================================
 * Utility functions
 * ============================================================================ */

/* Remove conflicting macros - avoid conflicts with system definitions */
#undef max
#undef min

/* Safe min/max functions for integers - avoid macro conflicts */
static inline int hicma_parsec_max(int a, int b) { 
    return a > b ? a : b;  /**< Return maximum of two integers */
}

static inline int hicma_parsec_min(int a, int b) { 
    return a < b ? a : b;  /**< Return minimum of two integers */
}

/* Safe min/max functions for doubles - avoid macro conflicts */
static inline double hicma_parsec_dmax(double a, double b) { 
    return a > b ? a : b;  /**< Return maximum of two doubles */
}

static inline double hicma_parsec_dmin(double a, double b) { 
    return a < b ? a : b;  /**< Return minimum of two doubles */
}

/**
 * @brief Calculate the most suitable process grid dimensions
 * 
 * This function finds the optimal number of rows (P) for a 2D process grid
 * given the total number of processes. It aims to create a grid that is
 * as close to square as possible for optimal load balancing.
 * 
 * @param nb_process Number of processes available
 * @return int The number of rows in the process grid (P)
 */
static inline int hicma_parsec_process_grid_calculation(int nb_process) {
    int P;
    /* Start from the square root and work downwards to find the largest divisor */
    for (P = (int)(sqrt(nb_process + 1.0)); P > 0; P--) {
        if (0 == nb_process % P) break;  /* Found a valid divisor */
    }
    return P;
}

/* ============================================================================
 * Data structures
 * ============================================================================ */

/**
 * @brief Main parameter structure for HiCMA PaRSEC operations
 * 
 * This structure contains all the parameters needed for HiCMA operations
 * including system configuration, matrix properties, algorithm settings,
 * and performance tuning parameters.
 */
typedef struct hicma_kernel_time_interval_s {
    double start_time;
    double end_time;
    struct hicma_kernel_time_interval_s *next;
} hicma_kernel_time_interval_t;

typedef struct hicma_parsec_params_s {
    /* ========================================================================
     * System configuration
     * ======================================================================== */
    int rank;                 /**< MPI rank of current process */
    int nodes;                /**< Number of compute nodes */
    int cores;                /**< Number of CPU cores per node */
    int gpus;                 /**< Number of GPUs per node */
    int gpu_type;             /**< NVIDIA GPU architecture type */
    int P;                    /**< Number of rows in process grid */
    int Q;                    /**< Number of columns in process grid */
    int mpi_initialized_by_hicma; /**< Whether HiCMA performed MPI_Init */
    
    /* ========================================================================
     * Matrix dimensions and properties
     * ======================================================================== */
    int M;                    /**< Number of rows in the matrix */
    int N;                    /**< Number of columns in the matrix */
    int K;                    /**< K dimension for GEMM operations */
    int MB;                   /**< Number of rows in a tile */
    int NB;                   /**< Number of columns in a tile */
    int NP;                   /**< Number of processes */
    int HMB;                  /**< Small MB for recursive H-DAGs */
    int HNB;                  /**< Small NB for recursive H-DAGs */
    int MT;                   /**< Number of tiles in row dimension */
    int NT;                   /**< Number of tiles in column dimension */
    int KT;                   /**< Number of tiles in K dimension */
    int RHS;                  /**< Right hand side vector size */
    
    /* ========================================================================
     * Algorithm control flags
     * ======================================================================== */
    int check;                /**< Enable result checking */
    int verbose;              /**< Enable verbose output */
    int maxrank;              /**< Maximum rank for low-rank approximation */
    int genmaxrank;           /**< Max rank for matrix generation */
    int compmaxrank;          /**< Max rank for computation */
    int fixedrk;              /**< Fixed rank threshold for HCORE_GEMM recompression */
    int adaptive_decision;    /**< Enable adaptive tile format decisions */
    int adaptive_decision_runtime;    /**< Enable adaptive tile format decisions during runtime */
    int adaptive_memory;      /**< Enable adaptive memory allocation: 0=memory allocated once; 1=memory reallocated per tile after precision decision */
    int arena_pinned_memory;  /**< Use pinned host memory for arenas when GPUs are enabled */
    int lookahead;            /**< Lookahead depth */
    int kind_of_problem;      /**< Type of problem being solved */
    int send_full_tile;       /**< Send full tile instead of compressed */
    int reorder_gemm;         /**< Enable GEMM reordering */
    int kind_of_cholesky;     /**< Type of Cholesky decomposition */
    int auto_band;            /**< Enable automatic band size tuning */
    int sparse;               /**< Enable sparse matrix handling */
    int adaptive_maxrank;     /**< Enable adaptive maxrank adjustment */
    int left_looking;         /**< Use left-looking algorithm */
    int nruns;                /**< Number of benchmark runs */
    
    /* ========================================================================
     * Band configuration
     * ======================================================================== */
    int band_size_dense;      /**< Dense band size */
    int band_size_dist;       /**< Band size distribution */
    int band_p;               /**< Row process grid for band distribution */
    int band_size_dense_dp;   /**< Dense band size in double precision */
    int band_size_dense_sp;   /**< Dense band size in single precision */
    int band_size_dense_hp;   /**< Dense band size in half precision */
    int band_size_low_rank_dp; /**< Low-rank band size in double precision */
    
    /* ========================================================================
     * Application-specific parameters
     * ======================================================================== */
    int numobj;               /**< Number of objects */
    int latitude;             /**< Latitude parameter for climate modeling */
    int nsnp;                 /**< Number of SNPs for genomics */
    int rbf_kernel;           /**< RBF kernel type */
    int order;                /**< Matrix reordering method */
    int tensor_gemm;          /**< Tensor core GEMM type */
    int datatype_convert;     /**< Data type conversion strategy */ 

    /* ========================================================================
     * Physical and numerical parameters
     * ======================================================================== */
    double wave_k;            /**< Wave number for electrodynamics problem */
    double radius;            /**< Radius in RBF kernel */
    double density;           /**< Density of sphere packing for RBF application */
    double fixedacc;          /**< Fixed accuracy threshold */
    double add_diag;          /**< Value added to diagonal elements in some problems */
    double band_size_auto_tuning_termination; /**< Termination threshold for band size auto-tuning on GPU */

    /* ========================================================================
     * File paths and I/O
     * ======================================================================== */
    char *mesh_file;          /**< Path to mesh file */
    char *pheno_file;         /**< Path to phenotype file */
    char exe_file_path[200];  /**< Executable file path */

    /* ========================================================================
     * Kernel parameters
     * ======================================================================== */
    int time_slots;           /**< Number of time slots */
    double sigma;             /**< Sigma parameter */
    double beta;              /**< Beta parameter */
    double nu;                /**< Nu parameter */
    double beta_time;         /**< Time-dependent beta parameter */
    double nu_time;           /**< Time-dependent nu parameter */
    double nonsep_param;      /**< Non-separable parameter */
    double noise;             /**< Noise level */

    /* ========================================================================
     * Decision statistics
     * ======================================================================== */
    double nb_dense_dp;       /**< Number of dense double precision tiles */
    double nb_dense_sp;       /**< Number of dense single precision tiles */
    double nb_dense_hp;       /**< Number of dense half precision tiles */
    double nb_dense_fp8;      /**< Number of dense FP8 precision tiles */
    double nb_low_rank_dp;    /**< Number of low-rank double precision tiles */
    double nb_low_rank_sp;    /**< Number of low-rank single precision tiles */
    uint64_t *nb_gemms;
    uint64_t *nb_datatype_conversions;
    int counter_stride;

    /* ========================================================================
     * Performance metrics
     * ======================================================================== */
    double cpu_perf;          /**< CPU performance (1 core) */
    double gpu_perf_nb_nb_nb; /**< GPU performance: GEMM NB*NB*NB */
    double gpu_perf_nb_nb_r;  /**< GPU performance: GEMM NB*NB*rank */
    double gpu_perf_nb_r_r;   /**< GPU performance: GEMM NB*rank*rank */

    /* ========================================================================
     * GPU memory information
     * ======================================================================== */
    double gpu_free_memory;   /**< Available GPU memory per GPU */
    double gpu_total_memory;  /**< Total GPU memory per GPU */

    /* ========================================================================
     * GPU band size limitations
     * ======================================================================== */
    int band_size_dense_gpu_memory_max;  /**< Max band size limited by GPU memory */
    int band_size_dense_gpu_balance_max; /**< Max band size limited by GPU load balance */
    int band_size_dense_gpu_time_max;    /**< Max band size limited by GPU timing */

    /* ========================================================================
     * Decision arrays for tile precision and operations
     * ======================================================================== */
    uint16_t *decisions;      /**< Precision decisions for each tile */
    uint16_t *decisions_gemm_last; /**< A*B precision of the last local GEMM for each tile */
    uint16_t *decisions_gemm_first; /**< A*B precision of the GEMM at k == 0 for each tile */
    uint16_t *decisions_storage_final; /**< Final stored precision after POTRF/TRSM for each tile */
    uint16_t *decisions_send; /**< Data conversion decisions for each tile */
    uint16_t *decisions_gemm_gpu; /**< GPU GEMM type decisions for each tile */

    /* ========================================================================
     * Problem type strings
     * ======================================================================== */
    char *str_problem[NB_PROBLEM]; /**< Array of problem type names */

    /* ========================================================================
     * Timing measurements
     * ======================================================================== */
    double time_init;         /**< Initialization time */
    double time_starsh;       /**< StarSH matrix generation time */
    double time_hicma;        /**< HiCMA computation time */
    double time_opt_band;     /**< Band optimization time */
    double time_regenerate;   /**< Matrix regeneration time (included in time_opt_band) */
    double time_reorder;      /**< Matrix reordering time */
    double time_analysis;     /**< Matrix analysis time */
    double time_decision_kernel; /**< Kernel decision time */
    double time_decision_sender; /**< Sender decision time */
    double time_syrk_app;     /**< SYRK application time */

    /* ========================================================================
     * Memory usage tracking
     * ======================================================================== */
    double memory_per_node;        /**< Memory usage per node */
    double memory_per_node_maxrank; /**< Memory usage per node with maxrank */

    /* ========================================================================
     * Norm calculations
     * ======================================================================== */
    int band_size_norm;       /**< Band size for norm calculation */
    double norm_global;       /**< Global matrix norm */
    double norm_global_diff;  /**< Global norm difference */
    double* norm_tile;        /**< Per-tile norm array */

    /* ========================================================================
     * Log-likelihood calculations
     * ======================================================================== */
    double log_det_dp;        /**< Log determinant in double precision */
    double log_det_mp;        /**< Log determinant in mixed precision */

    /* ========================================================================
     * Rank statistics
     * ======================================================================== */
    int *rank_array;          /**< Array to store rank information */
    
    /* Initial rank statistics */
    int imaxrk;               /**< Initial maximum rank */
    int iminrk;               /**< Initial minimum rank */
    double iavgrk;            /**< Initial average rank */
    
    /* Rank statistics after auto-band tuning */
    int imaxrk_auto_band;     /**< Initial max rank after auto-band tuning */
    int iminrk_auto_band;     /**< Initial min rank after auto-band tuning */
    double iavgrk_auto_band;  /**< Initial avg rank after auto-band tuning */
    
    /* Final rank statistics */
    int fmaxrk;               /**< Final maximum rank */
    int fminrk;               /**< Final minimum rank */
    double favgrk;            /**< Final average rank */

    /* ========================================================================
     * Critical path analysis
     * ======================================================================== */
    long long int critical_path_trsm_message; /**< Message size for TRSM in critical path */
    long long int total_critical_path_trsm_message; /**< Total TRSM message size in critical path */
    double critical_path_time; /**< Critical path execution time */

    /* ========================================================================
     * Operation counts (FLOPS)
     * ======================================================================== */
    unsigned long *op_band;    /**< Operation counts for band */
    unsigned long *op_offband; /**< Operation counts for off-band */
    unsigned long *op_path;    /**< Operation counts for critical path */
    unsigned long *op_offpath; /**< Operation counts for off-critical path */
    unsigned long total_band;  /**< Total operations in band */
    unsigned long total_offband; /**< Total operations in off-band */
    unsigned long total_path;  /**< Total operations in critical path */
    unsigned long total_offpath; /**< Total operations in off-critical path */
    unsigned long total_flops; /**< Total floating point operations */
    double gflops;             /**< Giga-FLOPS achieved */
    double flops;              /**< Total FLOPS */

    /* ========================================================================
     * Result validation
     * ======================================================================== */
    double result_accuracy;    /**< Accuracy of computed results */

    /* ========================================================================
     * Profiling and timing hooks
     * ======================================================================== */
    double start_time_potrf;      /**< Start time for POTRF */
    double start_time_syrk;       /**< Start time for SYRK */
    double start_time_gemm;       /**< Start time for GEMM */
    double start_time_kernel_add; /**< Start time for kernel addition */
    double start_time_kernel_sqr_sum_vec; /**< Start time for kernel square sum vector */
    double *gather_time;          /**< Array to gather timing information */
    double *gather_time_tmp;      /**< Temporary array for timing information */
    double *kernel_time_cpu;      /**< Per-CPU-worker GEMM timing accumulator */
    double *kernel_time_gpu;      /**< Per-GPU-device GEMM timing accumulator */
    double *kernel_time_gpu_event; /**< Per-GPU CUDA-event GEMM timing accumulator */
    hicma_kernel_time_interval_t **kernel_time_cpu_intervals; /**< Per-CPU-worker GEMM timing intervals */
    hicma_kernel_time_interval_t **kernel_time_gpu_intervals; /**< Per-GPU-device GEMM timing intervals */
    int kernel_time_cpu_count;    /**< Number of CPU timing accumulator entries */
    int kernel_time_gpu_count;    /**< Number of GPU timing accumulator entries */
    int kernel_time_gpu_event_count; /**< Number of CUDA-event GPU timing accumulator entries */
    double potrf_time;            /**< Total POTRF execution time */
    double trsm_time;             /**< Total TRSM execution time */
    double syrk_time;             /**< Total SYRK execution time */
    double potrf_time_temp;       /**< Temporary POTRF time */
    double trsm_time_temp;        /**< Temporary TRSM time */
    double syrk_time_temp;        /**< Temporary SYRK time */
    
    /* PaRSEC profiling hooks */
    parsec_hook_t* wrap_potrf;           /**< POTRF profiling hook */
    parsec_hook_t* wrap_trsm;            /**< TRSM profiling hook */
    parsec_hook_t* wrap_syrk;            /**< SYRK profiling hook */
    parsec_hook_t* wrap_gemm;            /**< GEMM profiling hook */
    parsec_hook_t* wrap_potrf_complete;  /**< POTRF completion hook */
    parsec_hook_t* wrap_trsm_complete;   /**< TRSM completion hook */
    parsec_hook_t* wrap_syrk_complete;   /**< SYRK completion hook */
    parsec_hook_t* wrap_gemm_complete;   /**< GEMM completion hook */

    /* ========================================================================
     * Task management
     * ======================================================================== */
    uint32_t nb_local_tasks;     /**< Number of local tasks */

    /* ========================================================================
     * Algorithm control and status
     * ======================================================================== */
    int uplo;                    /**< Upper/lower triangular flag */
    int band_size_dist_provided; /**< Whether band size distribution was provided */
    int info;                    /**< Return status information */
    int *info_gpu;               /**< GPU return status information */

    /* ========================================================================
     * GPU configuration
     * ======================================================================== */
    int gpu_rows;                /**< Number of GPU rows */
    int gpu_cols;                /**< Number of GPU columns */

    /* ========================================================================
     * Application-specific data
     * ======================================================================== */
    double* points;              /**< Pointer to spatial points data */

    /* Hamming distance specific */
    int nb_unique_elem;          /**< Number of unique elements */
    int *array_unique_elem;      /**< Array of unique elements */
    int current_hamming_id;      /**< Current Hamming distance ID */
} hicma_parsec_params_t;

/* ============================================================================
 * GPU workspace configuration
 * ============================================================================ */

/* ============================================================================
 * GPU workspace structures
 * ============================================================================ */

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

/**
 * @brief GPU stream workspace for POTRF operations
 * 
 * Contains CUDA handles and GPU memory buffers for efficient
 * POTRF computation on GPU.
 */
typedef struct parsec_potrf_stream_workspace_s {
    /* CUDA handles */
    cusolverDnHandle_t handle_cusolver;     /**< cuSolver handle */
    cublasHandle_t handle_cublas;           /**< cuBLAS handle */
    cublasHandle_t handle_cublas_deviceptr; /**< cuBLAS handle deviceptr */
    cublasHandle_t handle_cublas_tensor;    /**< cuBLAS tensor core handle */
    
    /* GPU memory buffers */
    void *gpu_buffer;       /**< General GPU buffer for DnDpotrf */
    int gpu_buffer_size;    /**< Size of general GPU buffer */
    void *gpu_buffer_A;     /**< Buffer A: NB * NB * sizeof(double) */
    void *gpu_buffer_B;     /**< Buffer B: NB * NB * sizeof(double) */
    void *gpu_buffer_C;     /**< Buffer C: NB * NB * sizeof(float) */
    void *gpu_buffer_mbr;   /**< Buffer for matrix-by-rank: NB * maxrank * sizeof(double) */
    void *gpu_buffer_rr;    /**< Buffer for rank-by-rank: maxrank * maxrank * sizeof(double) */

#if HAVE_FP8
    /* FP8 precision support */
    cublasLtHandle_t lightHandle;           /**< cuBLASLt handle */
    cublasLtMatmulDesc_t matmulDesc;        /**< Matrix multiplication descriptor */
    cublasLtMatrixLayout_t Adesc;           /**< Matrix A layout */
    cublasLtMatrixLayout_t Bdesc;           /**< Matrix B layout */
    cublasLtMatrixLayout_t Cdesc;           /**< Matrix C layout, FP16 */
    int fp8_ld;                             /**< 16-aligned leading dimension for FP8 cuBLASLt */
    void *workspace;                        /**< FP8 workspace */
    size_t workspaceSize;                   /**< FP8 workspace size */
    cublasLtMatmulHeuristicResult_t heuristicResultsArray; /**< Heuristic results */
    cublasLtMatmulPreference_t pref;        /**< Matrix multiplication preferences */
#endif

} parsec_potrf_stream_workspace_t;

/**
 * @brief GPU workspace for POTRF operations
 */
typedef struct parsec_potrf_gpu_workspace_s {
    parsec_potrf_stream_workspace_t *stream_workspace;  /**< Stream workspace */
    parsec_device_cuda_module_t *cuda_device;           /**< CUDA device module */
} parsec_potrf_gpu_workspace_t;

/**
 * @brief Main POTRF workspace structure
 */
typedef struct parsec_potrf_workspace_s {
    parsec_potrf_gpu_workspace_t *gpu_workspace;  /**< GPU workspace */
    int info;                                     /**< Return status */
} parsec_potrf_workspace_t;

#endif /* PARSEC_HAVE_DEV_CUDA_SUPPORT || PARSEC_HAVE_DEV_HIP_SUPPORT */

/* ============================================================================
 * Main data structure
 * ============================================================================ */

/**
 * @brief Main data structure for HiCMA PaRSEC operations
 * 
 * Contains all matrix descriptors and GPU workspace for HiCMA computations.
 */
typedef struct hicma_parsec_data_s {
    /* ========================================================================
     * Matrix descriptors
     * ======================================================================== */
    // TODO: dcAd and dcAcpy need to be full matrix? Consider matrix type requirements
    //parsec_matrix_block_cyclic_t dcAd;      /**< Dense mixed-precision matrix */
    //parsec_matrix_block_cyclic_t dcAcpy;     /**< Copy of dense mixed-precision matrix */
    parsec_matrix_sym_block_cyclic_t dcAd;       /**< Dense mixed-precision matrix */
    parsec_matrix_sym_block_cyclic_t dcAcpy;     /**< Copy of dense mixed-precision matrix */
    parsec_matrix_sym_block_cyclic_band_t dcA;        /**< Main matrix for TLR or mixed-precision+TLR */
    parsec_matrix_sym_block_cyclic_band_t dcAr;       /**< Rank matrix */
    parsec_matrix_sym_block_cyclic_band_t dcNorm;     /**< Norm matrix */
    parsec_matrix_sym_block_cyclic_band_t dcDist;     /**< Distribution matrix for kernel execution */
    parsec_matrix_sym_block_cyclic_band_t dcRank;     /**< Rank information matrix (if PRINT_RANK enabled) */
    parsec_matrix_block_cyclic_t dcFake;              /**< Fake matrix for auto-band distribution */
    parsec_matrix_sym_block_cyclic_band_t dcReorder;  /**< Reordering matrix for GEMM */
    
    /* ========================================================================
     * Result validation matrices
     * ======================================================================== */
    parsec_matrix_sym_block_cyclic_t dcA1;   /**< Result matrix 1 for validation */
    parsec_matrix_sym_block_cyclic_t dcA0;   /**< Result matrix 0 for validation */

    /* ========================================================================
     * Application-specific matrices
     * ======================================================================== */
    parsec_matrix_block_cyclic_t dcB;        /**< Phenotype matrix */
    parsec_matrix_block_cyclic_t dcX;        /**< Solution matrix */
    parsec_matrix_block_cyclic_t dcP;        /**< Prediction matrix */

    /* ========================================================================
     * GPU workspace
     * ======================================================================== */
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    parsec_potrf_workspace_t *ws_gpu;        /**< GPU workspace */
#endif
} hicma_parsec_data_t;

/* ============================================================================
 * Supporting data structures
 * ============================================================================ */

/**
 * @brief StarSH parameter structure
 */
typedef struct starsh_params_s {
    void *data;              /**< User data pointer */
    STARSH_kernel *kernel;   /**< StarSH kernel function */
    STARSH_int *index;       /**< Index array */
} starsh_params_t;

/**
 * @brief Rank statistics structure
 */
typedef struct hicma_parsec_stat_s {
    int max;     /**< Maximum rank */
    int min;     /**< Minimum rank */
    double avg;  /**< Average rank */
} hicma_parsec_stat_t;

/**
 * @brief Matrix analysis structure for sparse operations
 * 
 * This structure contains analysis information for sparse matrix operations,
 * including rank information, operation counts, and memory usage statistics.
 */
typedef struct hicma_parsec_matrix_analysis_s {
    /* ========================================================================
     * Rank information
     * ======================================================================== */
    DATATYPE_ANALYSIS *initial_rank;  /**< Initial rank array */
    uint8_t *final_rank;              /**< Final rank array */
    
    /* ========================================================================
     * TRSM analysis
     * ======================================================================== */
    DATATYPE_ANALYSIS **trsm_initial; /**< Initial TRSM rank matrix */
    DATATYPE_ANALYSIS *trsm_num_initial; /**< Initial TRSM number array */
    DATATYPE_ANALYSIS **trsm;         /**< TRSM rank matrix */
    DATATYPE_ANALYSIS *trsm_num;      /**< TRSM number array */
    unsigned long int total_trsm_num; /**< Total number of TRSM operations */
    
    /* ========================================================================
     * Row analysis
     * ======================================================================== */
    DATATYPE_ANALYSIS **row;          /**< Row rank matrix */
    DATATYPE_ANALYSIS *row_num;       /**< Row number array */
    
    /* ========================================================================
     * SYRK analysis
     * ======================================================================== */
    DATATYPE_ANALYSIS **syrk_local;   /**< Local SYRK rank matrix */
    DATATYPE_ANALYSIS *syrk_local_num; /**< Local SYRK number array */
    unsigned long int total_syrk_num; /**< Total number of SYRK operations */
    
    /* ========================================================================
     * GEMM analysis
     * ======================================================================== */
    DATATYPE_ANALYSIS **gemm_local;   /**< Local GEMM rank matrix */
    DATATYPE_ANALYSIS *gemm_local_num; /**< Local GEMM number array */
    DATATYPE_ANALYSIS *gemm_local_memory; /**< Local GEMM memory array */
    uint8_t *gemm_local_memory_arena_indicator; /**< Memory arena indicator (prevents double free) */
    unsigned long long int total_gemm_num; /**< Total number of GEMM operations */
    
    /* ========================================================================
     * Memory and density statistics
     * ======================================================================== */
    unsigned long long int total_memory; /**< Total memory usage */
    double initial_density;              /**< Initial matrix density */
    double density_trsm;                 /**< TRSM density */
    double density_gemm;                 /**< GEMM density */
} hicma_parsec_matrix_analysis_t;


/**
 * @brief Initialize and configure PaRSEC runtime system
 * 
 * This function sets up the PaRSEC runtime environment for HiCMA computations.
 * It initializes the distributed computing infrastructure, configures the
 * execution context, and prepares the system for parallel matrix operations.
 * 
 * The function handles:
 * - MPI initialization and process grid setup
 * - PaRSEC context creation and configuration
 * - Device initialization (CPU/GPU)
 * - Memory management setup
 * - Communication topology establishment
 * 
 * @param argc Number of command line arguments
 * @param argv Command line arguments array
 * @param params Pointer to HiCMA parameters structure (modified during setup)
 * @return Pointer to initialized PaRSEC context, or NULL on failure
 * 
 * @note This function must be called before any HiCMA computations
 * @note The function modifies params to reflect the actual system configuration
 * @note MPI must be initialized before calling this function
 */
parsec_context_t *setup_parsec(int argc, char* argv[], hicma_parsec_params_t *params);

/**
 * @brief Clean up and finalize PaRSEC runtime system
 * 
 * This function properly shuts down the PaRSEC runtime environment and
 * releases all allocated resources. It ensures clean termination of the
 * distributed computing system and proper cleanup of GPU resources.
 * 
 * The function handles:
 * - PaRSEC context finalization
 * - GPU device cleanup and memory deallocation
 * - Communication system shutdown
 * - Resource deallocation and cleanup
 * - Final MPI synchronization
 * 
 * @param parsec Pointer to PaRSEC context to clean up
 * @param params Pointer to HiCMA parameters structure
 * 
 * @note This function should be called before program termination
 * @note CUDA handles are automatically cleaned up by the CUDA runtime
 * @note All allocated memory should be freed before calling this function
 */
void cleanup_parsec(parsec_context_t* parsec, hicma_parsec_params_t *params); 

/**
 * @brief Parse command line arguments for HiCMA configuration
 * 
 * This function processes command line arguments and configures the HiCMA
 * parameters accordingly. It supports a comprehensive set of options for
 * controlling matrix dimensions, algorithm behavior, precision settings,
 * and performance tuning parameters.
 * 
 * Supported arguments include:
 * - Matrix dimensions (M, N, MB, NB)
 * - Algorithm selection (Cholesky type, precision mode)
 * - Performance tuning (band size, rank limits)
 * - I/O options (input files, output formats)
 * - Debugging and profiling options
 * 
 * @param _argc Pointer to argument count (modified to remove processed args)
 * @param _argv Pointer to argument vector (modified to remove processed args)
 * @param params Pointer to HiCMA parameters structure (populated with parsed values)
 * 
 * @note This function may exit the program if critical parameters are missing
 * @note MPI initialization requires MPI_THREAD_MULTIPLE support
 * @note Unrecognized arguments are ignored with a warning
 */
void parse_arguments(int *_argc, char*** _argv, hicma_parsec_params_t *params); 

/**
 * @brief Initialize HiCMA parameters with default values
 * 
 * This function initializes the HiCMA parameters structure with sensible
 * default values for all configuration options. It sets up the basic
 * system configuration, algorithm parameters, and performance tuning
 * settings before command line argument parsing.
 * 
 * Default initialization includes:
 * - System configuration (process grid, device settings)
 * - Matrix dimensions and tile sizes
 * - Algorithm control flags and precision settings
 * - Performance tuning parameters
 * - File paths and I/O settings
 * - Timing and profiling configuration
 * 
 * @param params Pointer to HiCMA parameters structure to initialize
 * @param argv Command line arguments (used to extract executable path)
 * @return 0 on success, non-zero on error
 * 
 * @note This function should be called before parse_arguments()
 * @note Default values are chosen for optimal performance on typical systems
 * @note Some parameters may be overridden by command line arguments
 */
int hicma_parsec_params_init(hicma_parsec_params_t *params, char **argv);

/**
 * @brief Print initial HiCMA configuration parameters
 * 
 * This function displays the initial configuration parameters that will be
 * used for the HiCMA computation. It provides a comprehensive overview of
 * system settings, algorithm configuration, and performance tuning parameters.
 * 
 * @param params Pointer to HiCMA parameters structure
 * 
 * @note Only rank 0 prints to avoid duplicate output in parallel execution
 * @note Output includes all major configuration parameters for debugging and verification
 */
void hicma_parsec_params_print_initial( hicma_parsec_params_t *params );

/**
 * @brief Print final HiCMA computation results and statistics
 * 
 * This function displays the final results, performance statistics, and
 * analysis data from the HiCMA computation. It provides comprehensive
 * information about execution time, memory usage, accuracy, and other
 * performance metrics.
 * 
 * @param argc Number of command line arguments
 * @param argv Command line arguments array
 * @param params Pointer to HiCMA parameters structure
 * @param analysis Pointer to matrix analysis structure with computation statistics
 * 
 * @note Only rank 0 prints to avoid duplicate output in parallel execution
 * @note Output format is designed for automated result analysis and benchmarking
 */
void hicma_parsec_params_print_final( int argc, char **argv,
        hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t *analysis );

/**
 * @brief Initialize StarSH kernel parameters for matrix generation
 * 
 * This function configures the StarSH kernel system for generating hierarchical
 * matrices with specific mathematical properties. It sets up kernel functions,
 * parameters, and data structures needed for matrix generation based on the
 * problem type and application requirements.
 * 
 * The function configures different kernel types based on kind_of_problem:
 * - Random TLR matrices for general testing
 * - Spatial kernels for electrodynamics and molecular dynamics
 * - Time-space kernels for climate modeling
 * - RBF kernels for machine learning applications
 * 
 * @param params_kernel Pointer to StarSH kernel parameters (initialized)
 * @param params Pointer to HiCMA parameters containing problem configuration
 * @return 0 on success, non-zero on error
 * 
 * @note This function configures different kernel types based on kind_of_problem
 * @note Kernel parameters are extracted from HICMA parameters and validated
 * @note The function must be called before matrix generation
 */
int hicma_parsec_kernel_init( starsh_params_t *params_kernel,
        hicma_parsec_params_t *params );

/**
 * @brief Validate and adjust HiCMA parameters for consistency
 * 
 * This function performs comprehensive validation of HiCMA parameters to ensure
 * they are consistent and compatible with the target system and algorithm
 * requirements. It checks parameter ranges, dependencies, and system constraints.
 * 
 * Validation includes:
 * - Matrix dimension compatibility with process grid
 * - Memory requirements vs. available system memory
 * - GPU capability vs. requested precision modes
 * - Algorithm parameter consistency
 * - System resource availability
 * 
 * @param params Pointer to HiCMA parameters structure to validate
 * @return 0 on success, non-zero on error
 * 
 * @note This function may modify parameters to ensure consistency
 * @note Critical errors cause program termination with error messages
 * @note Warnings are issued for non-critical parameter adjustments
 */
int hicma_parsec_params_check( hicma_parsec_params_t *params );

/**
 * @brief Initialize HiCMA data structures and matrix descriptors
 * 
 * This function initializes all data structures needed for HiCMA computations,
 * including matrix descriptors, GPU workspaces, and memory allocation. It sets
 * up the distributed matrix layout and prepares the system for computation.
 * 
 * Initialization includes:
 * - Matrix descriptor creation and distribution setup
 * - GPU workspace allocation and configuration
 * - Memory pool setup for different precision levels
 * - Data structure initialization for analysis and statistics
 * - Communication pattern setup for distributed operations
 * 
 * @param data Pointer to HiCMA data structure to initialize
 * @param params Pointer to HiCMA parameters structure
 * @return 0 on success, non-zero on error
 * 
 * @note This function allocates significant memory resources
 * @note GPU memory allocation depends on available GPU resources
 * @note Matrix distribution is optimized for the process grid configuration
 */
int hicma_parsec_data_init( hicma_parsec_data_t *data,
        hicma_parsec_params_t *params );

/**
 * @brief Automatic tuning of dense band size for optimal performance
 *
 * This function automatically determines the optimal dense band size for the
 * Cholesky factorization algorithm. It performs a search over different band
 * sizes to find the configuration that provides the best performance for the
 * given problem size and system configuration.
 *
 * The auto-tuning process:
 * - Temporarily disables GPU acceleration for accurate CPU-only benchmarking
 * - Tests different band sizes using binary search or linear search
 * - Selects the band size that minimizes execution time
 * - Re-enables GPU acceleration after tuning is complete
 *
 * @param parsec PaRSEC context for distributed computation
 * @param data Pointer to HICMA data structures containing matrix descriptors
 * @param params Pointer to HICMA parameters (modified during tuning)
 * @param params_kernel Pointer to STARSH kernel parameters
 * @return 0 on success, non-zero on error
 *
 * @note This function modifies params->band_size_dense if a better size is found
 * @note GPU acceleration is temporarily disabled during the tuning process
 * @note Memory allocation may be adjusted based on the new band size
 */
int hicma_parsec_band_size_dense_auto_tuning( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel );

/**
 * @brief Reorder GEMM operations for improved cache locality
 *
 * This function reorders General Matrix-Matrix multiplication (GEMM) operations
 * to improve cache locality and memory access patterns. The reordering is
 * disabled by default but can be enabled for specific performance optimizations.
 *
 * The reordering process:
 * - Analyzes the matrix access patterns
 * - Reorders operations to minimize cache misses
 * - Optimizes memory bandwidth utilization
 * - Maintains numerical correctness
 *
 * @param parsec PaRSEC context for distributed computation
 * @param data Pointer to HICMA data structures containing matrix descriptors
 * @param params Pointer to HICMA parameters controlling reordering behavior
 * @return 0 on success, non-zero on error
 *
 * @note This function is disabled by default for compatibility
 * @note Reordering may affect numerical precision in some cases
 * @note Performance benefits depend on matrix size and system architecture
 */
int hicma_parsec_reorder_gemm( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params );

/**
 * @brief Calculate memory requirements for Cholesky factorization
 *
 * This function calculates the memory requirements for the Cholesky factorization
 * algorithm, including both dense and low-rank storage needs. It analyzes the
 * matrix structure and determines the optimal memory allocation strategy.
 *
 * The calculation process:
 * - Analyzes matrix dimensions and structure
 * - Calculates dense band memory requirements
 * - Estimates low-rank storage needs
 * - Determines peak memory usage during factorization
 * - Provides memory optimization recommendations
 *
 * @param parsec PaRSEC context for distributed computation
 * @param params Pointer to HICMA parameters containing matrix dimensions and settings
 * @return 0 on success, non-zero on error
 *
 * @note Memory calculation is essential for large-scale problems
 * @note Results are used to optimize memory allocation strategies
 * @note Calculation considers both CPU and GPU memory requirements
 */
int hicma_parsec_memory_calculation( parsec_context_t *parsec,
        hicma_parsec_params_t *params );


/**
 * @brief Calculate the number of FLOPS and the critical path time
 *
 * This function calculates the total number of floating-point operations (FLOPS)
 * and the critical path time for the Cholesky factorization algorithm. It provides
 * performance analysis metrics for understanding computational complexity and
 * identifying performance bottlenecks.
 *
 * The calculation process:
 * - Counts floating-point operations for each kernel type
 * - Calculates the critical path through the task dependency graph
 * - Estimates theoretical peak performance
 * - Provides performance analysis metrics
 *
 * @param params Pointer to HICMA parameters containing matrix dimensions and settings
 * @return 0 on success, non-zero on error
 *
 * @note FLOPS calculation is essential for performance analysis
 * @note Critical path time determines theoretical minimum execution time
 * @note Results are used for performance optimization and benchmarking
 */
int hicma_parsec_cholesky_stat( hicma_parsec_params_t *params );

/**
 * @brief Free memory allocated for HICMA data structures
 *
 * This function deallocates all memory that was allocated for HICMA data structures,
 * including matrix descriptors, temporary buffers, and GPU memory. It ensures
 * proper cleanup of resources to prevent memory leaks.
 *
 * The cleanup process:
 * - Deallocates matrix descriptors and data structures
 * - Frees temporary buffers and workspace memory
 * - Releases GPU memory allocations
 * - Cleans up PaRSEC data distribution structures
 * - Resets pointers to prevent dangling references
 *
 * @param parsec PaRSEC context for distributed computation
 * @param data Pointer to HICMA data structures to be freed
 * @param params Pointer to HICMA parameters
 * @param params_kernel Pointer to STARSH kernel parameters
 * @param analysis Pointer to matrix analysis data structures
 *
 * @note This function should be called before program termination
 * @note All pointers are set to NULL after deallocation
 * @note GPU memory is properly released if CUDA/HIP support is enabled
 */
void hicma_parsec_free_memory( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_matrix_analysis_t *analysis );

/**
 * @brief Gather rank information from distributed matrix to local array
 *
 * This function collects rank information from the distributed matrix dcAr
 * and stores it in a local rank_array. It is used to gather rank statistics
 * for analysis and optimization purposes.
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [in] dcAr: distributed matrix containing rank data
 * @param [out] rank_array: local array to store gathered rank information
 * @param [in] band_size_dense: size of the dense band region
 * @return 0 on success, non-zero on error
 *
 * @note The rank_array must be pre-allocated with sufficient size
 * @note Only rank information from the dense band is gathered
 */
int parsec_rank_gather(parsec_context_t *parsec,
        parsec_tiled_matrix_t *dcAr,
        int *rank_array,
        int band_size_dense);

/**
 * @brief Validate rank correctness and mark invalid tiles in dense band
 *
 * This function performs validation of rank values in the distributed matrix
 * and marks tiles with invalid ranks by setting them to -1. It ensures
 * data integrity before proceeding with computations.
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [inout] dcAr: distributed matrix containing rank data to validate
 * @param [in] band_size_dense: size of the dense band region to check
 * @return 0 on success, non-zero on error
 *
 * @note Invalid rank values are set to -1 for identification
 * @note Only the dense band region is validated
 * @note This function modifies the input matrix dcAr
 */
int hicma_parsec_rank_check(parsec_context_t *parsec,
        parsec_tiled_matrix_t *dcAr,
        int band_size_dense);

/**
 * @brief Gather rank distribution and print statistics for analysis
 *
 * This function collects rank information from all distributed tiles,
 * computes distribution statistics, and prints them for analysis.
 * It provides insights into rank distribution patterns and helps
 * with performance optimization and debugging.
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [in] dcRank: distributed matrix containing rank data
 * @param [in] band_size_dense: size of the dense band region
 * @return 0 on success, non-zero on error
 *
 * @note Statistics are printed to stdout for analysis
 * @note Only rank information from the dense band is analyzed
 * @note This function is primarily used for debugging and optimization
 */
int hicma_parsec_rank_print(parsec_context_t *parsec,
                      parsec_tiled_matrix_t *dcRank,
                      int band_size_dense);

/**
 * @brief Warm up system for POTRF computation
 * 
 * This function performs a warm-up computation to initialize the system
 * and ensure optimal performance for subsequent POTRF operations. It
 * creates a small test matrix and performs a Cholesky factorization
 * to warm up both CPU and GPU resources.
 * 
 * The warm-up process:
 * - Creates a small test matrix with specified random seed
 * - Performs POTRF factorization on the test matrix
 * - Initializes GPU devices and memory pools
 * - Establishes optimal execution paths
 * 
 * @param rank MPI rank of the current process
 * @param uplo Upper or lower triangular flag for factorization
 * @param random_seed Random seed for test matrix generation
 * @param parsec PaRSEC context for distributed computation
 * 
 * @note This function creates a small test matrix for warm-up
 * @note GPU devices are warmed up after CPU to ensure proper initialization
 * @note The warm-up matrix is automatically cleaned up after use
 */
void hicma_parsec_warmup_potrf(int rank, dplasma_enum_t uplo, int random_seed, parsec_context_t *parsec);


/**
 * @brief Cholesky factorization for dense matrices with TLR (Tile Low-Rank) format
 *
 * This function performs Cholesky factorization (POTRF) on dense matrices
 * using the Tile Low-Rank (TLR) format with double precision arithmetic.
 * It implements a 2-flow algorithm for improved performance and memory efficiency.
 *
 * The factorization process:
 * - Decomposes the input matrix A into L*L^T form
 * - Uses TLR format for memory-efficient storage
 * - Implements 2-flow algorithm for optimal performance
 * - Supports distributed computation across multiple nodes
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [in] data: HICMA data structure containing the matrix to factorize
 * @param [in] params: HICMA parameters controlling factorization behavior
 * @param [out] info: Return status (0 on success, >0 if matrix is not positive definite)
 * @return 0 on success, non-zero on error
 *
 * @note Uses double precision arithmetic throughout
 * @note Implements 2-flow algorithm for performance optimization
 * @note Supports distributed computation across multiple MPI ranks
 * @note Returns info > 0 if the leading minor is not positive definite
 */
int potrf_L_dense_tlr_dp( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params );


/**
 * @brief GPU implementation of mixed-precision dense Cholesky factorization
 * 
 * Performs mixed-precision dense Cholesky factorization on GPU using
 * optimized cuBLAS and cuSOLVER kernels.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int potrf_L_dense_mp_gpu( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params );

/**
 * @brief GPU implementation of mixed-precision dense Cholesky factorization
 *        Adaptively change precision during runtime
 * 
 * Performs mixed-precision dense Cholesky factorization on GPU using
 * optimized cuBLAS and cuSOLVER kernels.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int potrf_L_dense_mp_gpu_fp8_adaptive( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params );

/**
 * @brief GPU implementation of mixed-precision dense Cholesky factorization with FP8
 * 
 * Performs mixed-precision dense Cholesky factorization on GPU with FP8 support
 * using optimized kernels and memory management.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int potrf_L_dense_mp_gpu_fp8( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params );

/**
 * @brief GPU implementation of mixed-precision dense Cholesky factorization with FP8 and single precision
 * 
 * Performs mixed-precision dense Cholesky factorization on GPU with FP8 support
 * and single precision arithmetic for optimal performance.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int potrf_L_dense_mp_gpu_fp8_sp( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params );

/**
 * @brief Cholesky factorization for sparse matrices with TLR format (general case)
 *
 * This function performs Cholesky factorization (POTRF) on sparse matrices
 * using the Tile Low-Rank (TLR) format with double precision arithmetic.
 * It implements a 2-flow algorithm optimized for general sparse matrix patterns.
 *
 * The factorization process:
 * - Handles sparse matrix structures efficiently
 * - Uses TLR format for memory-efficient storage of low-rank blocks
 * - Implements 2-flow algorithm for optimal performance
 * - Leverages matrix analysis for sparsity pattern optimization
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [in] data: HICMA data structure containing the sparse matrix to factorize
 * @param [in] params: HICMA parameters controlling factorization behavior
 * @param [in] analysis: Matrix analysis results for sparsity pattern optimization
 * @param [out] info: Return status (0 on success, >0 if matrix is not positive definite)
 * @return 0 on success, non-zero on error
 *
 * @note Optimized for general sparse matrix patterns
 * @note Uses matrix analysis for performance optimization
 * @note Implements 2-flow algorithm for improved efficiency
 * @note Supports distributed computation across multiple MPI ranks
 */
int potrf_L_sparse_tlr_dp_general( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t* analysis );


/**
 * @brief Cholesky factorization for sparse matrices with workload balancing
 *
 * This function performs Cholesky factorization (POTRF) on sparse matrices
 * using the Tile Low-Rank (TLR) format with double precision arithmetic.
 * It implements a 2-flow algorithm with advanced workload balancing to
 * optimize performance across distributed systems.
 *
 * The factorization process:
 * - Handles sparse matrix structures with workload balancing
 * - Uses TLR format for memory-efficient storage of low-rank blocks
 * - Implements 2-flow algorithm with load balancing
 * - Dynamically distributes work to minimize idle time
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [in] data: HICMA data structure containing the sparse matrix to factorize
 * @param [in] params: HICMA parameters controlling factorization behavior
 * @param [in] analysis: Matrix analysis results for workload optimization
 * @param [out] info: Return status (0 on success, >0 if matrix is not positive definite)
 * @return 0 on success, non-zero on error
 *
 * @note Implements advanced workload balancing for optimal performance
 * @note Uses matrix analysis for load distribution optimization
 * @note Minimizes idle time across distributed nodes
 * @note Supports dynamic load balancing during computation
 */
int potrf_L_sparse_tlr_dp_balance( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t* analysis );


/**
 * @brief Cholesky factorization with mixed-precision and TLR format
 *
 * This function performs Cholesky factorization (POTRF) on dense matrices
 * using mixed-precision arithmetic combined with Tile Low-Rank (TLR) format.
 * It leverages different precision levels for optimal performance and accuracy.
 *
 * The factorization process:
 * - Uses mixed-precision arithmetic for performance optimization
 * - Combines TLR format for memory efficiency
 * - Balances computational speed with numerical accuracy
 * - Supports distributed computation across multiple nodes
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [in] data: HICMA data structure containing the matrix to factorize
 * @param [in] params: HICMA parameters controlling mixed-precision behavior
 * @param [out] info: Return status (0 on success, >0 if matrix is not positive definite)
 * @return 0 on success, non-zero on error
 *
 * @note Uses mixed-precision arithmetic for optimal performance
 * @note Combines TLR format with precision optimization
 * @note Balances speed and accuracy requirements
 * @note Supports distributed computation across multiple MPI ranks
 */
int potrf_L_dense_tlr_mp( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params );

/**
 * @brief Cholesky factorization with mixed-precision (Single + Half precision)
 *
 * This function performs Cholesky factorization (POTRF) using mixed-precision
 * arithmetic combining Single Precision (SP) and Half Precision (HP) floating-point
 * formats. This approach optimizes performance while maintaining numerical accuracy.
 *
 * The factorization process:
 * - Uses Single Precision (SP) for main computations
 * - Leverages Half Precision (HP) for intermediate calculations
 * - Optimizes memory bandwidth and computational throughput
 * - Maintains numerical stability through careful precision management
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [in] data: HICMA data structure containing the matrix to factorize
 * @param [in] params: HICMA parameters controlling mixed-precision behavior
 * @param [out] info: Return status (0 on success, >0 if matrix is not positive definite)
 * @return 0 on success, non-zero on error
 *
 * @note Combines Single Precision (SP) and Half Precision (HP) arithmetic
 * @note Optimizes performance through precision-aware computation
 * @note Maintains numerical stability with mixed precision
 * @note Supports distributed computation across multiple MPI ranks
 */
int hicma_parsec_shpotrf( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params );

/**
 * @brief Cholesky factorization with mixed-precision (Half + Single + Double precision)
 *
 * This function performs Cholesky factorization (POTRF) using mixed-precision
 * arithmetic combining Half Precision (HP), Single Precision (SP), and Double
 * Precision (DP) floating-point formats. This three-level precision approach
 * maximizes performance while ensuring numerical accuracy.
 *
 * The factorization process:
 * - Uses Half Precision (HP) for high-throughput operations
 * - Leverages Single Precision (SP) for intermediate calculations
 * - Employs Double Precision (DP) for critical accuracy-sensitive operations
 * - Dynamically selects precision based on computational requirements
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [in] data: HICMA data structure containing the matrix to factorize
 * @param [in] params: HICMA parameters controlling mixed-precision behavior
 * @param [out] info: Return status (0 on success, >0 if matrix is not positive definite)
 * @return 0 on success, non-zero on error
 *
 * @note Combines Half (HP), Single (SP), and Double (DP) precision arithmetic
 * @note Maximizes performance through three-level precision optimization
 * @note Ensures numerical accuracy with precision-aware computation
 * @note Supports distributed computation across multiple MPI ranks
 */
int hicma_parsec_hsdpotrf( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params );

/**
 * @brief Regenerate tiles in the dense band region with updated parameters
 *
 * This function regenerates the tiles within the dense band region of a
 * distributed matrix using updated kernel parameters. It is used to
 * refresh the matrix data when kernel parameters change or for
 * iterative refinement processes.
 *
 * The regeneration process:
 * - Identifies tiles within the specified dense band
 * - Applies updated kernel parameters to regenerate tile data
 * - Maintains matrix structure and distribution
 * - Preserves data outside the dense band region
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [in] uplo: Matrix storage format (currently supports PlasmaLower)
 * @param [inout] dcA: distributed matrix to regenerate (modified in-place)
 * @param [in] params: HICMA parameters controlling regeneration behavior
 * @param [in] params_kernel: STARSH kernel parameters for tile generation
 * @param [in] band_size_dense: size of the dense band region to regenerate
 * @return 0 on success, non-zero on error
 *
 * @note Currently supports only lower triangular storage (PlasmaLower)
 * @note Only tiles within the dense band are regenerated
 * @note Matrix distribution and structure are preserved
 * @note Kernel parameters are applied to all tiles in the band
 */
int parsec_band_regenerate( parsec_context_t *parsec,
                int uplo,
                parsec_tiled_matrix_t *dcA,
                hicma_parsec_params_t *params,
                starsh_params_t *params_kernel,
                int band_size_dense);

/**
 * @brief Uncompress approximate matrix from TLR format to full dense format
 *
 * This function converts a compressed Tile Low-Rank (TLR) matrix back to
 * its full dense representation. It reconstructs the original matrix from
 * the compressed format using rank information and low-rank factors.
 *
 * The uncompression process:
 * - Reads rank information from dcAr matrix
 * - Reconstructs low-rank blocks from compressed data in dcA
 * - Generates full dense matrix in dcA0 output matrix
 * - Validates reconstruction accuracy and reports status
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [in] uplo: Matrix storage format (currently supports PlasmaLower)
 * @param [out] dcA0: output matrix in full dense format (pre-allocated)
 * @param [in] dcA: compressed matrix in TLR format
 * @param [in] dcAr: matrix containing rank information for each tile
 * @param [in] analysis: matrix analysis results for optimization
 * @param [in] band_size_dense: size of the dense band region
 * @param [in] maxrank: maximum rank allowed for low-rank blocks
 * @param [inout] info: status information (0 on success, >0 on error)
 * @return 0 on success, non-zero on error
 *
 * @note Currently supports only lower triangular storage (PlasmaLower)
 * @note Output matrix dcA0 must be pre-allocated with correct dimensions
 * @note Rank information in dcAr determines reconstruction accuracy
 * @note Maxrank parameter controls memory usage and accuracy trade-off
 */
int hicma_parsec_matrix_uncompress( parsec_context_t *parsec,
                int uplo,
                parsec_tiled_matrix_t *dcA0,
                parsec_tiled_matrix_t *dcA,
                parsec_tiled_matrix_t *dcAr,
                hicma_parsec_matrix_analysis_t *analysis,
                int band_size_dense,
                int maxrank,
                int *info);

/**
 * @brief Compress matrix from dense format to Tile Low-Rank (TLR) format
 *
 * This function compresses a dense matrix into the Tile Low-Rank (TLR) format
 * to reduce memory usage and improve computational efficiency. It analyzes
 * the matrix structure and applies low-rank approximation to suitable blocks.
 *
 * The compression process:
 * - Analyzes matrix blocks for low-rank structure
 * - Applies SVD or similar decomposition to compress blocks
 * - Stores compressed data in TLR format
 * - Records rank information for each compressed block
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [in] data: HICMA data structure containing the matrix to compress
 * @param [in] params_tlr: HICMA parameters controlling TLR compression
 * @param [in] params_kernel: STARSH kernel parameters for compression
 * @return 0 on success, non-zero on error
 *
 * @note Compression reduces memory usage for suitable matrices
 * @note TLR format maintains numerical accuracy within specified tolerance
 * @note Rank information is stored for later uncompression
 * @note Compression parameters control accuracy vs. memory trade-off
 */
int hicma_parsec_matrix_compress( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params_tlr,
        starsh_params_t *params_kernel );


/**
 * @brief Automatically tune dense band size for optimal performance
 *
 * This function automatically determines the optimal size of the dense band
 * region based on rank distribution analysis and system characteristics.
 * It performs performance analysis to find the best balance between
 * dense and low-rank computations.
 *
 * The auto-tuning process:
 * - Analyzes rank distribution across the matrix
 * - Tests different band sizes for performance
 * - Considers system memory and computational resources
 * - Selects optimal band size for given constraints
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [in] dcAr: distributed matrix containing rank information
 * @param [in] dcFake: auxiliary matrix used for distribution analysis
 * @param [in] rank_array: local array containing rank statistics
 * @param [inout] params: HICMA parameters (band size is updated)
 * @return 0 on success, non-zero on error
 *
 * @note Automatically updates band_size_dense in params structure
 * @note Uses rank distribution analysis for optimization
 * @note Considers system resources and performance characteristics
 * @note Results are used to optimize subsequent computations
 */
int parsec_band_size_dense_auto_tuning(parsec_context_t *parsec,
                          parsec_tiled_matrix_t *dcAr,
                          parsec_tiled_matrix_t *dcFake,
                          int *rank_array,
                          hicma_parsec_params_t *params );

/**
 * @brief Automatically tune dense band size using binary search optimization
 *
 * This function automatically determines the optimal size of the dense band
 * region using a binary search approach for efficient optimization.
 * It systematically tests different band sizes to find the configuration
 * that provides the best performance characteristics.
 *
 * The binary search optimization process:
 * - Defines search range based on matrix dimensions and rank distribution
 * - Uses binary search to efficiently explore band size options
 * - Evaluates performance metrics for each tested configuration
 * - Converges to optimal band size with minimal evaluations
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [in] dcAr: distributed matrix containing rank information
 * @param [in] dcFake: auxiliary matrix used for distribution analysis
 * @param [in] rank_array: local array containing rank statistics
 * @param [inout] params: HICMA parameters (band size is updated)
 * @return 0 on success, non-zero on error
 *
 * @note Uses binary search for efficient optimization
 * @note Automatically updates band_size_dense in params structure
 * @note More efficient than exhaustive search for large parameter spaces
 * @note Converges to near-optimal solution with logarithmic complexity
 */
int parsec_band_size_dense_auto_tuning_binary_search(parsec_context_t *parsec,
        parsec_tiled_matrix_t *dcAr,
        parsec_tiled_matrix_t *dcFake,
        int *rank_array,
        hicma_parsec_params_t *params );

/**
 * @brief Adaptively adjust maximum rank parameters based on system performance
 * 
 * This function dynamically adjusts the maximum rank parameters (maxrank and
 * compmaxrank) based on system performance characteristics and memory constraints.
 * It analyzes the current system state and optimizes rank limits for optimal
 * performance and memory usage.
 * 
 * The adaptive adjustment considers:
 * - Available system memory and GPU memory
 * - Current performance characteristics
 * - Matrix properties and rank distribution
 * - System load and resource availability
 * 
 * @param params Pointer to HiCMA parameters structure (modified with new rank limits)
 * @return 0 on success, non-zero on error
 * 
 * @note This function modifies maxrank and compmaxrank in the parameters structure
 * @note Adjustments are based on real-time system performance analysis
 * @note The function should be called after system warm-up for accurate measurements
 */
int hicma_parsec_adaptive_maxrank( hicma_parsec_params_t *params );

/**
 * @brief Count floating-point operations for different kernel types
 * 
 * This function calculates the number of floating-point operations (FLOPS)
 * for different types of linear algebra kernels based on their dimensions
 * and operation type. It provides accurate operation counts for performance
 * analysis and benchmarking.
 * 
 * Supported operation types:
 * - 'G' or 'g': GEMM (General Matrix Multiply)
 * - 'S' or 's': SYRK (Symmetric Rank-K update)
 * - 'T' or 't': TRSM (Triangular Solve with Multiple right-hand sides)
 * - 'P' or 'p': POTRF (Cholesky factorization)
 * 
 * @param op Operation type character ('G', 'S', 'T', 'P')
 * @param a First dimension parameter
 * @param b Second dimension parameter
 * @param c Third dimension parameter
 * @param d Fourth dimension parameter (if needed)
 * @return Number of floating-point operations for the specified kernel
 * 
 * @note Operation counts are calculated using standard BLAS formulas
 * @note Results are used for performance analysis and GFLOP/s calculations
 * @note Function returns 0 for unsupported operation types
 */
unsigned long int hicma_parsec_op_counts(char op, unsigned long int a, unsigned long int b, unsigned long int c, unsigned long int d);

/**
 * @brief Get previous k index for reordered GEMM operations
 * 
 * This function retrieves the previous k index in the reordered GEMM computation
 * sequence. It is used to determine the correct order of operations when GEMM
 * reordering is enabled for improved cache locality and performance.
 * 
 * @param descRG Pointer to reorder GEMM descriptor
 * @param m Row index of the current tile
 * @param n Column index of the current tile
 * @param k Current k index
 * @return Previous k index in the reordered sequence, or -1 if not found
 * 
 * @note This function is only used when GEMM reordering is enabled
 * @note Returns -1 for the first operation in the sequence
 */
int reorder_gemm_k_pre(parsec_tiled_matrix_t *descRG, int m, int n, int k);

/**
 * @brief Get next k index for reordered GEMM operations
 * 
 * This function retrieves the next k index in the reordered GEMM computation
 * sequence. It is used to determine the correct order of operations when GEMM
 * reordering is enabled for improved cache locality and performance.
 * 
 * @param descRG Pointer to reorder GEMM descriptor
 * @param m Row index of the current tile
 * @param n Column index of the current tile
 * @param k Current k index
 * @return Next k index in the reordered sequence, or -1 if not found
 * 
 * @note This function is only used when GEMM reordering is enabled
 * @note Returns -1 for the last operation in the sequence
 */
int reorder_gemm_k_next(parsec_tiled_matrix_t *descRG, int m, int n, int k);


/**
 * @brief Check norm difference between computed and reference matrices
 *
 * This function computes and checks the norm difference between the computed
 * matrix and a reference matrix to verify the correctness of the computation.
 * It is used for validation and testing purposes to ensure numerical accuracy.
 *
 * The validation process:
 * - Computes the norm of the difference matrix
 * - Compares against expected tolerance values
 * - Reports numerical accuracy metrics
 * - Validates computation correctness
 *
 * @param parsec PaRSEC context for distributed computation
 * @param data Pointer to HICMA data structures containing computed matrices
 * @param params_tlr Pointer to HICMA parameters
 * @param params_kernel Pointer to STARSH kernel parameters
 * @return 0 on success, non-zero on error
 *
 * @note This function is primarily used for testing and validation
 * @note Norm difference should be within machine precision for correct results
 * @note Results are printed for debugging and verification purposes
 */
int hicma_parsec_matrix_check_norm_diff(parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params_tlr,
        starsh_params_t *params_kernel );

/**
 * @brief Calculate the logarithm of the determinant of a matrix
 *
 * This function computes the logarithm of the determinant of a matrix
 * in distributed format. It is particularly useful for large matrices
 * where the determinant itself might overflow or underflow, but its
 * logarithm remains within representable range.
 *
 * The computation process:
 * - Performs Cholesky factorization of the input matrix
 * - Computes the sum of logarithms of diagonal elements
 * - Returns the logarithm of the determinant
 * - Handles numerical stability for large matrices
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [in] dcA: distributed matrix for which to compute log determinant
 * @param [out] log_det: computed logarithm of the determinant
 * @return 0 on success, non-zero on error
 *
 * @note Returns log(det(A)) to avoid overflow/underflow issues
 * @note Matrix must be positive definite for valid result
 * @note Uses Cholesky factorization for efficient computation
 * @note Supports distributed computation across multiple MPI ranks
 */
int hicma_parsec_matrix_det(parsec_context_t *parsec,
                       parsec_tiled_matrix_t *dcA,
                       double *log_det);

/**
 * @brief Free allocated memory and cleanup resources
 *
 * This function performs cleanup operations to free allocated memory
 * and release system resources. It should be called at the end of
 * computations to prevent memory leaks and ensure proper resource
 * management.
 *
 * The cleanup process:
 * - Frees all allocated memory blocks
 * - Releases system resources (GPU memory, file handles, etc.)
 * - Cleans up temporary data structures
 * - Resets internal state variables
 *
 * @param [in] parsec: PaRSEC context for distributed computation
 * @param [in] dcA: distributed matrix to free memory for
 * @param [in] params: HICMA parameters controlling cleanup behavior
 * @param [in] indicator: matrix type indicator (0 for full matrix, 1 for lower triangular)
 * @return 0 on success, non-zero on error
 *
 * @note Should be called at the end of program execution
 * @note Frees all memory allocated by HICMA functions
 * @note Releases GPU memory and other system resources
 * @note Prevents memory leaks and resource exhaustion
 */
int parsec_memory_free_tile(parsec_context_t *parsec,
        parsec_tiled_matrix_t *dcA,
        hicma_parsec_params_t *params, int indicator);

/**
 * @brief Compute column-wise sum of matrix elements
 * 
 * This function computes the sum of elements in each column of the distributed
 * matrix and stores the results in a vector. It performs the summation across
 * all processes and returns the global column sums.
 * 
 * The function:
 * - Computes local column sums for each process
 * - Performs global reduction to get total column sums
 * - Handles different matrix storage formats (dense, low-rank)
 * - Supports various precision levels
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param A Pointer to the distributed matrix
 * @param params Pointer to HiCMA parameters structure
 * @param vec Output vector containing column sums (allocated by caller)
 * 
 * @note The output vector must be allocated with size equal to matrix columns
 * @note Results are computed in single precision regardless of matrix precision
 * @note Function performs global reduction across all processes
 */
void hicma_parsec_matrix_sum_vec( parsec_context_t *parsec,
        parsec_tiled_matrix_t *A,
        hicma_parsec_params_t *params,
        float *vec);

/**
 * @brief Print symmetric matrix tile in human-readable format
 * 
 * This function extracts and prints a specific tile from a symmetric distributed
 * matrix in a formatted, human-readable way. It is primarily used for debugging
 * and verification purposes to inspect matrix contents during computation.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param dcA Pointer to the distributed symmetric matrix descriptor
 * @param node Node identifier for the tile location
 * @param p Row index of the tile in the tile grid
 * @param q Column index of the tile in the tile grid
 * 
 * @note This function allocates temporary memory that is freed before return
 * @note Output format is compatible with MATLAB matrix notation
 * @note Only the upper triangular part is printed for symmetric matrices
 */
void parsec_print_cm_sym(parsec_context_t* parsec, parsec_tiled_matrix_t *dcA,  int node, int p, int q);

/**
 * @brief Print general matrix tile in human-readable format
 * 
 * This function extracts and prints a specific tile from a general (non-symmetric)
 * distributed matrix in a formatted, human-readable way. It is primarily used
 * for debugging and verification purposes to inspect matrix contents during computation.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param dcA Pointer to the distributed general matrix descriptor
 * @param node Node identifier for the tile location
 * @param p Row index of the tile in the tile grid
 * @param q Column index of the tile in the tile grid
 * 
 * @note This function allocates temporary memory that is freed before return
 * @note Output format is compatible with MATLAB matrix notation
 * @note The entire matrix tile is printed (not just triangular part)
 */
void parsec_print_cm(parsec_context_t* parsec, parsec_tiled_matrix_t * dcA, int node, int p, int q);

/**
 * @brief Print int8 matrix tile in human-readable format
 * 
 * This function extracts and prints a specific tile from a distributed matrix
 * with int8 data type in a formatted, human-readable way. It is primarily used
 * for debugging and verification purposes to inspect matrix contents during
 * computation, particularly for integer-based algorithms.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param dcA Pointer to the distributed matrix descriptor with int8 data
 * @param node Node identifier for the tile location
 * @param p Row index of the tile in the tile grid
 * @param q Column index of the tile in the tile grid
 * 
 * @note This function allocates temporary memory that is freed before return
 * @note Output format shows integer values instead of floating-point
 * @note The entire matrix tile is printed with proper integer formatting
 */
void parsec_print_cm_int8(parsec_context_t* parsec, parsec_tiled_matrix_t * dcA, int node, int p, int q);

/**
 * @brief Replicate matrix data to rows for distributed computation
 * 
 * This function replicates matrix data across rows in a distributed matrix
 * environment. It is used to ensure data availability across different
 * computational nodes for parallel processing.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param dcAA Source matrix descriptor
 * @param dcC Destination matrix descriptor
 * @return 0 on success, error code on failure
 */
int hicma_parsec_replicate_to_rows(parsec_context_t *parsec, parsec_tiled_matrix_t *dcAA, parsec_tiled_matrix_t *dcC);

/**
 * @brief Apply square root kernel operation to matrix elements
 * 
 * Performs element-wise square root operation on matrix tiles with optional
 * diagonal addition and radius scaling. This is commonly used in numerical
 * algorithms requiring square root transformations.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param dcA Matrix descriptor to apply square root operation
 * @param rad Radius parameter for scaling
 * @param add_diag Value to add to diagonal elements
 * @param kernel Kernel type identifier for operation variant
 * @return 0 on success, error code on failure
 */
int hicma_parsec_sqrt_kernel(parsec_context_t *parsec, parsec_tiled_matrix_t *dcA,
                            double rad, double add_diag, int kernel);

/**
 * @brief Generate genotype matrix for genetic algorithm computations
 * 
 * Creates a genotype matrix with specified characteristics for genetic
 * algorithm operations. The matrix contains genetic information encoded
 * according to the specified genotype type.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param A Matrix descriptor for the generated genotype matrix
 * @param params HICMA parameters controlling generation process
 * @param genetype Type of genotype to generate (encoding scheme)
 * @return 0 on success, error code on failure
 */
int parsec_genotype_generator(parsec_context_t *parsec,
        parsec_tiled_matrix_t *A,
        hicma_parsec_params_t *params,
        int genetype);

/**
 * @brief Generate binary matrix for Hamming distance computations
 * 
 * Creates a binary representation of the input matrix optimized for
 * Hamming distance calculations. Converts input data to binary format
 * suitable for efficient distance computations.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param A Matrix descriptor for the binary matrix generation
 * @param params HICMA parameters controlling generation process
 * @return 0 on success, error code on failure
 */
int parsec_hamming_binary_generator(parsec_context_t *parsec,
        parsec_tiled_matrix_t *A,
        hicma_parsec_params_t *params);

/**
 * @brief Generate mixed precision GEMM operation with decision mapping
 * 
 * Creates a mixed precision General Matrix-Matrix multiplication operation
 * using a decision map to determine precision levels for different matrix
 * regions. This enables adaptive precision for optimal performance and accuracy.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param A Matrix descriptor for the operation
 * @param decision_map Array mapping matrix regions to precision levels
 * @param params HICMA parameters controlling the operation
 * @return 0 on success, error code on failure
 */
int parsec_gemmmp_TN_generator(parsec_context_t *parsec,
        parsec_tiled_matrix_t *A,
        uint16_t* decision_map,
        hicma_parsec_params_t *params);        

/**
 * @brief Allocate distributed matrix memory for general matrices
 * 
 * Allocates memory for a distributed matrix in the PaRSEC runtime system.
 * This function handles memory allocation across multiple nodes for
 * general (non-symmetric) matrix operations.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param A Matrix descriptor to allocate memory for
 * @param params_tlr HICMA parameters controlling allocation strategy
 * @return 0 on success, error code on failure
 */
int parsec_dist_allocate(parsec_context_t *parsec,
                         parsec_tiled_matrix_t *A,
                         hicma_parsec_params_t *params_tlr);

/**
 * @brief Allocate distributed matrix memory for symmetric matrices
 * 
 * Allocates memory for a distributed symmetric matrix in the PaRSEC runtime
 * system. This function optimizes memory allocation by taking advantage of
 * symmetry properties to reduce memory requirements.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param A Symmetric matrix descriptor to allocate memory for
 * @param params_tlr HICMA parameters controlling allocation strategy
 * @return 0 on success, error code on failure
 */
int parsec_dist_allocate_sym(parsec_context_t *parsec,
                         parsec_tiled_matrix_t *A,
                         hicma_parsec_params_t *params_tlr);

/**
 * @brief Allocate distributed matrix memory for Hamming distance computations
 * 
 * Allocates memory for a distributed matrix optimized for Hamming distance
 * calculations. This function handles integer data types and optimizes
 * memory layout for binary operations.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param A Matrix descriptor for Hamming distance operations
 * @param inttype Integer data type for the matrix elements
 * @param params_tlr HICMA parameters controlling allocation strategy
 * @return 0 on success, error code on failure
 */
int parsec_dist_allocate_hamming(parsec_context_t *parsec,
                         parsec_tiled_matrix_t *A, 
                         int inttype,
                         hicma_parsec_params_t *params_tlr);

/**
 * @brief Allocate distributed matrix memory for mixed precision GEMM operations
 * 
 * Allocates memory for a distributed matrix with mixed precision support
 * for General Matrix-Matrix multiplication operations. Uses decision mapping
 * to optimize memory allocation based on precision requirements.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param A Matrix descriptor for mixed precision GEMM operations
 * @param decision_map Array mapping matrix regions to precision levels
 * @param params_tlr HICMA parameters controlling allocation strategy
 * @return 0 on success, error code on failure
 */
int parsec_dist_allocate_gemmmp_TN(parsec_context_t *parsec,
                         parsec_tiled_matrix_t *A, 
                         uint16_t* decision_map,
                         hicma_parsec_params_t *params_tlr);

/**
 * @brief Copy vector data using BCDD (Block Cyclic Data Distribution) format
 * 
 * Copies vector data from a distributed matrix to a local array using
 * Block Cyclic Data Distribution format. This function handles the
 * conversion between distributed and local memory layouts.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param dcA Distributed matrix descriptor containing the vector data
 * @param copy_array Local array to store the copied vector data
 * @return 0 on success, error code on failure
 */
int parsec_copy_vector_bcdd(parsec_context_t *parsec,
        parsec_tiled_matrix_t *dcA,
        float *copy_array);

/**
 * @brief Add vector to matrix elements in distributed computation
 * 
 * Performs element-wise addition of a vector to matrix elements in a
 * distributed environment. The vector is broadcasted and added to
 * corresponding matrix elements across all tiles.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param A Matrix descriptor for the target matrix
 * @param vec Vector to add to matrix elements
 * @param params_tlr HICMA parameters controlling the operation
 * @param data HICMA data structures for computation
 * @return 0 on success, error code on failure
 */
int hicma_parsec_matrix_add(parsec_context_t *parsec,
        parsec_tiled_matrix_t* A,
        float* vec,              
        hicma_parsec_params_t * params_tlr,
        hicma_parsec_data_t *data);

/**
 * @brief Integer-based symmetric rank-k update operation
 * 
 * Performs the symmetric rank-k update operation C = alpha*A*A^T + beta*C
 * using integer arithmetic for improved performance in specific applications.
 * This is an optimized version of SYRK for integer data types.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param uplo Upper or lower triangular part of matrix C
 * @param trans Transpose operation on matrix A
 * @param alpha Scaling factor for A*A^T
 * @param A Input matrix for the rank-k update
 * @param beta Scaling factor for matrix C
 * @param C Result matrix (symmetric)
 * @param inttype Integer data type for computation
 * @param params_tlr HICMA parameters controlling the operation
 * @param data HICMA data structures for computation
 * @return 0 on success, error code on failure
 */
int dplasma_isyrk( parsec_context_t *parsec,
               dplasma_enum_t uplo,
               dplasma_enum_t trans,
               double alpha,
               parsec_tiled_matrix_t *A,
               double beta,
               parsec_tiled_matrix_t *C, int inttype, 
               hicma_parsec_params_t * params_tlr, 
               hicma_parsec_data_t *data);

/**
 * @brief Single precision symmetric rank-k update operation
 * 
 * Performs the symmetric rank-k update operation C = alpha*A*B^T + beta*C
 * using single precision floating-point arithmetic. This is the standard
 * SYRK operation for symmetric matrix updates.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param transA Transpose operation on matrix A
 * @param transB Transpose operation on matrix B
 * @param alpha Scaling factor for A*B^T
 * @param A First input matrix
 * @param B Second input matrix
 * @param beta Scaling factor for matrix C
 * @param C Result matrix (symmetric)
 * @param params_tlr HICMA parameters controlling the operation
 * @param data HICMA data structures for computation
 * @return 0 on success, error code on failure
 */
int hicma_parsec_syrk( parsec_context_t *parsec,
               dplasma_enum_t transA, dplasma_enum_t transB,
               float alpha, parsec_tiled_matrix_t *A,
                            parsec_tiled_matrix_t *B,
               float beta,  parsec_tiled_matrix_t *C, hicma_parsec_params_t * params_tlr,
               hicma_parsec_data_t *data);

/**
 * @brief Extended precision general matrix-matrix multiplication
 * 
 * Performs the general matrix-matrix multiplication operation C = alpha*A*B + beta*C
 * using double precision floating-point arithmetic with extended precision
 * support for high-accuracy computations.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param transA Transpose operation on matrix A
 * @param transB Transpose operation on matrix B
 * @param alpha Scaling factor for A*B
 * @param A First input matrix
 * @param B Second input matrix
 * @param beta Scaling factor for matrix C
 * @param C Result matrix
 * @param params_tlr HICMA parameters controlling the operation
 * @param data HICMA data structures for computation
 * @return 0 on success, error code on failure
 */
int hicma_parsec_gemmex( parsec_context_t *parsec,
               dplasma_enum_t transA, dplasma_enum_t transB,
               double alpha, parsec_tiled_matrix_t *A,
                             parsec_tiled_matrix_t *B,
               double beta,  parsec_tiled_matrix_t *C, hicma_parsec_params_t * params_tlr,
               hicma_parsec_data_t *data);

/**
 * @brief Hamming distance computation using binary matrix operations
 * 
 * Performs Hamming distance computation between two matrices using binary
 * operations. This function is optimized for genetic algorithm applications
 * and similarity computations where binary representations are used.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param transA Transpose operation on matrix A
 * @param transB Transpose operation on matrix B
 * @param alpha Scaling factor for the operation
 * @param A First input matrix (binary representation)
 * @param B Second input matrix (binary representation)
 * @param beta Scaling factor for matrix C
 * @param C Result matrix containing Hamming distances
 * @param params_tlr HICMA parameters controlling the operation
 * @param data HICMA data structures for computation
 * @return 0 on success, error code on failure
 */
int hicma_parsec_hamming_binary( parsec_context_t *parsec,
               dplasma_enum_t transA, dplasma_enum_t transB,
               float alpha, parsec_tiled_matrix_t *A,
                            parsec_tiled_matrix_t *B,
               float beta,  parsec_tiled_matrix_t *C, hicma_parsec_params_t * params_tlr,
               hicma_parsec_data_t *data);

/**
 * @brief Mixed precision general matrix-matrix multiplication (GEMM)
 * 
 * Performs the general matrix-matrix multiplication operation C = alpha*A*B + beta*C
 * using mixed precision arithmetic. This function automatically selects optimal
 * precision levels for different parts of the computation to balance accuracy
 * and performance.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param transA Transpose operation on matrix A
 * @param transB Transpose operation on matrix B
 * @param alpha Scaling factor for A*B
 * @param A First input matrix
 * @param B Second input matrix
 * @param beta Scaling factor for matrix C
 * @param C Result matrix
 * @param params_tlr HICMA parameters controlling precision selection
 * @param data HICMA data structures for computation
 * @return 0 on success, error code on failure
 */
int parsec_gemmmp_TN(parsec_context_t *parsec,
                    dplasma_enum_t transA, dplasma_enum_t transB,
                    float alpha, parsec_tiled_matrix_t* A, parsec_tiled_matrix_t* B,
                    float beta,  parsec_tiled_matrix_t* C, hicma_parsec_params_t * params_tlr,
                    hicma_parsec_data_t *data);

/**
 * @brief Warmup routine for Cholesky factorization (L variant)
 * 
 * Performs a warmup operation for the Cholesky factorization algorithm
 * using the lower triangular (L) variant. This function initializes
 * computational resources and prepares the system for optimal performance
 * during subsequent Cholesky factorization operations.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param dcA Matrix descriptor for the Cholesky factorization
 * @param params HICMA parameters controlling the warmup process
 * @param sym Symmetry flag (0 for general, 1 for symmetric)
 * @return 0 on success, error code on failure
 */
int hicma_parsec_potrf_L_warmup(parsec_context_t *parsec,
                       parsec_tiled_matrix_t *dcA,
                       hicma_parsec_params_t *params,
                       int sym);

/**
 * @brief free memory
 *
 * @param [inout] dcA:      the data, already distributed and allocated
 * @param [in] params:      parameters
 * @param [in] indicator:   0, only band; otherwise all
 */
int parsec_band_free_memory(parsec_context_t *parsec,
        parsec_tiled_matrix_t *dcA,
        hicma_parsec_params_t *params, int indicator);

/**
 * Cast diagonal tiles from FP32 to FP64
 *
 * @param [in] parsec:       parsec context
 * @param [inout] dcA:      the rank data, already distributed and allocated
 */
int hicma_parsec_cast_double_diag(parsec_context_t *parsec,
        parsec_tiled_matrix_t *dcA,
        hicma_parsec_params_t *params );

/**
 * @brief Compute squared sum of vector elements in distributed computation
 * 
 * Calculates the sum of squared elements from a distributed vector/matrix.
 * This function is commonly used for computing vector norms and statistical
 * measures in distributed linear algebra operations.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param A Input matrix/vector descriptor
 * @param C Result matrix descriptor (may be used for intermediate storage)
 * @return Sum of squared elements as double precision value
 */
double hicma_parsec_sqr_sum_vec( parsec_context_t *parsec,
                parsec_tiled_matrix_t *A, parsec_tiled_matrix_t *C);

/**
 * @brief Flush memory in Cholesky factorization for multiple calls
 *
 * This function flushes and resets the memory state in the Cholesky factorization
 * data structures. It is required when calling Cholesky factorization multiple
 * times to ensure clean state and prevent data corruption from previous runs.
 *
 * The flush process:
 * - Clears intermediate computation results
 * - Resets matrix state to initial condition
 * - Flushes GPU memory buffers if applicable
 * - Prepares data structures for fresh computation
 *
 * @param parsec PaRSEC context for distributed computation
 * @param data Pointer to HICMA data structures containing matrix descriptors
 * @param params Pointer to HICMA parameters
 *
 * @note This API is essential when calling Cholesky factorization multiple times
 * @note Memory flush ensures numerical correctness across multiple runs
 * @note GPU memory is properly flushed if CUDA/HIP support is enabled
 */
void hicma_parsec_memory_flush_choleksy( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params );


/**
 * @brief Extract unique elements from a distributed matrix
 * 
 * Identifies and extracts all unique elements from a distributed matrix.
 * This function is useful for data analysis, pattern recognition, and
 * preprocessing operations where unique values need to be identified.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param A Matrix descriptor containing the input data
 * @param params_tlr HICMA parameters controlling the operation
 * @return 0 on success, error code on failure
 */
int hicma_parsec_matrix_unique_element( parsec_context_t *parsec,
        parsec_tiled_matrix_t *A,
        hicma_parsec_params_t *params_tlr);

/**
 * @brief Validate data read operations in distributed computation
 * 
 * Performs validation checks on data read operations to ensure data
 * integrity and correct handling of distributed matrix data. This function
 * is used for debugging and verification of data flow correctness.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param A Matrix descriptor to validate
 * @param params_tlr HICMA parameters controlling validation
 * @return 0 if data read is correct, error code if issues found
 */
int hicma_parsec_check_read_data( parsec_context_t *parsec,
        parsec_tiled_matrix_t *A,
        hicma_parsec_params_t *params_tlr);

/**
 * @brief CPU implementation of Hamming distance binary conversion
 * 
 * Converts input matrix to binary representation for Hamming distance computation.
 * For each element in the input matrix:
 * - If element equals target value: output = 0
 * - If element differs from target value: output = 1
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param aout Output matrix (int8_t) - binary representation
 * @param ain Input matrix (int8_t) - original values
 * @param lda Leading dimension of input matrix
 * @param value Target value for binary conversion
 */
void hicma_parsec_hamming_subtract_ones_CPU(int nrows, int ncols, int8_t *aout, int8_t *ain, int lda, int value);

/**
 * @brief CPU implementation of Hamming distance identity matrix generation
 * 
 * Generates an identity matrix for Hamming distance computation by comparing
 * each element in the input matrix with a target value and setting the output
 * to 1 if they match, 0 otherwise.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param aout Output matrix (uint8_t) - identity matrix
 * @param ain Input matrix (int8_t) - original values
 * @param lda Leading dimension of input matrix
 * @param value Target value for identity matching
 */
void hicma_parsec_hamming_get_id_matrix_CPU(int nrows, int ncols, uint8_t *aout, int8_t *ain, int lda, int value);

/**
 * @brief CPU implementation of 32-bit integer memory copy
 * 
 * Performs a memory copy operation for 32-bit integer arrays. This function
 * is optimized for integer data and provides a simple interface for copying
 * integer matrices between memory locations.
 * 
 * @param nrows Number of rows in the arrays
 * @param ncols Number of columns in the arrays
 * @param _src Source array (int*)
 * @param _dest Destination array (int*)
 */
void hicma_parsec_memcpy_int32_CPU( int nrows, int ncols, int *_src, int *_dest);

/**
 * @brief CPU implementation of Hamming distance matrix merging
 * 
 * Merges two matrices for Hamming distance computation by adding the values
 * from the input matrix to the output matrix. The input matrix is accessed
 * in transposed order for efficient computation.
 * 
 * @param nrows Number of rows in the matrices
 * @param ncols Number of columns in the matrices
 * @param aout Output matrix (int) - accumulates the merged result
 * @param ain Input matrix (int) - values to add (transposed)
 * @param lda Leading dimension of both matrices
 */
void hicma_parsec_hamming_merge_CPU(int nrows, int ncols, int *aout, int *ain, int lda);

/**
 * @brief Perform datatype conversion on a dense matrix
 *
 * @param [in] parsec: The PaRSEC context
 * @param [in] uplo: Upper or lower triangular part to convert
 * @param [inout] A: The matrix to convert, already distributed and allocated
 * @param [in] allocate_type: Type of conversion to perform
 *                       - "double", "d" -> double precision floating point
 *                       - "float", "single", "s" -> single precision floating point
 *                       - "int8", "i8" -> 8-bit signed integer
 *                       - "int16", "i16" -> 16-bit signed integer
 *                       - "int32", "int", "i32", "i" -> 32-bit signed integer
 *                       - "int64", "i64" -> 64-bit signed integer
 *                       - "uint8", "u8" -> 8-bit unsigned integer
 *                       - "uint16", "u16" -> 16-bit unsigned integer
 *                       - "uint32", "uint", "u32", "u" -> 32-bit unsigned integer
 *                       - "uint64", "u64" -> 64-bit unsigned integer
 *                       - "half", "fp16", "h" -> 16-bit floating point (half precision)
 *                       - "fp8" -> 8-bit floating point representation
 *                       - "fp4" -> 4-bit floating point representation
 *                       - "int4" -> 4-bit integer representation
 *                       - "1bit" -> 1-bit representation
 * @return 0 on success
 */
int hicma_parsec_memory_allocation_dense_static(parsec_context_t *parsec,
        dplasma_enum_t uplo,
        parsec_tiled_matrix_t *A,
        char *allocate_type);

/**
 * @brief Test performance of CPU vs GPU computation
 * 
 * Performs benchmark tests to compare CPU and GPU performance
 * for various matrix operations to optimize computation strategy.
 * 
 * @param[in] params HICMA PaRSEC parameters
 */
void hicma_parsec_performance_testing_cpu_gpu( hicma_parsec_params_t * params );

/**
 * @brief Find optimal band size for dense GPU operations with maximum memory
 * 
 * Determines the optimal band size for dense GPU operations considering
 * available GPU memory and performance characteristics.
 * 
 * @param[in] params HICMA PaRSEC parameters
 */
void hicma_parsec_find_band_size_dense_gpu_memory_max( hicma_parsec_params_t * params );

#ifdef __cplusplus
}
#endif

#endif /* _HICMA_PARSEC_INTERNAL_H */

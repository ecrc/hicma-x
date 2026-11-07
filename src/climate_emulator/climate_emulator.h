/**
 * @file climate_emulator.h
 * @brief HICMA PaRSEC climate emulator header file
 * 
 * This header file contains declarations for climate emulation functions including
 * forward and inverse computations, data I/O operations, GPU acceleration, and
 * performance analysis for climate modeling applications.
 * 
 * The climate emulator implements spherical harmonic transforms for climate data
 * processing, supporting both CPU and GPU computations through the PaRSEC runtime.
 * 
 * Key features:
 * - Forward spherical harmonic transform (spatial to spectral)
 * - Inverse spherical harmonic transform (spectral to spatial)
 * - GPU acceleration using CUDA/HIP
 * - MPI-IO support for parallel file operations
 * - Performance monitoring and validation tools
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

#ifndef HICMA_PARSEC_CLIMATE_EMULATOR_H
#define HICMA_PARSEC_CLIMATE_EMULATOR_H

/* ============================================================================
 * HICMA PaRSEC includes
 * ============================================================================ */
#include "hicma_parsec.h"

/* ============================================================================
 * System includes
 * ============================================================================ */
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Configuration constants
 * ============================================================================ */
/** @brief Default latitude/longitude resolution (L=720 for global coverage) */
#define CLIMATE_EMULATOR_L_SIZE               720
/** @brief Number of time slots per input file */
#define CLIMATE_EMULATOR_TIME_SLOT_PER_FILE   1
/** @brief Total number of input files */
#define CLIMATE_EMULATOR_NUM_FILE             1 
/** @brief Enable debug information output */
#define CLIMATE_EMULATOR_DEBUG_INFO           0

/* ============================================================================
 * Feature flags
 * ============================================================================ */
#undef WRITE_A
#undef MPIIO 
#undef CHECKSOLVE 
#undef PREDICTION 

/** @brief Enable writing output to files */
#define CLIMATE_EMULATOR_WRITE_TO_FILE  0
/** @brief Enable inverse spherical harmonic transform */
#define CLIMATE_EMULATOR_ENABLE_INVERSE 0
/** @brief Enable reading input from files */
#define CLIMATE_EMULATOR_READ_FROM_FILE 0

/* The sources test these shorter spellings; keep them equal to the switches
 * above (they used to be undefined, which silently disabled every block they
 * guard, including the file-read error checks) */
#define READ_FROM_FILE  CLIMATE_EMULATOR_READ_FROM_FILE
#define ENABLE_INVERSE  CLIMATE_EMULATOR_ENABLE_INVERSE
#define WRITE_TO_FILE   CLIMATE_EMULATOR_WRITE_TO_FILE
#define DEBUG_INFO_CLIMATE_EMULATOR CLIMATE_EMULATOR_DEBUG_INFO

#if CLIMATE_EMULATOR_READ_FROM_FILE
#undef CLIMATE_EMULATOR_WRITE_TO_FILE
#define CLIMATE_EMULATOR_WRITE_TO_FILE  0
#endif

/** @brief Use MPI-IO for file operations */
#define CLIMATE_EMULATOR_MPIIO 1

/* ============================================================================
 * GPU support structures
 * ============================================================================ */
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

/**
 * @brief GPU stream workspace for climate emulator computations
 * 
 * This structure holds GPU-specific data buffers and handles for a single
 * CUDA/HIP execution stream, including cuBLAS handles and temporary buffers
 * for matrix operations.
 * 
 * The workspace is designed to support multiple concurrent streams on a single
 * GPU device, enabling overlapping computation and memory transfers.
 */
typedef struct climate_emulator_stream_workspace_s {
    cublasHandle_t handle;                    /**< cuBLAS handle for this stream */
    cuDoubleComplex *Gmtheta_r;              /**< Buffer for Gmtheta_r matrix: gb->f_data_M * gb->Ep_N */
    cuDoubleComplex *Fmnm;                   /**< Buffer for Fmnm matrix: gb->Et1_M * gb->Ep_N */
    cuDoubleComplex *tmp1;                   /**< Temporary buffer 1: gb->Et2_M * gb->P_N */
    cuDoubleComplex *tmp2;                   /**< Temporary buffer 2: gb->Et2_M * gb->Ep_N */
} climate_emulator_stream_workspace_t;

/**
 * @brief GPU workspace for a single GPU device
 * 
 * Contains stream workspaces for all execution streams on a single GPU device.
 * This structure manages multiple streams to enable concurrent execution of
 * different climate computation tasks.
 */
typedef struct climate_emulator_gpu_workspace_s {
    climate_emulator_stream_workspace_t *stream_workspace;  /**< Array of stream workspaces */
    parsec_device_cuda_module_t *cuda_device;              /**< CUDA device module */
} climate_emulator_gpu_workspace_t;

/**
 * @brief Main GPU workspace structure
 * 
 * Top-level structure containing GPU workspaces for all devices and error info.
 * This structure coordinates GPU resources across multiple devices in a
 * multi-GPU system.
 */
typedef struct climate_emulator_workspace_s {
    climate_emulator_gpu_workspace_t *gpu_workspace;       /**< Array of GPU workspaces */
    int info;                                              /**< Error/info code */
} climate_emulator_workspace_t;
#endif

/* ============================================================================
 * Main climate emulator structure
 * ============================================================================ */

/**
 * @brief Main climate emulator structure
 * 
 * This structure contains all the data and configuration needed for climate
 * emulation, including matrix descriptors, data arrays, and performance metrics.
 * It supports both forward (spatial to spectral) and inverse (spectral to spatial)
 * spherical harmonic transforms.
 * 
 * The structure is organized into logical sections:
 * - Configuration parameters
 * - Forward computation matrices
 * - Backward computation matrices
 * - Additional data arrays
 * - Performance metrics
 * - GPU workspace (if enabled)
 */
typedef struct climate_emulator_struct_s {
    /* ============================================================================
     * Configuration parameters
     * ============================================================================ */
    int L;                      /**< Latitude/longitude resolution (spherical harmonic degree) */
    int T;                      /**< Number of time slots */
    int NB;                     /**< Block size for matrix operations */
    int gpus;                   /**< Number of available GPUs */
    int nodes;                  /**< Number of compute nodes */
    int rank;                   /**< MPI rank of current process */
    int time_slot_per_file;     /**< Time slots per input file */
    int num_file;               /**< Total number of input files */
    int file_per_node;          /**< Files distributed per node */
    int verbose;               /**< Verbose */ 
    char *data_dir;             /**< Base directory for input/output files */

    /* ============================================================================
     * Forward computation matrices (spatial to spectral)
     * ============================================================================ */
    parsec_matrix_block_cyclic_t desc_f_data;  /**< Descriptor for f_data matrix */
    complex double *f_data;                     /**< Input climate data matrix */
    int f_data_M;                              /**< Rows in f_data matrix */
    int f_data_N;                              /**< Columns in f_data matrix */

    parsec_matrix_block_cyclic_t desc_Et1;     /**< Descriptor for Et1 matrix */
    complex double *Et1;                        /**< Et1 transformation matrix */
    int Et1_M;                                 /**< Rows in Et1 matrix */
    int Et1_N;                                 /**< Columns in Et1 matrix */

    parsec_matrix_block_cyclic_t desc_Et2;     /**< Descriptor for Et2 matrix */
    complex double *Et2;                        /**< Et2 transformation matrix */
    int Et2_M;                                 /**< Rows in Et2 matrix */
    int Et2_N;                                 /**< Columns in Et2 matrix */

    parsec_matrix_block_cyclic_t desc_Ep;      /**< Descriptor for Ep matrix */
    complex double *Ep;                         /**< Ep transformation matrix */
    int Ep_M;                                  /**< Rows in Ep matrix */
    int Ep_N;                                  /**< Columns in Ep matrix */

    parsec_matrix_block_cyclic_t desc_Slmn;    /**< Descriptor for Slmn matrix */
    complex double *Slmn;                       /**< Slmn transformation matrix */
    int Slmn_M;                                /**< Rows in Slmn matrix */
    int Slmn_N;                                /**< Columns in Slmn matrix */

    parsec_matrix_block_cyclic_t desc_Ie;      /**< Descriptor for Ie matrix */
    complex double *Ie;                         /**< Ie transformation matrix (even) */
    int Ie_M;                                  /**< Rows in Ie matrix */
    int Ie_N;                                  /**< Columns in Ie matrix */

    parsec_matrix_block_cyclic_t desc_Io;      /**< Descriptor for Io matrix */
    complex double *Io;                         /**< Io transformation matrix (odd) */
    int Io_M;                                  /**< Rows in Io matrix */
    int Io_N;                                  /**< Columns in Io matrix */

    parsec_matrix_block_cyclic_t desc_P;       /**< Descriptor for P matrix */
    complex double *P;                          /**< P transformation matrix */
    int P_M;                                   /**< Rows in P matrix */
    int P_N;                                   /**< Columns in P matrix */

    parsec_matrix_block_cyclic_t desc_D;       /**< Descriptor for D matrix */
    complex double *D;                          /**< D transformation matrix */
    int D_M;                                   /**< Rows in D matrix */
    int D_N;                                   /**< Columns in D matrix */

    parsec_matrix_block_cyclic_t desc_flm;     /**< Descriptor for flm matrix */
    parsec_matrix_block_cyclic_t desc_flmERA;  /**< Descriptor for flmERA matrix */
    double *flm;                               /**< Output spherical harmonic coefficients */
    double *flmERA;                            /**< Reference ERA spherical harmonic coefficients */
    int flm_M;                                 /**< Rows in flm matrices */
    int flm_N;                                 /**< Columns in flm matrices */

    /* ============================================================================
     * Backward computation matrices (spectral to spatial)
     * ============================================================================ */
    parsec_matrix_block_cyclic_t desc_Zlm;     /**< Descriptor for Zlm matrix */
    double *Zlm;                               /**< Zlm transformation matrix */
    int Zlm_M;                                 /**< Rows in Zlm matrix */
    int Zlm_N;                                 /**< Columns in Zlm matrix */

    parsec_matrix_block_cyclic_t desc_SC;      /**< Descriptor for SC matrix */
    double *SC;                                /**< SC transformation matrix */
    int SC_M;                                  /**< Rows in SC matrix */
    int SC_N;                                  /**< Columns in SC matrix */

    parsec_matrix_block_cyclic_t desc_f_spatial; /**< Descriptor for f_spatial matrix */
    double *f_spatial;                          /**< Output spatial field */
    int f_spatial_M;                           /**< Rows in f_spatial matrix */
    int f_spatial_N;                           /**< Columns in f_spatial matrix */

    parsec_matrix_block_cyclic_t desc_flmT;    /**< Descriptor for flmT matrix (used before SYRK) */
    int flmT_M;                                /**< Rows in flmT matrix */
    int flmT_N;                                /**< Columns in flmT matrix */
    int flmT_numNB;                            /**< Number of blocks in flmT */

    parsec_matrix_block_cyclic_t desc_A;       /**< Descriptor for A matrix (input to SYRK) */
    int A_M;                                   /**< Rows in A matrix */
    int A_N;                                   /**< Columns in A matrix */

    /* ============================================================================
     * Additional data arrays
     * ============================================================================ */
    double *phi;                               /**< Phi coordinate array */
    int phi_M;                                 /**< Rows in phi array */
    int phi_N;                                 /**< Columns in phi array */

    double *ts_test;                           /**< Test time series data */
    int ts_test_M;                             /**< Rows in ts_test array */
    int ts_test_N;                             /**< Columns in ts_test array */

    /* ============================================================================
     * Performance metrics
     * ============================================================================ */
    double mse;                                /**< Mean squared error */
    double time_init;                          /**< Initialization time */
    double time_forward;                       /**< Forward computation time */
    double time_forward_reshape;               /**< Forward reshape time */
    double time_backward;                      /**< Backward computation time */
    double time_mse;                           /**< MSE computation time */
    double time_syrk;                          /**< SYRK operation time */

    double flops_forward;                      /**< Floating point operations for forward */
    double perf_forward;                       /**< Forward computation performance (GFlops) */
    double flops_backward;                     /**< Floating point operations for backward */
    double perf_backward;                      /**< Backward computation performance (GFlops) */

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    climate_emulator_workspace_t *ws;          /**< GPU workspace */
#endif

} climate_emulator_struct_t;

/* ============================================================================
 * Function declarations
 * ============================================================================ */

/* ============================================================================
 * Initialization and cleanup
 * ============================================================================ */

/**
 * @brief Initialize climate emulator
 * 
 * Sets up the PaRSEC context and configures default parameters for climate
 * emulation including band sizes and Cholesky decomposition type.
 * 
 * This function initializes the PaRSEC runtime environment and sets up
 * default configuration parameters for optimal climate computation performance.
 * 
 * @param[in] argc Number of command line arguments
 * @param[in] argv Command line arguments array
 * @param[in] params HICMA PaRSEC parameters
 * @param[in] params_kernel STARSH kernel parameters
 * @param[in] data HICMA PaRSEC data structure
 * @return Pointer to initialized PaRSEC context, or NULL on failure
 */
parsec_context_t *climate_emulator_init( int argc, char ** argv,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_data_t *data);

/**
 * @brief Read climate emulator data
 * 
 * Reads all input matrices and data files needed for climate emulation,
 * including transformation matrices and climate data. Sets up matrix
 * descriptors and allocates memory for computations.
 * 
 * This function performs the following operations:
 * - Reads transformation matrices (Et1, Et2, Ep, Slmn, Ie, Io, P, D)
 * - Reads climate data (f_data)
 * - Sets up matrix descriptors with proper dimensions
 * - Allocates memory for computation buffers
 * - Initializes GPU resources if enabled
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] gb Climate emulator structure
 * @param[in] params HICMA PaRSEC parameters
 */
void climate_emulator_reading_data(parsec_context_t *parsec, climate_emulator_struct_t *gb, hicma_parsec_params_t* params);

/**
 * @brief Finalize climate emulator
 * 
 * Cleans up allocated memory, destroys matrix descriptors, and finalizes
 * GPU resources if they were initialized.
 * 
 * This function ensures proper cleanup of all resources to prevent memory
 * leaks and maintain system stability.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] gb Climate emulator structure
 * @param[in] params HICMA PaRSEC parameters
 */
void climate_emulator_fini(parsec_context_t *parsec, climate_emulator_struct_t *gb, hicma_parsec_params_t* params);

/* ============================================================================
 * Utility functions
 * ============================================================================ */

/**
 * @brief Print matrix column in double precision
 * 
 * Debug function to print a matrix in column-major format with double precision.
 * Useful for debugging and validation of matrix data during computation.
 * 
 * @param[in] data Matrix data
 * @param[in] M Number of rows
 * @param[in] N Number of columns
 * @param[in] lda Leading dimension
 */
void climate_emulator_print_matrix_col_double(double *data, int M, int N, int lda);

/**
 * @brief Print matrix column in complex precision
 * 
 * Debug function to print a matrix in column-major format with complex precision.
 * Useful for debugging and validation of complex matrix data during computation.
 * 
 * @param[in] data Matrix data
 * @param[in] M Number of rows
 * @param[in] N Number of columns
 * @param[in] lda Leading dimension
 */
void climate_emulator_print_matrix_col_complex(complex double *data, int M, int N, int lda);

/* ============================================================================
 * CSV reading functions
 * ============================================================================ */

/**
 * @brief Read CSV file with complex data
 * 
 * Reads a CSV file containing complex numbers and loads them into a matrix
 * descriptor. The data is distributed across MPI processes according to
 * the block-cyclic distribution.
 * 
 * This function handles complex data parsing and matrix distribution
 * for parallel climate computations.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] filename Input filename
 * @param[in] desc Matrix descriptor to populate
 * @param[in] MB Block size for rows
 * @param[in] NB Block size for columns
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int climate_emulator_read_csv_complex(parsec_context_t *parsec,
        char *filename,
        parsec_matrix_block_cyclic_t *desc,
        int MB,
        int NB,
        hicma_parsec_params_t *params);

/**
 * @brief Read CSV file with double precision data
 * 
 * Reads a CSV file containing double precision numbers and loads them into
 * a matrix descriptor with block-cyclic distribution.
 * 
 * This function handles double precision data parsing and matrix distribution
 * for parallel climate computations.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] filename Input filename
 * @param[in] desc Matrix descriptor to populate
 * @param[in] MB Block size for rows
 * @param[in] NB Block size for columns
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int climate_emulator_read_csv_double(parsec_context_t *parsec,
        char *filename,
        parsec_matrix_block_cyclic_t *desc,
        int MB,
        int NB,
        hicma_parsec_params_t *params);

/**
 * @brief Read CSV file with double precision data and timeslot information
 * 
 * Reads a CSV file with double precision data that includes timeslot
 * information for time-series climate data.
 * 
 * This function is specifically designed for time-series climate data
 * where each column represents a different time point.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] filename Input filename
 * @param[in] desc Matrix descriptor to populate
 * @param[in] MB Block size for rows
 * @param[in] NB Block size for columns
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int climate_emulator_read_csv_double_timeslot(parsec_context_t *parsec,
        char *filename,
        parsec_matrix_block_cyclic_t *desc,
        int MB,
        int NB,
        hicma_parsec_params_t *params);

/**
 * @brief Read CSV file with double precision data and convert to complex
 * 
 * Reads double precision data from CSV and converts it to complex format
 * for use in complex matrix operations.
 * 
 * This function is useful when input data is real but computations
 * require complex arithmetic.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] filename Input filename
 * @param[in] desc Matrix descriptor to populate
 * @param[in] MB Block size for rows
 * @param[in] NB Block size for columns
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int climate_emulator_read_csv_double2complex(parsec_context_t *parsec,
        char *filename,
        parsec_matrix_block_cyclic_t *desc,
        int MB,
        int NB,
        hicma_parsec_params_t *params);

/**
 * @brief Read CSV file with double precision data, convert to complex, and handle timeslots
 * 
 * Reads double precision data from CSV, converts to complex format, and
 * handles timeslot information for time-series climate data.
 * 
 * This function combines the functionality of complex conversion and
 * timeslot handling for time-series climate computations.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] filename Input filename
 * @param[in] desc Matrix descriptor to populate
 * @param[in] MB Block size for rows
 * @param[in] NB Block size for columns
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int climate_emulator_read_csv_double2complex_timeslot(parsec_context_t *parsec,
        char *filename,
        parsec_matrix_block_cyclic_t *desc,
        int MB,
        int NB,
        hicma_parsec_params_t *params);

/* ============================================================================
 * Forward computation functions
 * ============================================================================ */

/**
 * @brief Perform forward computation with pre-computed version
 * 
 * Executes the forward spherical harmonic transform using pre-computed
 * transformation matrices. This is the main function for converting
 * spatial climate data to spectral coefficients.
 * 
 * The forward transform implements the algorithm:
 * 1. Compute Gmtheta_r = f_data * Ep
 * 2. Compute Fmnm = Et1 * Gmtheta_r + Et2 * P * Gmtheta_r * D
 * 3. Apply spherical harmonic transformations using Slmn, Ie, and Io
 * 4. Reshape and separate real/imaginary parts into flm coefficients
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] gb Climate emulator structure
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int climate_emulator_geqsht_forward_pre_computed_version(parsec_context_t *parsec,
        climate_emulator_struct_t *gb,
        hicma_parsec_params_t *params);

/**
 * @brief Core forward computation function
 * 
 * Implements the core algorithm for forward spherical harmonic transform.
 * Performs matrix multiplications and transformations to convert spatial
 * climate data to spherical harmonic coefficients.
 * 
 * This function is the computational kernel that performs the actual
 * matrix operations for the forward transform.
 * 
 * @param[in] flm Output spherical harmonic coefficients
 * @param[in] f_data Input climate data matrix
 * @param[in] Et1 ET1 transformation matrix
 * @param[in] Et2 ET2 transformation matrix
 * @param[in] Ep EP transformation matrix
 * @param[in] Slmn SLMN transformation matrix
 * @param[in] Ie IE transformation matrix (even)
 * @param[in] Io IO transformation matrix (odd)
 * @param[in] P P transformation matrix
 * @param[in] D D transformation matrix
 * @param[in] Gmtheta_r Temporary buffer for Gmtheta_r
 * @param[in] Fmnm Temporary buffer for Fmnm
 * @param[in] tmp1 Temporary buffer 1
 * @param[in] tmp2 Temporary buffer 2
 * @param[in] gb Climate emulator structure
 */
void climate_emulator_geqsht_forward_pre_computed_version_core(
		double *flm,
        complex double *f_data,
        complex double *Et1,
        complex double *Et2,
        complex double *Ep,
        complex double *Slmn,
        complex double *Ie,
        complex double *Io,
        complex double *P,
        complex double *D,
        complex double *Gmtheta_r,
        complex double *Fmnm,
        complex double *tmp1,
        complex double *tmp2,
        climate_emulator_struct_t *gb);

/**
 * @brief Reshape forward computation data
 * 
 * Reshapes the output data from forward computation to prepare for
 * further processing or output.
 * 
 * This function reorganizes the computed spherical harmonic coefficients
 * into the proper format for storage or further computation.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] gb Climate emulator structure
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int climate_emulator_geqsht_forward_reshape(parsec_context_t *parsec,
        climate_emulator_struct_t *gb,
        hicma_parsec_params_t *params);

/* ============================================================================
 * Utility computation functions
 * ============================================================================ */

/**
 * @brief Compute difference between double precision matrices
 * 
 * Calculates the element-wise difference between two matrices and
 * can be used for validation or error checking.
 * 
 * This function is useful for comparing computed results with
 * reference solutions or for convergence testing.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] descA Matrix descriptor A
 * @param[in] descB Matrix descriptor B
 * @return 0 on success, non-zero on failure
 */
int climate_emulator_diff_double(parsec_context_t *parsec,
        parsec_matrix_block_cyclic_t *descA,
        parsec_matrix_block_cyclic_t *descB);

/**
 * @brief Get single index from n and m values
 * 
 * Converts spherical harmonic indices (n,m) to a single linear index
 * for accessing triangular matrices. This is a utility function for
 * spherical harmonic coefficient indexing.
 * 
 * The function implements the standard triangular indexing scheme:
 * index = n * (n + 1) / 2 + m
 * 
 * @param[in] n Spherical harmonic degree n
 * @param[in] m Spherical harmonic order m
 * @return Single index value for triangular matrix access
 */
int climate_emulator_getSingleIndex(int n, int m);

/* ============================================================================
 * Inverse computation functions
 * ============================================================================ */

/**
 * @brief Perform inverse computation with pre-computed version
 * 
 * Executes the inverse spherical harmonic transform using pre-computed
 * transformation matrices. This converts spectral coefficients back
 * to spatial climate data.
 * 
 * The inverse transform implements the algorithm:
 * 1. Compute Smt matrix by summing over spherical harmonics
 * 2. Apply final transformation: f_spatial = Smt * SC
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] gb Climate emulator structure
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int climate_emulator_geqsht_inverse_pre_computed_version(parsec_context_t *parsec,
        climate_emulator_struct_t *gb,
        hicma_parsec_params_t *params);

/**
 * @brief Core inverse computation function
 * 
 * Implements the core algorithm for inverse spherical harmonic transform.
 * Performs matrix multiplications to convert spherical harmonic
 * coefficients back to spatial climate data.
 * 
 * This function is the computational kernel that performs the actual
 * matrix operations for the inverse transform.
 * 
 * @param[in] flm Input spherical harmonic coefficients
 * @param[in] f_spatial Output spatial climate data
 * @param[in] Zlm ZLM transformation matrix
 * @param[in] SC SC transformation matrix
 * @param[in] Smt Temporary matrix for intermediate results
 * @param[in] gb Climate emulator structure
 */
void climate_emulator_geqsht_inverse_pre_computed_version_core(
        double *flm,
        double *f_spatial,
        double *Zlm,
        double *SC,
        double *Smt,
        climate_emulator_struct_t *gb);

/* ============================================================================
 * Analysis functions
 * ============================================================================ */

/**
 * @brief Compute mean squared error
 * 
 * Calculates the mean squared error between computed and reference
 * spherical harmonic coefficients for validation purposes.
 * 
 * This function is essential for validating the accuracy of the
 * climate emulation computations.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] gb Climate emulator structure
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int climate_emulator_mse(parsec_context_t *parsec,
        climate_emulator_struct_t *gb,
        hicma_parsec_params_t *params);

/**
 * @brief Core function to sum double precision matrix
 * 
 * Utility function to compute the sum of all elements in a double
 * precision matrix. Used for debugging and validation.
 * 
 * @param[in] A Matrix data
 * @param[in] M Number of rows
 * @param[in] N Number of columns
 * @return Sum of all matrix elements
 */
double climate_emulator_sum_double_core(double *A, int M, int N);

/**
 * @brief Core function to sum complex matrix
 * 
 * Utility function to compute the sum of all elements in a complex
 * matrix. Used for debugging and validation.
 * 
 * @param[in] A Matrix data
 * @param[in] M Number of rows
 * @param[in] N Number of columns
 * @return Sum of all matrix elements
 */
complex double climate_emulator_sum_complex_core(complex double *A, int M, int N);

/* ============================================================================
 * I/O functions
 * ============================================================================ */

/**
 * @brief Write climate emulator data
 * 
 * Writes matrix data to a file in either MPI-IO or binary format
 * depending on configuration. Supports both symmetric and general matrices.
 * 
 * This function handles the conversion from PaRSEC tiled format to
 * Lapack format before writing, ensuring compatibility with external tools.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] filename Output filename
 * @param[in] A Matrix to write
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 * @param[in] symm Symmetry flag (1 for symmetric, 0 for general)
 */
void climate_emulator_write(parsec_context_t *parsec,
        char *filename,
        parsec_tiled_matrix_t *A,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        int symm);

/**
 * @brief Read climate emulator data
 * 
 * Reads matrix data from a file using MPI-IO. The data is distributed
 * across processes according to the matrix descriptor.
 * 
 * This function creates MPI derived datatypes for proper data distribution
 * and converts from Lapack format to PaRSEC tiled format.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] filename Input filename
 * @param[in] A Matrix to read into
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 * @param[in] symm Symmetry flag (1 for symmetric, 0 for general)
 */
void climate_emulator_read(parsec_context_t *parsec,
        char *filename,
        parsec_tiled_matrix_t *A,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        int symm);

/**
 * @brief Compute the Frobenius norm of a (triangular) matrix
 *
 * This function computes the Frobenius norm of a triangular matrix by:
 * 1. Creating a taskpool to compute norms of individual tiles in parallel
 * 2. Accumulating per-thread norm contributions
 * 3. Performing MPI reduction to get the global norm across all processes
 * 4. Add noise to diagonal if set
 *
 * @param [in] parsec:       PaRSEC context for task scheduling
 * @param [in] uplo:         Specifies upper or lower triangular part
 * @param [in] A:            The tiled matrix descriptor
 * @param [inout] params_tlr: TLR parameters (norm_global and norm_tile will be updated)
 * @param [in] datatype_str: String specifying the data type for computation
 * @return The computed Frobenius norm of the matrix
 */
double climate_emulator_matrix_norm_get( parsec_context_t *parsec,
        dplasma_enum_t uplo,
        parsec_tiled_matrix_t *A,
        hicma_parsec_params_t *params_tlr,
        char *datatype_str);

/* ============================================================================
 * GPU functions
 * ============================================================================ */
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

/**
 * @brief Initialize GPU for climate emulator
 * 
 * Sets up GPU resources including cuBLAS handles, memory allocation,
 * and stream management for GPU-accelerated climate computations.
 * 
 * This function initializes all available GPU devices and creates
 * execution streams with associated cuBLAS handles and memory buffers.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] gb Climate emulator structure
 * @param[in] params HICMA PaRSEC parameters
 */
void climate_emulator_gpu_init(parsec_context_t *parsec, climate_emulator_struct_t *gb, hicma_parsec_params_t *params);

/**
 * @brief Finalize GPU for climate emulator
 * 
 * Cleans up GPU resources including cuBLAS handles and allocated
 * memory buffers.
 * 
 * This function should be called before program termination to prevent
 * memory leaks and ensure proper GPU resource cleanup.
 * 
 * @param[in] gb Climate emulator structure
 */
void climate_emulator_gpu_fini(climate_emulator_struct_t *gb);

/**
 * @brief GPU core forward computation function
 * 
 * GPU-accelerated version of the forward spherical harmonic transform.
 * Performs matrix operations on GPU using cuBLAS for improved performance.
 * 
 * The algorithm follows the same steps as the CPU version:
 * 1. Compute Gmtheta_r = f_data * Ep
 * 2. Compute Fmnm = Et1 * Gmtheta_r + Et2 * P * Gmtheta_r * D
 * 3. Apply spherical harmonic transformations using Slmn, Ie, and Io
 * 4. Reshape and separate real/imaginary parts into flm coefficients
 * 
 * @param[in] flm Output spherical harmonic coefficients
 * @param[in] f_data Input climate data matrix
 * @param[in] Et1 ET1 transformation matrix
 * @param[in] Et2 ET2 transformation matrix
 * @param[in] Ep EP transformation matrix
 * @param[in] Slmn SLMN transformation matrix
 * @param[in] Ie IE transformation matrix (even)
 * @param[in] Io IO transformation matrix (odd)
 * @param[in] P P transformation matrix
 * @param[in] D D transformation matrix
 * @param[in] cuda_device CUDA device module
 * @param[in] gpu_task GPU task
 * @param[in] cuda_stream CUDA execution stream
 * @param[in] gb Climate emulator structure
 */
void climate_emulator_geqsht_forward_pre_computed_version_gpu_core(
        double *flm,
        cuDoubleComplex *f_data,
        cuDoubleComplex *Et1,
        cuDoubleComplex *Et2,
        cuDoubleComplex *Ep,
        cuDoubleComplex *Slmn,
        cuDoubleComplex *Ie,
        cuDoubleComplex *Io,
        cuDoubleComplex *P,
        cuDoubleComplex *D,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        climate_emulator_struct_t *gb);

/**
 * @brief Reshape data on GPU
 * 
 * GPU-accelerated data reshaping operation for climate data processing.
 * 
 * This function converts complex matrix data to real format by separating
 * real and imaginary parts, optimized for GPU execution.
 * 
 * @param[in] T Target data array
 * @param[in] S Source data array
 * @param[in] L Size parameter
 * @param[in] ldaS Leading dimension of source
 * @param[in] stream CUDA stream for execution
 */
void climate_emulator_reshape_GPU( double *T, cuDoubleComplex *S,
        int L, int ldaS,
        cudaStream_t stream );

/**
 * @brief Print complex matrix on GPU
 * 
 * Debug function to print complex matrix data from GPU memory.
 * 
 * This function transfers data from GPU to CPU for printing, useful
 * for debugging GPU computations.
 * 
 * @param[in] A Matrix data on GPU
 * @param[in] M Number of rows
 * @param[in] N Number of columns
 * @param[in] lda Leading dimension
 * @param[in] stream CUDA stream for execution
 */
void climate_emulator_print_complex_GPU( cuDoubleComplex *A,
        int M, int N, int lda,
        cudaStream_t stream );

/**
 * @brief Print double matrix on GPU
 * 
 * Debug function to print double matrix data from GPU memory.
 * 
 * This function transfers data from GPU to CPU for printing, useful
 * for debugging GPU computations.
 * 
 * @param[in] A Matrix data on GPU
 * @param[in] M Number of rows
 * @param[in] N Number of columns
 * @param[in] lda Leading dimension
 * @param[in] stream CUDA stream for execution
 */
void climate_emulator_print_double_GPU( double *A,
        int M, int N, int lda,
        cudaStream_t stream );

/**
 * @brief GPU core inverse computation function
 * 
 * GPU-accelerated version of the inverse spherical harmonic transform.
 * Converts spherical harmonic coefficients back to spatial climate data.
 * 
 * This function implements the same algorithm as the CPU version but
 * optimized for GPU execution using cuBLAS operations.
 * 
 * @param[in] n Tile index
 * @param[in] flm Input spherical harmonic coefficients
 * @param[in] f_spatial Output spatial climate data
 * @param[in] Zlm ZLM transformation matrix
 * @param[in] SC SC transformation matrix
 * @param[in] cuda_device CUDA device module
 * @param[in] gpu_task GPU task
 * @param[in] cuda_stream CUDA execution stream
 * @param[in] gb Climate emulator structure
 */
void climate_emulator_geqsht_inverse_pre_computed_version_gpu_core(
        int n,
        double *flm,
        double *f_spatial,
        double *Zlm,
        double *SC,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        climate_emulator_struct_t *gb);

/**
 * @brief Calculate GPU load balance
 * 
 * Utility function to determine load balancing across GPU devices
 * for parallel climate computations.
 * 
 * This function implements a simple round-robin distribution strategy
 * to balance computational load across available GPU devices.
 * 
 * @param[in] n Work index
 * @param[in] nodes Number of compute nodes
 * @param[in] nb_cuda_devices Number of CUDA devices
 * @return Load balance factor for GPU assignment
 */
static inline int climate_emulator_gpu_load_balance(int n, int nodes, int nb_cuda_devices ) {
    return n / nodes % nb_cuda_devices; 
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* HICMA_PARSEC_CLIMATE_EMULATOR_H */

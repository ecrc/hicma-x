/**
 * @file hicma_parsec.h
 * @brief Main header file for HICMA PaRSEC library
 * 
 * This is the primary header file that provides access to all HICMA PaRSEC functionality.
 * It includes all component headers and declares the main public API functions for:
 * - Matrix generation and manipulation
 * - Cholesky factorization (dense, mixed-precision, and TLR)
 * - Matrix analysis and compression
 * - I/O operations
 * - GPU acceleration support
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 * 
 * @version 1.0.0
 **/

#ifndef HICMA_PARSEC_H
#define HICMA_PARSEC_H

/* ============================================================================
 * System includes
 * ============================================================================ */
#include <stdbool.h>

/* ============================================================================
 * HICMA PaRSEC component headers
 * ============================================================================ */
#include "hicma_parsec_internal.h"
#include "hicma_parsec_flat_file.h"
#include "hicma_parsec_datatype_convert.h"
#include "hicma_parsec_decision.h"
#include "hicma_parsec_core.h"
#include "hicma_parsec_sparse_analysis.h"
#include "hicma_parsec_rank_statistics.h"
#include "hicma_parsec_io.h"
#include "hicma_parsec_gpu.h"
#include "hicma_parsec_distribution.h"
#include "hicma_parsec_check.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Library initialization and finalization
 * ============================================================================ */

/**
 * @brief Initialize HICMA PaRSEC library
 * 
 * Initializes the PaRSEC context and sets up the HICMA library for computation.
 * This function must be called before any other HICMA functions.
 * 
 * @param[in] argc Number of command line arguments
 * @param[in] argv Command line arguments array
 * @param[in] params HICMA PaRSEC parameters structure
 * @param[in] params_kernel STARSH kernel parameters
 * @param[in] data HICMA PaRSEC data structure
 * @return Pointer to initialized PaRSEC context, or NULL on failure
 */
parsec_context_t * hicma_parsec_init( int argc, char ** argv,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_data_t *data);

/**
 * @brief Finalize HICMA PaRSEC library
 * 
 * Cleans up resources and finalizes the HICMA library. This function should
 * be called when the library is no longer needed.
 * 
 * @param[in] parsec PaRSEC context to finalize
 * @param[in] argc Number of command line arguments
 * @param[in] argv Command line arguments array
 * @param[in] params HICMA PaRSEC parameters structure
 * @param[in] params_kernel STARSH kernel parameters
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] analysis Matrix analysis structure
 */
void hicma_parsec_fini( parsec_context_t* parsec,
        int argc, char ** argv,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_data_t *data,
        hicma_parsec_matrix_analysis_t *analysis);

/* ============================================================================
 * Matrix generation and manipulation
 * ============================================================================ */

/**
 * @brief Generate matrix data for computation
 * 
 * Generates matrix data based on the specified parameters and kernel configuration.
 * This function creates the initial matrix structure for subsequent operations.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] data_tlr HICMA PaRSEC data structure
 * @param[in] params_tlr HICMA PaRSEC parameters
 * @param[in] params_kernel STARSH kernel parameters
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_matrix_generation( parsec_context_t *parsec,
        hicma_parsec_data_t *data_tlr,
        hicma_parsec_params_t *params_tlr,
        starsh_params_t *params_kernel );

/* ============================================================================
 * Matrix analysis
 * ============================================================================ */

/**
 * @brief Analyze matrix before Cholesky factorization
 * 
 * Performs pre-analysis of the matrix to determine optimal computation strategies
 * and prepare for Cholesky factorization.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 * @param[in] params_kernel STARSH kernel parameters
 * @param[in] analysis Matrix analysis structure to populate
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_matrix_pre_analysis( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_matrix_analysis_t *analysis );

/**
 * @brief Analyze matrix after Cholesky factorization
 * 
 * Performs post-analysis of the matrix to evaluate factorization quality
 * and gather performance metrics.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 * @param[in] params_kernel STARSH kernel parameters
 * @param[in] analysis Matrix analysis structure
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_matrix_post_analysis( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_matrix_analysis_t *analysis );

/* ============================================================================
 * Cholesky factorization
 * ============================================================================ */

/**
 * @brief Main routine for mixed-precision dense/TLR Cholesky factorization
 * 
 * Performs Cholesky factorization of a symmetric positive definite matrix
 * using mixed-precision arithmetic and TLR (Tile Low-Rank) representation.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 * @param[in] analysis Matrix analysis structure
 * @return info: 0 on all nodes if successful, > 0 if the leading minor of order i 
 *         of A is not positive definite, so the factorization could not be completed,
 *         and the solution has not been computed. Info will be equal to i on the
 *         node that owns the diagonal element (i,i), and 0 on all other nodes
 */
int hicma_parsec_potrf( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t *analysis );

/* ============================================================================
 * Kernel matrix operations
 * ============================================================================ */

/**
 * @brief Process kernel matrix operations
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] AA Tiled matrix descriptor
 * @param[in] params HICMA PaRSEC parameters
 */
void hicma_kernal_matrix(parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        parsec_tiled_matrix_t *AA,
        hicma_parsec_params_t *params);

/**
 * @brief Process kernel matrix operations with parameters
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 */
void hicma_parsec_kernel_matrix(parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params);

/**
 * @brief Process kernel matrix operations from file
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 */
void hicma_parsec_kernel_matrix_file(parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params);

/* ============================================================================
 * Matrix I/O operations
 * ============================================================================ */

/**
 * @brief Write matrix A to file
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] filename Output filename
 * @param[in] A Matrix to write
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 * @param[in] symm Symmetry flag
 */
void hicma_parsec_writeA(parsec_context_t *parsec,
        const char *filename,
        parsec_tiled_matrix_t *A,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        int symm);

/**
 * @brief Write matrix B to file
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] filename Output filename
 * @param[in] A Matrix to write
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 * @param[in] symm Symmetry flag
 */
void hicma_parsec_writeB(parsec_context_t *parsec,
        const char *filename,
        parsec_tiled_matrix_t *A,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        int symm);

/* ============================================================================
 * Matrix operations
 * ============================================================================ */

/**
 * @brief Perform GEMM (General Matrix Multiply) operation
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 */
void perform_gemm(parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params);        

/**
 * @brief Calculate the Frobenius norm of a tiled matrix
 * 
 * This function calculates the Frobenius norm of a tiled matrix using PaRSEC task-based parallelism.
 * The function supports various datatypes and handles both symmetric and general matrices.
 * The norm is computed as sqrt(sum of squares of all matrix elements).
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution
 * @param[in] uplo Specifies which part of the matrix to use:
 *                 - dplasmaUpper: Use upper triangular part (for symmetric matrices)
 *                 - dplasmaLower: Use lower triangular part (for symmetric matrices)
 *                 - dplasmaGeneral: Use entire matrix (for general matrices)
 * @param[in] A Pointer to the tiled matrix descriptor
 * @param[in] params_tlr HICMA PaRSEC parameters containing matrix information and norm storage
 * @param[in] datatype_str String representation of the matrix datatype. Supported values:
 *                         - "double", "d" -> double precision floating point
 *                         - "float", "single", "s" -> single precision floating point
 *                         - "int8", "i8" -> 8-bit signed integer
 *                         - "int16", "i16" -> 16-bit signed integer
 *                         - "int32", "int", "i32", "i" -> 32-bit signed integer
 *                         - "int64", "i64" -> 64-bit signed integer
 *                         - "uint8", "u8" -> 8-bit unsigned integer
 *                         - "uint16", "u16" -> 16-bit unsigned integer
 *                         - "uint32", "uint", "u32", "u" -> 32-bit unsigned integer
 *                         - "uint64", "u64" -> 64-bit unsigned integer
 *                         - "half", "fp16", "h" -> 16-bit floating point (half precision)
 *                         - "fp8" -> 8-bit floating point representation
 *                         - "fp4" -> 4-bit floating point representation
 *                         - "int4" -> 4-bit integer representation
 *                         - "1bit" -> 1-bit representation
 * 
 * @return The calculated Frobenius norm of the matrix
 * 
 * @note This function uses PaRSEC task-based parallelism to compute the norm efficiently.
 *       The result is stored in params_tlr->norm_global and individual tile norms are stored
 *       in params_tlr->norm_tile array.
 * 
 * @note For symmetric matrices, only the specified triangular part is used in the computation.
 *       For general matrices, all elements are included regardless of the uplo parameter.
 * 
 * @note The function assumes column-major storage layout for the matrix data.
 * 
 * @see hicma_parsec_core_matrix_norm_get for the core norm computation function
 */
double hicma_parsec_matrix_norm_get( parsec_context_t *parsec,
        dplasma_enum_t uplo,
        parsec_tiled_matrix_t *A,
        hicma_parsec_params_t *params_tlr,
        double *norm_tile,
        double *norm_global,
        char *datatype_str);

/**
 * @brief Triangular matrix-matrix multiply (TRMM) for LLN path.
 *
 * Current implementation supports Left/Lower/NoTrans with Unit or NonUnit diagonal.
 *
 * @param[in] parsec PaRSEC context
 * @param[in] side Matrix side (must be dplasmaLeft)
 * @param[in] uplo Triangle selection (must be dplasmaLower)
 * @param[in] trans Transpose mode (must be dplasmaNoTrans)
 * @param[in] diag Diagonal type (dplasmaUnit or dplasmaNonUnit)
 * @param[in] alpha Scalar multiplier
 * @param[in] A Triangular matrix descriptor
 * @param[in,out] B Right-hand side matrix descriptor
 * @return 0 on success, non-zero on error
 */
int hicma_parsec_trmm(parsec_context_t *parsec,
                      dplasma_enum_t side, dplasma_enum_t uplo,
                      dplasma_enum_t trans, dplasma_enum_t diag,
                      double alpha,
                      const parsec_tiled_matrix_t *A,
                      parsec_tiled_matrix_t *B,
                      hicma_parsec_data_t *data,
                      hicma_parsec_params_t *params);

#ifdef __cplusplus
}
#endif

#endif /* HICMA_PARSEC_H */

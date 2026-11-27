/**
 * @file hicma_parsec_check.h
 * @brief HICMA PaRSEC correctness checking header file
 * 
 * This header file contains declarations for correctness checking functions including
 * matrix generation verification, Cholesky factorization validation, compression
 * quality assessment, and analysis verification for the HICMA library.
 * 
 * The file provides comprehensive testing and validation capabilities for:
 * - Matrix generation and verification
 * - Cholesky factorization correctness
 * - Compression quality assessment
 * - Analysis result validation
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2023-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 * 
 * @version 0.2.0
 */

#ifndef HICMA_PARSEC_CHECK_H
#define HICMA_PARSEC_CHECK_H

/* ============================================================================
 * HICMA PaRSEC includes
 * ============================================================================ */
#include "hicma_parsec_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Matrix generation and verification functions
 * ============================================================================ */

/**
 * @brief Kernel operator for generating matrix tiles
 * 
 * Generates matrix data using the STARSH kernel operator for matrix generation.
 * This function is called by the PaRSEC runtime to generate matrix tiles
 * during computation. It handles the generation of individual tiles based on
 * the specified kernel function and parameters.
 * 
 * @param[in] es Execution stream for PaRSEC runtime
 * @param[in] descA Matrix descriptor containing matrix metadata
 * @param[in,out] _A Pointer to matrix tile data to be generated
 * @param[in] uplo Upper/lower specification (PlasmaUpper, PlasmaLower, PlasmaUpperLower)
 * @param[in] m Row index of the tile to generate
 * @param[in] n Column index of the tile to generate
 * @param[in] op_data Operation data containing kernel parameters
 * @return 0 on success, non-zero on failure
 * 
 * @note This function is designed to work with PaRSEC's task-based execution model
 * @note The kernel function is called with appropriate tile dimensions and offsets
 */
int starsh_generate_map_operator(parsec_execution_stream_t *es,
        const parsec_tiled_matrix_t *descA, void *_A, int uplo,
        int m, int n, void *op_data);

/**
 * @brief Generate complete matrix using STARSH kernel
 * 
 * Generates a complete matrix using the STARSH kernel for matrix generation.
 * This function creates a task pool to generate all matrix tiles in parallel
 * using the specified kernel function and parameters.
 * 
 * @param[in] parsec PaRSEC context for task execution
 * @param[in] uplo Upper/lower specification for matrix generation
 * @param[in,out] A Matrix to be generated (will be filled with generated data)
 * @param[in] params STARSH parameters containing kernel function and data
 * @return 0 on success, non-zero on failure
 * 
 * @note The function creates a copy of the parameters to ensure thread safety
 * @note Matrix generation is performed using PaRSEC's apply pattern
 */
int starsh_generate_map(parsec_context_t *parsec, int uplo,
        parsec_tiled_matrix_t *A, starsh_params_t *params);

/* ============================================================================
 * Factorization correctness checking functions
 * ============================================================================ */

/**
 * @brief Check Cholesky factorization correctness via ||L*L'-A||
 * 
 * Verifies the correctness of Cholesky factorization by computing L*L' and
 * comparing it with the original matrix A. The function computes the relative
 * error ||L*L'-A||/||A|| and compares it against a specified threshold.
 * 
 * @param[in] parsec PaRSEC context for matrix operations
 * @param[in] verbose Verbosity level for output (0 = silent, 1 = verbose)
 * @param[in] uplo Upper/lower specification (PlasmaUpper, PlasmaLower)
 * @param[in] A Computed L matrix from Cholesky factorization
 * @param[in] A0 Original matrix A before factorization
 * @param[in] threshold Accuracy threshold for correctness validation
 * @param[out] result_accuracy Computed relative error ||L*L'-A||/||A||
 * @return 0 on success, non-zero on failure
 * 
 * @note The function performs the computation L*L' - A and computes Frobenius norms
 * @note Results are printed if verbose mode is enabled
 * @note Factorization is considered correct if relative error < threshold
 */
int check_dpotrf2( parsec_context_t *parsec, int verbose,
        int uplo,
        parsec_tiled_matrix_t *A,
        parsec_tiled_matrix_t *A0, double threshold, double *result_accuracy );

/**
 * @brief Check solution L difference from the dense counterpart
 * 
 * Compares the computed L matrix with the original A matrix to verify
 * factorization accuracy. This function computes the difference between
 * the computed L and the reference L0, providing a direct comparison
 * of the factorization results.
 * 
 * @param[in] parsec PaRSEC context for matrix operations
 * @param[in] verbose Verbosity level for output (0 = silent, 1 = verbose)
 * @param[in] uplo Upper/lower specification (PlasmaUpper, PlasmaLower)
 * @param[in] A Computed L matrix from factorization
 * @param[in] A0 Reference L matrix for comparison
 * @param[in] threshold Accuracy threshold for correctness validation
 * @return 0 on success, non-zero on failure
 * 
 * @note The function computes ||L-L0||/||L|| and compares against threshold
 * @note Results are printed if verbose mode is enabled
 * @note Matrices are considered equal if relative difference < threshold
 */
int check_diff( parsec_context_t *parsec, int verbose,
                  int uplo,
                  parsec_tiled_matrix_t *A,
                  parsec_tiled_matrix_t *A0, double threshold );

/* ============================================================================
 * HICMA-specific checking functions
 * ============================================================================ */

/**
 * @brief Check compression quality and perform dense Cholesky validation
 * 
 * Evaluates the quality of matrix compression by analyzing compression ratios
 * and accuracy metrics. The function performs several validation steps:
 * 1. Generates the original problem matrix
 * 2. Uncompresses the approximate matrix
 * 3. Computes the difference between original and uncompressed matrices
 * 4. Performs dense Cholesky factorization on the uncompressed matrix
 * 5. Computes determinant for validation
 * 
 * @param[in] parsec PaRSEC context for matrix operations
 * @param[in] data HICMA data structure containing matrices and parameters
 * @param[in] params HICMA parameters for computation
 * @param[in] params_kernel STARSH kernel parameters for matrix generation
 * @param[in] analysis Matrix analysis structure with rank information
 * @return 0 on success, non-zero on failure
 * 
 * @note This function is only executed when params->check is non-zero
 * @note The function handles both band memory and general memory layouts
 * @note Dense Cholesky is performed to validate the uncompressed matrix
 */
int hicma_parsec_check_compression( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_matrix_analysis_t *analysis );

/**
 * @brief Check Cholesky factorization results comprehensively
 * 
 * Performs comprehensive verification of Cholesky factorization results including
 * accuracy, numerical stability, and correctness. The function validates:
 * 1. Difference between HiCMA and dense Cholesky results
 * 2. Factorization correctness via ||L*L'-A|| computation
 * 3. Numerical accuracy against specified thresholds
 * 
 * @param[in] parsec PaRSEC context for matrix operations
 * @param[in] data HICMA data structure containing matrices and parameters
 * @param[in] params HICMA parameters for computation
 * @param[in] params_kernel STARSH kernel parameters for matrix generation
 * @param[in] analysis Matrix analysis structure with rank information
 * @return 0 on success, non-zero on failure
 * 
 * @note This function is only executed when params->check is non-zero
 * @note Validation includes both direct matrix comparison and factorization checking
 * @note Results are printed with appropriate formatting and color coding
 */
int hicma_parsec_check_dpotrf( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_matrix_analysis_t *analysis );

/**
 * @brief Check correctness of matrix analysis results
 * 
 * Verifies the correctness and consistency of matrix analysis results by
 * comparing the computed ranks with the expected ranks from the rank array.
 * This function validates that the sparse analysis correctly identified
 * the matrix structure and fill-in patterns.
 * 
 * @param[in] params HICMA parameters containing rank array and analysis settings
 * @param[in] analysis Matrix analysis structure with computed ranks
 * @return 0 on success, non-zero on failure
 * 
 * @note This function is only executed when params->sparse is non-zero
 * @note Validation is performed across all MPI ranks with collective verification
 * @note Results are reported with appropriate success/failure indicators
 */
int hicma_parsec_check_analysis( hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t *analysis );

#ifdef __cplusplus
}
#endif

#endif /* HICMA_PARSEC_CHECK_H */

/**
 * @file hicma_parsec_sparse_analysis.h
 * @brief HICMA PaRSEC sparse matrix analysis header file
 * 
 * This header file contains declarations for sparse matrix analysis functions including
 * initialization, memory management, TRSM/SYRK analysis, GEMM analysis, and distribution
 * calculations for the HICMA library. These functions are essential for optimizing
 * workload distribution and memory allocation in sparse hierarchical matrix computations,
 * particularly for Cholesky factorization algorithms.
 * 
 * The analysis framework provides:
 * - Workload balancing across processes based on matrix sparsity patterns
 * - Memory optimization through intelligent tile distribution
 * - Performance prediction and optimization for TRSM, SYRK, and GEMM operations
 * - Adaptive rank management for hierarchical low-rank structures
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 * 
 * @version 0.2.0
 */

#ifndef HICMA_PARSEC_SPARSE_ANALYSIS_H
#define HICMA_PARSEC_SPARSE_ANALYSIS_H

/* ============================================================================
 * HICMA PaRSEC includes
 * ============================================================================ */
#include "hicma_parsec_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Initialization and cleanup functions
 * ============================================================================ */

/**
 * @brief Initialize sparse analysis structure
 * 
 * Allocates and initializes the matrix analysis structure with the necessary
 * memory and data structures for sparse matrix analysis. This function sets up
 * the foundation for workload analysis and distribution calculations.
 * 
 * @param[in,out] analysis Matrix analysis structure to initialize
 * @param[in] NT Number of tiles in each dimension of the matrix
 * @param[in] rank Process rank (MPI rank) for this process
 * @return 0 on success, non-zero on failure
 * 
 * @note This function must be called before any other analysis functions
 * @note Memory allocated here must be freed with corresponding cleanup functions
 */
int hicma_parsec_sparse_analysis_init( hicma_parsec_matrix_analysis_t *analysis, int NT, int rank);

/**
 * @brief Free initial and final rank memory for TRSM and SYRK analysis
 * 
 * Frees memory allocated for TRSM and SYRK analysis structures, including
 * initial and final rank arrays. This function should be called to clean up
 * resources after sparse analysis is complete to prevent memory leaks.
 * 
 * @param[in] analysis Matrix analysis structure containing TRSM/SYRK data
 * @param[in] NT Number of tiles in each dimension of the matrix
 * 
 * @note This function is safe to call multiple times
 * @note Should be called after hicma_parsec_sparse_analysis_trsm_syrk_init()
 */
void hicma_parsec_sparse_analysis_trsm_syrk_free(
        hicma_parsec_matrix_analysis_t *analysis,
        unsigned long int NT);

/**
 * @brief Initialize initial and final rank arrays for TRSM and SYRK analysis
 * 
 * Sets up the initial and final rank arrays used in TRSM and SYRK operations
 * analysis. These arrays track the rank evolution during the factorization
 * process and are essential for workload prediction and optimization.
 * 
 * @param[in,out] analysis Matrix analysis structure to populate with rank data
 * @param[in] rank_array Array of current rank information for each tile
 * @param[in] NT Number of tiles in each dimension of the matrix
 * 
 * @note The rank_array should contain the current numerical rank of each tile
 * @note This function prepares data structures for subsequent analysis operations
 */
void hicma_parsec_sparse_analysis_trsm_syrk_init( 
        hicma_parsec_matrix_analysis_t *analysis,
        int *rank_array,
        unsigned long int NT );

/* ============================================================================
 * TRSM and SYRK analysis functions
 * ============================================================================ */

/**
 * @brief Allocate memory and perform TRSM and SYRK analysis
 * 
 * Allocates memory for TRSM and SYRK analysis structures and performs
 * workload analysis for triangular solve (TRSM) and symmetric rank-k
 * update (SYRK) operations. If the sparsity level is greater than 1,
 * this function also computes the final rank distribution for optimization.
 * 
 * @param[in] A Matrix descriptor for the input matrix
 * @param[in,out] analysis Matrix analysis structure to populate with results
 * @param[in] rank_array Array of rank information for each tile
 * @param[in] rank Process rank (MPI rank) for this process
 * @param[in] NT Number of tiles in each dimension of the matrix
 * @param[in] sparse Sparsity level (0=dense, 1=sparse, >1=highly sparse)
 * @return 0 on success, non-zero on failure
 * 
 * @note Higher sparsity levels enable more aggressive optimization strategies
 * @note Results are used for workload balancing and memory allocation decisions
 */
int hicma_parsec_sparse_analysis_trsm_syrk(  parsec_tiled_matrix_t *A,
        hicma_parsec_matrix_analysis_t *analysis,
        int *rank_array, int rank, unsigned long int NT, int sparse );

/* ============================================================================
 * GEMM analysis functions
 * ============================================================================ */

/**
 * @brief Analyze sparse matrix for GEMM operations and workload distribution
 * 
 * Performs comprehensive analysis of the sparse matrix structure for General
 * Matrix Multiply (GEMM) operations. This function analyzes the computational
 * workload distribution and optimizes the process assignment for each tile
 * based on the sparsity pattern and rank information.
 * 
 * @param[in] A Matrix descriptor for the input matrix
 * @param[in] Dist Distribution matrix descriptor for process assignment
 * @param[in,out] analysis Matrix analysis structure to populate with GEMM analysis
 * @param[in] rank_array Array of rank information; each process has this info
 * @param[in] rank Process ID (MPI rank) for this process
 * @param[in] NT Number of tiles in each dimension of the matrix
 * @param[in] sparse Sparsity level for optimization strategy selection
 * @return 0 on success, non-zero on failure
 * 
 * @note This function is crucial for load balancing in sparse matrix computations
 * @note Results influence the distribution of computational tasks across processes
 */
int hicma_parsec_sparse_analysis_gemm( parsec_tiled_matrix_t *A,
        parsec_tiled_matrix_t *Dist,
        hicma_parsec_matrix_analysis_t *analysis,
        int *rank_array, int rank, unsigned long int NT, int sparse );

/* ============================================================================
 * Memory management functions
 * ============================================================================ */

/**
 * @brief Free memory allocated for sparse analysis
 * 
 * Comprehensive cleanup function that frees all memory allocated during
 * sparse matrix analysis. This includes analysis structures, distribution
 * arrays, and any temporary data structures used during the analysis process.
 * 
 * @param[in] A Matrix descriptor (used for context during cleanup)
 * @param[in] analysis Matrix analysis structure to clean up
 * @param[in] NT Number of tiles in each dimension of the matrix
 * @param[in] rank Process rank (MPI rank) for this process
 * @param[in] sparse Sparsity level (affects which structures need cleanup)
 * @param[in] check Check flag for validation during cleanup
 * 
 * @note This function should be called after all analysis operations are complete
 * @note Safe to call multiple times; subsequent calls are no-ops
 */
void hicma_parsec_sparse_analysis_free( parsec_tiled_matrix_t *A,
        hicma_parsec_matrix_analysis_t *analysis,
        int NT, int rank, int sparse, int check);

/* ============================================================================
 * Main sparse analysis function
 * ============================================================================ */

/**
 * @brief Perform comprehensive sparse matrix analysis for workload optimization
 * 
 * This function performs a complete analysis of the sparse matrix structure
 * to optimize workload distribution and memory allocation for Cholesky factorization.
 * It includes initialization, TRSM/SYRK analysis, GEMM analysis, and distribution
 * calculations based on the sparsity level and matrix characteristics.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in,out] data HICMA data structure containing matrix descriptors
 * @param[in] params HICMA parameters including sparsity settings
 * @param[in,out] analysis Matrix analysis structure to populate
 * @return 0 on success, non-zero on failure
 * 
 * @note This function is the main entry point for sparse matrix analysis
 * @note Results are used for workload balancing and memory optimization
 */
int hicma_parsec_sparse_analysis( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t *analysis );

/* ============================================================================
 * Distribution and process management functions
 * ============================================================================ */

/**
 * @brief Redefine process ID for each tile based on sparse analysis
 * 
 * Recalculates and reassigns process IDs for each tile based on the results
 * of sparse matrix analysis. This function optimizes the distribution of
 * computational tasks across the process grid to achieve better load balancing
 * and reduce communication overhead in sparse matrix operations.
 * 
 * @param[in] A Matrix descriptor for the input matrix
 * @param[in] dist_array Distribution array containing new process assignments
 * @param[in] analysis Matrix analysis structure with workload information
 * @param[in] P Number of process rows in the process grid
 * @param[in] Q Number of process columns in the process grid
 * @param[in] band_size_dist Band size for distribution optimization
 * @param[in] rank Process rank (MPI rank) for this process
 * @param[in] NT Number of tiles in each dimension of the matrix
 * @return 0 on success, non-zero on failure
 * 
 * @note This function implements the optimized distribution strategy
 * @note Results are used to update the matrix descriptor's process mapping
 */
int hicma_parsec_sparse_dist_calculate( parsec_tiled_matrix_t *A,
        DATATYPE_ANALYSIS *dist_array, hicma_parsec_matrix_analysis_t *analysis,
        int P, int Q, int band_size_dist, int rank, int NT );

/**
 * @brief New rank_of function for symmetric two-dimensional block cyclic band distribution
 * 
 * Implements a specialized rank calculation function for symmetric matrices
 * with two-dimensional block cyclic distribution and band structure. This
 * function is optimized for sparse hierarchical matrices and provides
 * efficient process assignment based on the band structure and sparsity pattern.
 * 
 * @param[in] desc Data collection descriptor containing distribution information
 * @param[in] ... Variable arguments for rank calculation (typically tile indices)
 * @return Rank value (process ID) for the specified tile
 * 
 * @note This function replaces the standard rank_of for optimized sparse distribution
 * @note Uses the analysis results to determine optimal process assignment
 */
uint32_t hicma_parsec_sparse_balance_off_band_rank_of(parsec_data_collection_t * desc, ...);

/**
 * @brief Calculate number of local tasks for each process due to new distribution
 * 
 * Computes the number of computational tasks that will be assigned to each
 * process based on the optimized distribution strategy. This function is
 * essential for load balancing analysis and helps predict the computational
 * workload distribution across the process grid.
 * 
 * @param[in] A Matrix descriptor for the input matrix
 * @param[in] Dist Distribution matrix descriptor with process assignments
 * @param[in] analysis Matrix analysis structure with workload information
 * @param[in] rank Process rank (MPI rank) for this process
 * @param[in] NT Number of tiles in each dimension of the matrix
 * @param[in] sparse Sparsity level affecting task distribution
 * @param[out] nb_local_tasks Number of local tasks assigned to this process
 * @return 0 on success, non-zero on failure
 * 
 * @note This function is used for performance prediction and load balancing validation
 * @note Results help identify potential load imbalance issues
 */
int hicma_parsec_sparse_nb_tasks( parsec_tiled_matrix_t *A,
        parsec_tiled_matrix_t *Dist,
        hicma_parsec_matrix_analysis_t *analysis,
        int rank, int NT, int sparse, uint32_t *nb_local_tasks );

/* ============================================================================
 * Utility functions
 * ============================================================================ */

/**
 * @brief Reset the maxrank and compmaxrank for adaptive rank management
 * 
 * Resets the maximum rank values used in adaptive rank management algorithms.
 * This function is called when the rank distribution changes significantly
 * or when switching between different sparsity levels to ensure optimal
 * performance and memory usage.
 * 
 * @param[in] params HICMA PaRSEC parameters containing rank management settings
 * @return 0 on success, non-zero on failure
 * 
 * @note This function is part of the adaptive rank management system
 * @note Should be called when matrix characteristics change significantly
 */
int hicma_parsec_adaptive_maxrank( hicma_parsec_params_t *params );

/**
 * @brief Search task position in the analysis array using binary search
 * 
 * Performs a binary search on the analysis array to efficiently locate
 * a specific value. This function is used for fast lookup of task
 * positions and process assignments during sparse matrix analysis.
 * 
 * @param[in] array Analysis array to search in (must be sorted)
 * @param[in] value Value to search for in the array
 * @param[in] length Length of the array to search
 * @param[in] index1 Start index for the search range
 * @param[in] index2 End index for the search range
 * @return Index of the found value, or -1 if not found
 * 
 * @note The input array must be sorted for binary search to work correctly
 * @note This function is optimized for performance in large analysis arrays
 */
int binary_search_index( DATATYPE_ANALYSIS *array, DATATYPE_ANALYSIS value, int length, int index1, int index2 );

#ifdef __cplusplus
}
#endif

#endif /* HICMA_PARSEC_SPARSE_ANALYSIS_H */

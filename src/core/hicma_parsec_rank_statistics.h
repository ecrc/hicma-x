/**
 * @file hicma_parsec_rank_statistics.h
 * @brief HICMA PaRSEC rank statistics header file
 * 
 * This header file contains declarations for rank statistics functions including
 * rank analysis, process ID printing, and statistical computations for the HICMA library.
 * These functions are essential for analyzing the numerical rank distribution
 * in hierarchical low-rank matrix computations and understanding the parallel
 * distribution of matrix tiles across processes.
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

#ifndef HICMA_PARSEC_RANK_STATISTICS_H
#define HICMA_PARSEC_RANK_STATISTICS_H

/* ============================================================================
 * HICMA PaRSEC includes
 * ============================================================================ */
#include "hicma_parsec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Rank statistics functions
 * ============================================================================ */

/**
 * @brief Compute rank statistics for matrix
 * 
 * Analyzes the rank distribution of a matrix and computes minimum, maximum,
 * and average rank values.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] strid String identifier for the analysis
 * @param[in] dcAr Rank matrix descriptor
 * @param[in] params HICMA PaRSEC parameters
 * @param[out] minrk Minimum rank value
 * @param[out] maxrk Maximum rank value
 * @param[out] avgrk Average rank value
 */
void hicma_parsec_rank_stat(parsec_context_t* parsec, char* strid,
        parsec_tiled_matrix_t* dcAr,
        hicma_parsec_params_t *params,
        int* minrk, int* maxrk,
        double* avgrk);

/* ============================================================================
 * Process information functions
 * ============================================================================ */

/**
 * @brief Print process ID of each tile
 * 
 * Prints the process ID associated with each tile in the matrix for debugging
 * and analysis purposes. This function is useful for understanding the
 * distribution of matrix tiles across processes and visualizing the parallel
 * load balancing. The output uses color coding to distinguish between different
 * tile types (dense vs low-rank regions).
 * 
 * @param[in] data HICMA PaRSEC data structure containing matrix descriptors
 * @param[in] params HICMA PaRSEC parameters including tile dimensions and band sizes
 */
void hicma_parsec_process_id_print(hicma_parsec_data_t *data,
        hicma_parsec_params_t *params );

#ifdef __cplusplus
}
#endif

#endif /* HICMA_PARSEC_RANK_STATISTICS_H */

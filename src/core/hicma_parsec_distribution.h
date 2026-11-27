/**
 * @file hicma_parsec_distribution.h
 * @brief HICMA PaRSEC matrix distribution header file
 * 
 * This header file contains declarations for matrix distribution functions including
 * symmetric block cyclic band distribution initialization for the HICMA library.
 * The distribution system provides efficient data placement strategies for banded
 * symmetric matrices, optimizing memory access patterns and load balancing across
 * distributed computing nodes.
 * 
 * The distribution functions implement a hybrid approach that combines:
 * - Band distribution for tiles within the specified band size
 * - Off-band distribution for tiles outside the band
 * - Symmetric matrix optimization for triangular matrix operations
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

#ifndef HICMA_PARSEC_DISTRIBUTION_H
#define HICMA_PARSEC_DISTRIBUTION_H

/* ============================================================================
 * HICMA PaRSEC includes
 * ============================================================================ */
#include "hicma_parsec_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Matrix distribution functions
 * ============================================================================ */

/**
 * @brief Initialize symmetric two-dimensional block cyclic band distribution
 * 
 * This function initializes a symmetric two-dimensional block cyclic band distribution
 * structure. It inherits from off-band distribution and should be called after
 * initialization of off_band.
 * 
 * The function sets up the rank_of, vpid_of, data_of, rank_of_key,
 * vpid_of_key, and data_of_key functions for the descriptor. This enables
 * efficient data distribution where:
 * - Tiles within the band_size are distributed using band-optimized strategies
 * - Tiles outside the band use standard off-band distribution
 * - Symmetric matrix properties are exploited for optimal memory layout
 * 
 * @param[in,out] desc Pointer to the symmetric block cyclic band descriptor
 * @param[in] nodes Number of nodes in the distribution
 * @param[in] myrank Rank of the current process
 * @param[in] band_size Size of the band for banded matrix operations
 */
void hicma_parsec_parsec_matrix_sym_block_cyclic_band_init( parsec_matrix_sym_block_cyclic_band_t *desc,
                                         int nodes, int myrank, int band_size );

#ifdef __cplusplus
}
#endif

#endif /* HICMA_PARSEC_DISTRIBUTION_H */

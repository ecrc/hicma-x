/**
 * @file hicma_parsec_rank_statistics.c
 * @brief HICMA PaRSEC rank statistics implementation
 * 
 * This file implements rank statistics functions for analyzing the numerical rank
 * distribution in hierarchical low-rank matrix computations. It provides functions
 * to compute minimum, maximum, and average ranks across matrix tiles, as well as
 * utilities for debugging and visualizing the parallel distribution of tiles.
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"
#include "hicma_parsec_rank_statistics.h"

/**
 * @brief Global flag to use scratch work array while using HCORE functions
 * 
 * This flag controls whether to use a scratch work array during HCORE function
 * calls, which can improve performance by avoiding repeated memory allocations.
 */
int use_scratch = 1;

/**
 * @brief Calculate minimum, maximum, and average rank statistics for a matrix
 * 
 * This function computes rank statistics for a matrix stored in a specific format.
 * It iterates through the matrix tiles according to the specified triangular
 * structure (upper/lower/full) and calculates the minimum, maximum, and average
 * rank values across all valid tiles.
 * 
 * @param[in] uplo Character indicating matrix structure:
 *                 - 'L' for lower triangular
 *                 - 'U' for upper triangular  
 *                 - 'F' or other for full matrix
 * @param[in] Ark Pointer to the rank array containing rank values for each tile
 * @param[in] m Number of tile rows in the matrix
 * @param[in] n Number of tile columns in the matrix
 * @param[in] ld Leading dimension of the rank array
 * @param[out] stat Pointer to statistics structure to store computed values
 */
static void hicma_parsec_get_stat(char uplo, int *Ark, size_t m, size_t n, size_t ld,  hicma_parsec_stat_t *stat) {
    double final_avgrank;
    int final_maxrank = 0;
    int minrank = INT_MAX;
    int final_totalrank = 0;
    int *MAT = Ark;
    int64_t imt, jnt;
    int ntiles = 0;

    /* Iterate through all matrix tiles */
    for(imt = 0; imt < m; imt++) {
        for(jnt = 0; jnt < n; jnt++) {
            /* Skip diagonal elements */
            if(imt == jnt)
                continue;
            /* Skip upper triangular elements for lower triangular matrix */
            if(uplo == 'L' && imt < jnt)
                continue;
            /* Skip lower triangular elements for upper triangular matrix */
            if(uplo == 'U' && imt > jnt)
                continue;
            
            /* Get rank value for current tile */
            int *A = MAT + imt + jnt * ld;
            int rank = A[0];
            
            /* Update maximum rank */
            if(rank > final_maxrank) {
                final_maxrank = rank;
            }
            /* Update minimum rank */
            if(rank < minrank) {
                minrank = rank;
            }
            /* Accumulate total rank for average calculation */
            final_totalrank += rank;
            ntiles++;
        }
    }
    
    /* Calculate average rank */
    final_avgrank = (ntiles > 0) ? (final_totalrank) / (ntiles * 1.0) : 0.0;
    
    /* Store computed statistics */
    stat->min = minrank;
    stat->max = final_maxrank;
    stat->avg = final_avgrank;
}

/**
 * @brief Calculate rank statistics for low-rank tiles in a banded matrix structure
 * 
 * This function computes rank statistics specifically for the low-rank region
 * of a banded matrix. It processes only the tiles that are outside the dense
 * band region, which typically correspond to low-rank approximations in
 * hierarchical matrix computations.
 * 
 * @param[in] G Pointer to the global rank array containing rank values
 * @param[in] lda Leading dimension of the rank array (matrix size)
 * @param[in] band_size_dense Size of the dense band region (tiles with rank 0)
 * @param[out] stat Pointer to statistics structure to store computed values
 */
static void hicma_parsec_get_stat2(int *G, size_t lda, int band_size_dense, hicma_parsec_stat_t *stat) {
    int min = INT_MAX, max = 0;
    long int num = 0;
    long long int sum = 0;

    /* Process only the low-rank region (outside the dense band) */
    for(int i = band_size_dense; i < lda; i++){
        for(int j = 0; j < i - band_size_dense + 1; j++){
            int rank_value = G[j * lda + i];
            sum += rank_value;
            num++;
            
            /* Update minimum rank */
            if(rank_value < min)
                min = rank_value;
            /* Update maximum rank */
            if(rank_value > max)
                max = rank_value;
        } 
    }

    /* Store computed statistics */
    stat->min = min;
    stat->max = max;
    stat->avg = (num > 0) ? ((double)sum) / num : 0.0;
}


/**
 * @brief Compute rank statistics for a distributed matrix
 * 
 * This function gathers rank information from all processes and computes
 * comprehensive rank statistics (minimum, maximum, and average) for the
 * low-rank region of a hierarchical matrix. It also provides optional
 * debugging output to visualize the rank distribution.
 * 
 * @param[in] parsec PaRSEC context for parallel operations
 * @param[in] strid String identifier for the analysis (used in debug output)
 * @param[in] dcAr Distributed rank matrix descriptor
 * @param[in] params HICMA PaRSEC parameters containing matrix dimensions and settings
 * @param[out] minrk Pointer to store minimum rank value
 * @param[out] maxrk Pointer to store maximum rank value  
 * @param[out] avgrk Pointer to store average rank value
 */
void hicma_parsec_rank_stat(parsec_context_t* parsec, char* strid,
        parsec_tiled_matrix_t* dcAr,
        hicma_parsec_params_t *params,
        int* minrk, int* maxrk,
        double* avgrk) {

    int NT = params->NT;
    int rank = params->rank;
    int band_size_dense = params->band_size_dense;
    int maxrank = params->maxrank;

    if( band_size_dense >= NT ) {
        *minrk = 0;
        *maxrk = 0;
        *avgrk = 0.0;
        return;
    }

    /* Gather rank information from all processes to a global array */
    parsec_rank_gather(parsec, dcAr, params->rank_array, band_size_dense);

    /* Calculate rank statistics for the low-rank region */
    hicma_parsec_stat_t rankstat;
    hicma_parsec_get_stat2(params->rank_array, NT, band_size_dense, &rankstat);

    /* Return computed statistics */
    *minrk = rankstat.min;
    *maxrk = rankstat.max;
    *avgrk = rankstat.avg;

#if PRINT_RANK 
    /* Optional debug output: print rank matrix visualization */
    if(rank == 0){ /* Only rank 0 prints to avoid duplicate output */
        fprintf(stderr, "%s %d %d\n", strid, NT, NT);
        int i, j;
        for(i = 0; i < NT; i++){
            for(j = 0; j < NT; j++){
                if( i - j >= band_size_dense )
                    /* Print rank value for low-rank tiles */
                    fprintf(stderr, "%-3d ", params->rank_array[j*NT+i]);
                else if( i >= j )
                    /* Print 0 for dense band tiles (colored red) */
                    fprintf(stderr, RED "%-3d " RESET, 0);
            }
            fprintf(stderr, "\n");
        }
    }
#endif
}


/**
 * @brief Print process ID of each tile for debugging and load balancing analysis
 * 
 * This function prints the process ID associated with each tile in the matrix,
 * providing a visual representation of how tiles are distributed across processes.
 * The output uses color coding to distinguish between different tile types:
 * - Red/Cyan: Dense band region tiles (alternating colors for pattern visualization)
 * - White: Low-rank region tiles
 * 
 * @param[in] data HICMA PaRSEC data structure containing matrix descriptors
 * @param[in] params HICMA PaRSEC parameters including tile dimensions and band sizes
 */
void hicma_parsec_process_id_print(hicma_parsec_data_t *data,
        hicma_parsec_params_t *params ) { 
    
    /* Only rank 0 prints to avoid duplicate output */
    if( 0 == params->rank ) {
        printf("Print rank of each tile:\n");
        int label = 1;
        
        /* Iterate through lower triangular part of the matrix */
        for(int i = 0; i < params->NT; i++) {
            for(int j = 0; j <= i; j++) {
                /* Determine color label based on tile position pattern */
                label = (i-j) / params->band_p % 2;
                
                /* Get process ID for current tile */
                int process_id = data->dcA.super.super.rank_of(&data->dcA.super.super, i, j);
                
                if( i-j < params->band_size_dist ) {
                    /* Dense band region: use alternating colors for pattern visualization */
                    if( 0 == label )
                        printf(RED"% -3d "RESET, process_id);
                    else
                        printf(CYN"% -3d "RESET, process_id);
                }
                else {
                    /* Low-rank region: use white color */
                    printf(WHT"% -3d "RESET, process_id);
                }
            }
            printf("\n");
        }
    }
}

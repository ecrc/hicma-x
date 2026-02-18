/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"
#include "potrf_L_dense_tlr_mp.h"
#include "potrf_L_dense_mp_gpu.h"
#include "potrf_L_dense_mp_gpu_fp8.h"
#include "potrf_L_dense_mp_gpu_fp8_adaptive.h"
#include "potrf_L_dense_mp_gpu_fp8_sp.h"

/**
 * @file hicma_parsec_decision.c
 * @brief Implementation of decision-making functions for HICMA PaRSEC
 *
 * This file contains the core decision-making logic for HICMA PaRSEC operations.
 * It implements functions for initializing and updating precision decisions,
 * printing decision matrices, and providing datatype conversion utilities.
 * The decision system determines the optimal precision for each matrix tile
 * based on position, performance requirements, and numerical properties.
 */

/**
 * @brief Update decisions for each tile
 *
 * Updates the decision matrix for each tile based on current computation state.
 * This decision is used for matrix generation and Cholesky factorization.
 * The function analyzes the current matrix state and updates precision decisions
 * for optimal performance and accuracy.
 *
 * @param[in] parsec PaRSEC context
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 */
void  hicma_parsec_decisions_update( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params ) {

    /* Adaptive decision */
    if( params->adaptive_decision ) {
        hicma_parsec_decision_make_comp( parsec, (parsec_tiled_matrix_t*)&data->dcA, params );
        params->time_decision_kernel = sync_time_elapsed;
    }

    /* Count decision */ 
    hicma_parsec_decision_count( parsec, (parsec_tiled_matrix_t*)&data->dcA, params );

    /* Analysis data convert type */
    /* Serial version: hicma_parsec_decisions_send_analysis( params ) */
    hicma_parsec_decision_make_send( parsec, (parsec_tiled_matrix_t*)&data->dcA, params );
    params->time_decision_sender = sync_time_elapsed;

    if( params->verbose > 9 ) {
        print_decisions( params );
        //print_decisions_send( params );
        //print_decisions_gemm_gpu( params );
    }
}

/**
 * @brief Initialize decisions for each tile
 *
 * Initializes the decision matrix for each tile based on band size parameters.
 * This decision is used for matrix generation and Cholesky factorization.
 * The function sets precision decisions based on the tile's distance from the diagonal,
 * with higher precision used for tiles closer to the diagonal.
 *
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success
 */
int hicma_parsec_decision_init( hicma_parsec_params_t *params )
{
    for(int i = 0; i < params->NT; i++) {
        for(int j = 0; j <= i; j++) {
            /* decisions_gemm_gpu */
            params->decisions_gemm_gpu[j*params->NT+i] |= MASK_FP64;
            if( params->tensor_gemm & MASK_TF32 ) {
                params->decisions_gemm_gpu[j*params->NT+i] |= MASK_TF32;
            } else {
                params->decisions_gemm_gpu[j*params->NT+i] |= MASK_FP32;
            }
            switch( params->tensor_gemm & MASK_ONLY_FP16 ) {
                case MASK_TF16_A16_B16_C32_OP32:
                    params->decisions_gemm_gpu[j*params->NT+i] |= MASK_TF16_A16_B16_C32_OP32;
                    break;
                case MASK_TF16_A16_B16_C16_OP32:
                    params->decisions_gemm_gpu[j*params->NT+i] |= MASK_TF16_A16_B16_C16_OP32;
                    break;
                case MASK_TF16_A16_B16_C16_OP16:
                    params->decisions_gemm_gpu[j*params->NT+i] |= MASK_TF16_A16_B16_C16_OP16;
                    break;
                case MASK_BF16_A16_B16_C32_OP32:
                    params->decisions_gemm_gpu[j*params->NT+i] |= MASK_BF16_A16_B16_C32_OP32;
                    break;
                case MASK_BF16_A16_B16_C16_OP16:
                    params->decisions_gemm_gpu[j*params->NT+i] |= MASK_BF16_A16_B16_C16_OP16;
                    break;
            }

            /* decisions */
            if( i-j < params->band_size_dense_dp ) {
                params->decisions[j*params->NT+i] = DENSE_DP;
                params->decisions_gemm_gpu[j*params->NT+i] |= MASK_FP64; 
            } else if( i-j < params->band_size_dense_sp ) {
                params->decisions[j*params->NT+i] = DENSE_SP;
            } else if( i-j < params->band_size_dense_hp ) {
                params->decisions[j*params->NT+i] = DENSE_HP;
            } else if( i-j < params->band_size_dense ) {
                params->decisions[j*params->NT+i] = DENSE_FP8;
            } else if( i-j < params->band_size_low_rank_dp ) {
                params->decisions[j*params->NT+i] = LOW_RANK_DP;
            } else
                params->decisions[j*params->NT+i] = LOW_RANK_SP;
        }
    }

    if( 0 && params->verbose > 9 ) {
        print_decisions( params );
    }

    return 0;
}

/**
 * @brief Print decision matrix
 *
 * Prints the decision matrix showing precision choices for each tile.
 * Uses color coding to distinguish between different precision levels:
 * - Red: Double Precision (DP)
 * - Blue: Single Precision (SP)
 * - Purple: Low Rank Double Precision (LR_DP)
 * - Yellow: Low Rank Single Precision (LR_SP)
 * - Green: Half Precision (HP)
 * - Cyan: FP8
 *
 * @param[in] params HICMA PaRSEC parameters
 */
void print_decisions( hicma_parsec_params_t *params ) {
    sleep(1);
    fflush(stdout);

    /* Print decisions */
    if( 0 == params->rank ) {
        fprintf(stderr, "\ndecisions: DENSE_DP= %d DENSE_SP= %d LOW_RANK_DP= %d LOW_RANK_SP= %d DENSE_HP= %d DENSE_FP8= %d\n",
                DENSE_DP, DENSE_SP, LOW_RANK_DP, LOW_RANK_SP, DENSE_HP, DENSE_FP8);
        for(int i = 0; i < params->NT; i++) {
            for(int j = 0; j <= i; j++) {
                if( DENSE_DP == params->decisions[j*params->NT+i] )
                    fprintf(stderr, RED "%2d " RESET, params->decisions[j*params->NT+i]);
                else if( DENSE_SP == params->decisions[j*params->NT+i] )
                    fprintf(stderr, BLU "%2d " RESET, params->decisions[j*params->NT+i]);
                else if( LOW_RANK_DP == params->decisions[j*params->NT+i] )
                    fprintf(stderr, PUR "%2d " RESET, params->decisions[j*params->NT+i]);
                else if( LOW_RANK_SP == params->decisions[j*params->NT+i] )
                    fprintf(stderr, YEL "%2d " RESET, params->decisions[j*params->NT+i]);
                else if( DENSE_HP == params->decisions[j*params->NT+i] )
                    fprintf(stderr, GRN "%2d " RESET, params->decisions[j*params->NT+i]);
                else if( DENSE_FP8 == params->decisions[j*params->NT+i] )
                    fprintf(stderr, CYN"%2d " RESET, params->decisions[j*params->NT+i]);
                else
                    fprintf(stderr, WHT"E" RESET);
            }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }

        fflush(stdout);
        sleep(1);
}

static void print_runtime_decisions(const hicma_parsec_params_t *params,
        const uint16_t *decisions, const char *label, int fill_gemm_boundaries)
{
    if(params->rank != 0) return;

    fflush(stdout);
    sleep(1);
    fprintf(stderr, "\n%s: DP=%d SP=%d LR_DP=%d LR_SP=%d HP=%d FP8=%d\n", label,
            DENSE_DP, DENSE_SP, LOW_RANK_DP, LOW_RANK_SP, DENSE_HP, DENSE_FP8);
    for(int m = 0; m < params->MT; m++) {
        for(int n = 0; n <= m && n < params->NT; n++) {
            size_t index = (size_t)n * params->MT + m;
            uint16_t precision = (fill_gemm_boundaries && n == 0) ?
                params->decisions[index] : decisions[index];
            if(fill_gemm_boundaries && m == n) {
                fprintf(stderr, RED "%2d " RESET, DENSE_DP);
            } else if(precision == DENSE_DP) {
                fprintf(stderr, RED "%2u " RESET, (unsigned)precision);
            } else if(precision == DENSE_SP) {
                fprintf(stderr, BLU "%2u " RESET, (unsigned)precision);
            } else if(precision == LOW_RANK_DP) {
                fprintf(stderr, PUR "%2u " RESET, (unsigned)precision);
            } else if(precision == LOW_RANK_SP) {
                fprintf(stderr, YEL "%2u " RESET, (unsigned)precision);
            } else if(precision == DENSE_HP) {
                fprintf(stderr, GRN "%2u " RESET, (unsigned)precision);
            } else if(precision == DENSE_FP8) {
                fprintf(stderr, CYN "%2u " RESET, (unsigned)precision);
            } else {
                fprintf(stderr, "  -- ");
            }
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

void print_first_local_gemm_decisions(const hicma_parsec_params_t *params)
{
    print_runtime_decisions(params, params->decisions_gemm_first,
            "first_local_gemm_precision (A*B, k=0)", 1);
}

void print_last_local_gemm_decisions(const hicma_parsec_params_t *params)
{
    print_runtime_decisions(params, params->decisions_gemm_last,
            "last_local_gemm_precision (A*B, k=n-1)", 1);
}

void print_final_storage_decisions(const hicma_parsec_params_t *params)
{
    print_runtime_decisions(params, params->decisions_storage_final,
            "final_storage_precision (POTRF diagonal, TRSM off-diagonal)", 0);
}

void get_decisions(uint16_t *decisions, size_t size,
        parsec_tiled_matrix_t *descA ) {
    assert((int)size == size);
    assert(size == (size_t)descA->lmt * descA->lnt);

    /* Runtime decisions are updated only by the rank owning each output tile.
     * Remove stale replicas before summing so every tile has one contributor. */
    for( int n = 0; n < descA->lnt; n++ ) {
        for( int m = 0; m < descA->lmt; m++ ) {
            size_t index = (size_t)n * descA->lmt + m;
            if( m < n || descA->super.myrank !=
                    descA->super.rank_of(&descA->super, m, n) ) {
                decisions[index] = 0;
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, decisions, size, MPI_UNSIGNED_SHORT, MPI_SUM, MPI_COMM_WORLD);
}

/**
 * @brief Print send decision matrix
 *
 * Prints the send decision matrix showing precision choices for data transmission.
 * Uses the same color coding as print_decisions() to maintain consistency.
 * This matrix determines the precision used when sending data between processes.
 *
 * @param[in] params HICMA PaRSEC parameters
 */
void print_decisions_send( hicma_parsec_params_t *params ) {
    sleep(1);
    fflush(stdout);

    if( 0 == params->rank ) {
        fprintf(stderr, "\ndecisions_send: DENSE_DP= %d DENSE_SP= %d LOW_RANK_DP= %d LOW_RANK_SP= %d DENSE_HP= %d DENSE_FP8= %d\n",
                DENSE_DP, DENSE_SP, LOW_RANK_DP, LOW_RANK_SP, DENSE_HP, DENSE_FP8);
        for(int i = 0; i < params->NT; i++) {
            for(int j = 0; j <= i; j++) {
                if( DENSE_DP == params->decisions_send[j*params->NT+i] )
                    fprintf(stderr, RED "%2d " RESET, params->decisions_send[j*params->NT+i]);
                else if( DENSE_SP == params->decisions_send[j*params->NT+i] )
                    fprintf(stderr, BLU "%2d " RESET, params->decisions_send[j*params->NT+i]);
                else if( LOW_RANK_DP == params->decisions_send[j*params->NT+i] )
                    fprintf(stderr, PUR "%2d " RESET, params->decisions_send[j*params->NT+i]);
                else if( LOW_RANK_SP == params->decisions_send[j*params->NT+i] )
                    fprintf(stderr, YEL "%2d " RESET, params->decisions_send[j*params->NT+i]);
                else if( DENSE_HP == params->decisions_send[j*params->NT+i] )
                    fprintf(stderr, GRN "%2d " RESET, params->decisions_send[j*params->NT+i]);
                else if( DENSE_FP8 == params->decisions_send[j*params->NT+i] )
                    fprintf(stderr, CYN"%2d " RESET, params->decisions_send[j*params->NT+i]);
                else
                    fprintf(stderr, WHT"E" RESET);
            }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }

        fflush(stdout);
        sleep(1);
}

/**
 * @brief Analyze and set send decisions for data conversion
 *
 * Analyzes the decision matrix and sets appropriate send decisions based on
 * datatype conversion strategy. This function should be called after
 * hicma_parsec_decision_make to ensure proper decision propagation.
 * 
 * The function supports three conversion strategies:
 * - 0: Conservative conversion (HP/FP8 -> SP)
 * - 1: No conversion (not used)
 * - 2: Aggressive conversion with precision optimization
 *
 * @param[in] params HICMA PaRSEC parameters
 */
void hicma_parsec_decisions_send_analysis( hicma_parsec_params_t *params ) {
    /* Check */
    if( params->datatype_convert < 0 || params->datatype_convert > 2 ) {
        if( 0 == params->rank ) printf(RED "datatype_convert is wrong, which should be 0, 1, or 2; set it to 2\n");
        params->datatype_convert = 2;
    }

    int NT = params->NT;
    if( 0 == params->datatype_convert ) {
        for( int n = 0; n < NT; n++ ) {
            for( int m = n; m < NT; m++ ) {
                params->decisions_send[n*NT+m] = params->decisions[n*NT+m];
                if( DENSE_HP == params->decisions[n*NT+m] || DENSE_FP8 == params->decisions[n*NT+m] ) {
                    params->decisions_send[n*NT+m] = DENSE_SP;
                }
            }
        }
    } else if( 1 == params->datatype_convert ) {
        if( 0 == params->rank ) printf(RED "decisions_send is not used for datatype_convert == 1\n");
    } else if( 2 == params->datatype_convert ) {
        for( int n = 0; n < NT; n++ ) {
            for( int m = n; m < NT; m++ ) {
                /* Diagonal */
                if( m == n ) {
                    params->decisions_send[n*NT+m] = DENSE_SP;
                    for( int k = m+1; k < NT; k++ ) {
                        if( params->decisions[m*NT+k] == DENSE_DP ) {
                            params->decisions_send[n*NT+m] = DENSE_DP;
                            goto done_check_this_tile;
                        }
                    }
                    goto done_check_this_tile;
                }

                /* Set to HP by default */
                params->decisions_send[n*NT+m] = DENSE_HP;
#if HAVE_FP8
                params->decisions_send[n*NT+m] = DENSE_FP8;
#endif

                /* Store decision */
                uint16_t precision_tile = hicma_parsec_min( DENSE_SP,  params->decisions[n*NT+m] );

                /* Checking A */
                for( int k = n+1; k < m; k++ ) {
                    params->decisions_send[n*NT+m] = hicma_parsec_min(params->decisions_send[n*NT+m], params->decisions[k*NT+m]);

                    if( params->decisions_send[n*NT+m] <= precision_tile ) {
                        params->decisions_send[n*NT+m] = precision_tile;
                        goto done_check_this_tile;
                    }
                }

                /* Checking B */
                for( int k = m+1; k < NT; k++ ) {
                    params->decisions_send[n*NT+m] = hicma_parsec_min(params->decisions_send[n*NT+m], params->decisions[m*NT+k]);

                    if( params->decisions_send[n*NT+m] <= precision_tile ) {
                        params->decisions_send[n*NT+m] = precision_tile;
                        goto done_check_this_tile;
                    }
                }

                done_check_this_tile:
                    ;
            }
        }

        /* Set for the last potrf the same as decisions */
        params->decisions_send[(NT-2)*NT+NT-1] = params->decisions[(NT-2)*NT+NT-1];
        params->decisions_send[(NT-1)*NT+NT-1] = params->decisions[(NT-1)*NT+NT-1];
    }
}

/**
 * @brief Print GPU GEMM decision matrix
 *
 * Prints the GPU GEMM decision matrix showing precision masks for each tile.
 * Displays the hexadecimal values of precision masks used for GPU operations.
 * This matrix determines which precision levels are enabled for GPU computation.
 *
 * @param[in] params HICMA PaRSEC parameters
 */
void print_decisions_gemm_gpu( hicma_parsec_params_t *params ) {
    sleep(1);
    fflush(stdout);

    if( 0 == params->rank ) {
        fprintf(stderr, "decisions_gemm_gpu mask (Hex):\n"
                "MASK_FP64= 0x%x MASK_FP32= 0x%x MASK_TF32= 0x%x\n"
                "MASK_TF16_A16_B16_C32_OP32= 0x%x MASK_TF16_A16_B16_C16_OP32= 0x%x\n"
                "MASK_TF16_A16_B16_C16_OP16= 0x%x MASK_BF16_A16_B16_C32_OP32= 0x%x\n"
                "MASK_BF16_A16_B16_C16_OP16= 0x%x\n", 
                MASK_FP64, MASK_FP32, MASK_TF32, MASK_TF16_A16_B16_C32_OP32, MASK_TF16_A16_B16_C16_OP32,
                MASK_TF16_A16_B16_C16_OP16, MASK_BF16_A16_B16_C32_OP32, MASK_BF16_A16_B16_C16_OP16);
        for(int i = 0; i < params->NT; i++) {
            for(int j = 0; j <= i; j++) {
                fprintf(stderr,  "%2d ", params->decisions_gemm_gpu[j*params->NT+i]);
            }
            fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }

        fflush(stdout);
        sleep(1);
}

/* ============================================================================
 * Datatype decision functions for POTRF operations
 * ============================================================================ */

/**
 * @brief Get datatype for POTRF dense TLR mixed precision tile
 *
 * Determines the appropriate datatype index for POTRF dense TLR mixed precision
 * operations based on the decision matrix. Returns the corresponding PaRSEC
 * ADT index for the specified precision level.
 *
 * @param[in] decisions Decision array
 * @param[in] m Row index
 * @param[in] n Column index
 * @param[in] NT Matrix size
 * @return PaRSEC ADT index for the specified precision
 */
int decision_datatype_tile_potrf_L_dense_tlr_mp(uint16_t *decisions, int m, int n, int NT) {
    switch( decisions[n*NT+m] ) {
        case DENSE_DP:
            return PARSEC_potrf_L_dense_tlr_mp_FULL_DP_ADT_IDX;
        case DENSE_SP: case DENSE_HP: case DENSE_FP8:
            return PARSEC_potrf_L_dense_tlr_mp_FULL_SP_ADT_IDX;
        case LOW_RANK_DP:
            return PARSEC_potrf_L_dense_tlr_mp_UV_DP_ADT_IDX;
        case LOW_RANK_SP:
            return PARSEC_potrf_L_dense_tlr_mp_UV_SP_ADT_IDX;
        default:
            fprintf(stderr, "The decision is not correct! %d\n", decisions[n*NT+m]);
            return -1;
    }
}

/**
 * @brief Get datatype for POTRF dense mixed precision GPU tile
 *
 * Determines the appropriate datatype index for POTRF dense mixed precision
 * GPU operations based on the decision matrix. Returns the corresponding PaRSEC
 * ADT index for the specified precision level.
 *
 * @param[in] decisions Decision array
 * @param[in] m Row index
 * @param[in] n Column index
 * @param[in] NT Matrix size
 * @return PaRSEC ADT index for the specified precision
 */
int decision_datatype_tile_potrf_L_dense_mp_gpu(uint16_t *decisions, int m, int n, int NT) {
    switch( decisions[n*NT+m] ) {
        case DENSE_DP: 
            return PARSEC_potrf_L_dense_mp_gpu_FULL_DP_ADT_IDX;
        case DENSE_SP: case DENSE_HP: case DENSE_FP8:
            return PARSEC_potrf_L_dense_mp_gpu_FULL_SP_ADT_IDX;
        case LOW_RANK_DP:
            return PARSEC_potrf_L_dense_mp_gpu_UV_DP_ADT_IDX;
        case LOW_RANK_SP:
            return PARSEC_potrf_L_dense_mp_gpu_UV_SP_ADT_IDX;
        default:
            fprintf(stderr, "The decision is not correct! %d \n", decisions[n*NT+m]);
            return -1;
    }
} 

/**
 * @brief Get datatype for POTRF dense mixed precision GPU FP8 tile
 *
 * Determines the appropriate datatype index for POTRF dense mixed precision
 * GPU FP8 operations based on the decision matrix. Returns the corresponding PaRSEC
 * ADT index for the specified precision level.
 *
 * @param[in] decisions Decision array
 * @param[in] m Row index
 * @param[in] n Column index
 * @param[in] NT Matrix size
 * @return PaRSEC ADT index for the specified precision
 */
int decision_datatype_tile_potrf_L_dense_mp_gpu_fp8(uint16_t *decisions, int m, int n, int NT) {
    switch( decisions[n*NT+m] ) {
        case DENSE_DP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_FULL_DP_ADT_IDX;
        case DENSE_SP: case DENSE_HP: case DENSE_FP8:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_FULL_SP_ADT_IDX;
        case LOW_RANK_DP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_UV_DP_ADT_IDX;
        case LOW_RANK_SP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_UV_SP_ADT_IDX;
        default:
            fprintf(stderr, "The decision is not correct! %d \n", decisions[n*NT+m]);
            return -1;
    }
}


/**
 * @brief Get datatype for POTRF dense mixed precision GPU FP8 tile adaptive
 *
 * Determines the appropriate datatype index for POTRF dense mixed precision
 * GPU FP8 operations based on the decision matrix. Returns the corresponding PaRSEC
 * ADT index for the specified precision level.
 *
 * @param[in] decisions Decision array
 * @param[in] m Row index
 * @param[in] n Column index
 * @param[in] NT Matrix size
 * @return PaRSEC ADT index for the specified precision
 */
int decision_datatype_tile_potrf_L_dense_mp_gpu_fp8_adaptive(uint16_t *decisions, int m, int n, int NT) {
    switch( decisions[n*NT+m] ) {
        case DENSE_DP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_FULL_DP_ADT_IDX;
        case DENSE_SP: case DENSE_HP: case DENSE_FP8:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_FULL_SP_ADT_IDX;
        case LOW_RANK_DP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_UV_DP_ADT_IDX;
        case LOW_RANK_SP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_UV_SP_ADT_IDX;
        default:
            fprintf(stderr, "The decision is not correct! %d\n", decisions[n*NT+m]);
            return -1;
    }
}

/**
 * @brief Get datatype for POTRF dense mixed precision GPU FP8 SP tile
 *
 * Determines the appropriate datatype index for POTRF dense mixed precision
 * GPU FP8 SP operations based on the decision matrix. Returns the corresponding PaRSEC
 * ADT index for the specified precision level.
 *
 * @param[in] decisions Decision array
 * @param[in] m Row index
 * @param[in] n Column index
 * @param[in] NT Matrix size
 * @return PaRSEC ADT index for the specified precision
 */
int decision_datatype_tile_potrf_L_dense_mp_gpu_fp8_sp(uint16_t *decisions, int m, int n, int NT) {
    switch( decisions[n*NT+m] ) {
        case DENSE_DP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_sp_FULL_DP_ADT_IDX;
        case DENSE_SP: case DENSE_HP: case DENSE_FP8:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_sp_FULL_SP_ADT_IDX;
        case LOW_RANK_DP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_sp_UV_DP_ADT_IDX;
        case LOW_RANK_SP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_sp_UV_SP_ADT_IDX;
        default:
            fprintf(stderr, "The decision is not correct! %d \n", decisions[n*NT+m]);
            return -1;
    }
}

/**
 * @brief Get datatype for POTRF dense mixed precision GPU send tile
 *
 * Determines the appropriate datatype index for POTRF dense mixed precision
 * GPU send operations based on the decision matrix. Returns the corresponding PaRSEC
 * ADT index for the specified precision level.
 *
 * @param[in] decisions Decision array
 * @param[in] m Row index
 * @param[in] n Column index
 * @param[in] NT Matrix size
 * @return PaRSEC ADT index for the specified precision
 */
int decision_datatype_tile_send_potrf_L_dense_mp_gpu(uint16_t *decisions, int m, int n, int NT) {
    switch( decisions[n*NT+m] ) {
        case DENSE_DP:
            return PARSEC_potrf_L_dense_mp_gpu_FULL_DP_ADT_IDX;
        case DENSE_SP:
            return PARSEC_potrf_L_dense_mp_gpu_FULL_SP_ADT_IDX;
        case DENSE_HP:
            return PARSEC_potrf_L_dense_mp_gpu_FULL_HP_ADT_IDX;
        case DENSE_FP8:
            return PARSEC_potrf_L_dense_mp_gpu_FULL_FP8_ADT_IDX;
        case LOW_RANK_DP:
            return PARSEC_potrf_L_dense_mp_gpu_UV_DP_ADT_IDX;
        case LOW_RANK_SP:
            return PARSEC_potrf_L_dense_mp_gpu_UV_SP_ADT_IDX;
        default:
            fprintf(stderr, "The decision is not correct! %d \n", decisions[n*NT+m]);
            return -1;
    }
}

/**
 * @brief Get datatype for POTRF dense mixed precision GPU FP8 send tile
 *
 * Determines the appropriate datatype index for POTRF dense mixed precision
 * GPU FP8 send operations based on the decision matrix. Returns the corresponding PaRSEC
 * ADT index for the specified precision level.
 *
 * @param[in] decisions Decision array
 * @param[in] m Row index
 * @param[in] n Column index
 * @param[in] NT Matrix size
 * @return PaRSEC ADT index for the specified precision
 */
int decision_datatype_tile_send_potrf_L_dense_mp_gpu_fp8(uint16_t *decisions, int m, int n, int NT) {
    switch( decisions[n*NT+m] ) {
        case DENSE_DP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_FULL_DP_ADT_IDX;
        case DENSE_SP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_FULL_SP_ADT_IDX;
        case DENSE_HP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_FULL_HP_ADT_IDX;
        case DENSE_FP8:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_FULL_FP8_ADT_IDX;
        case LOW_RANK_DP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_UV_DP_ADT_IDX;
        case LOW_RANK_SP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_UV_SP_ADT_IDX;
        default:
            fprintf(stderr, "The decision is not correct! %d \n", decisions[n*NT+m]);
            return -1;
    }
}

/**
 * @brief Get datatype for POTRF dense mixed precision GPU FP8 send tile adaptive
 *
 * Determines the appropriate datatype index for POTRF dense mixed precision
 * GPU FP8 send operations based on the decision matrix. Returns the corresponding PaRSEC
 * ADT index for the specified precision level.
 *
 * @param[in] decisions Decision array
 * @param[in] m Row index
 * @param[in] n Column index
 * @param[in] NT Matrix size
 * @return PaRSEC ADT index for the specified precision
 */
int decision_datatype_tile_send_potrf_L_dense_mp_gpu_fp8_adaptive(uint16_t *decisions, int m, int n, int NT) {
    switch( decisions[n*NT+m] ) {
        case DENSE_DP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_FULL_DP_ADT_IDX;
        case DENSE_SP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_FULL_SP_ADT_IDX;
        case DENSE_HP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_FULL_HP_ADT_IDX;
        case DENSE_FP8:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_FULL_FP8_ADT_IDX;
        case LOW_RANK_DP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_UV_DP_ADT_IDX;
        case LOW_RANK_SP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_UV_SP_ADT_IDX;
        default:
            fprintf(stderr, "The decision is not correct! %d \n", decisions[n*NT+m]);
            return -1;
    }
}

/**
 * @brief Get datatype for POTRF dense mixed precision GPU FP8 SP send tile
 *
 * Determines the appropriate datatype index for POTRF dense mixed precision
 * GPU FP8 SP send operations based on the decision matrix. Returns the corresponding PaRSEC
 * ADT index for the specified precision level.
 *
 * @param[in] decisions Decision array
 * @param[in] m Row index
 * @param[in] n Column index
 * @param[in] NT Matrix size
 * @return PaRSEC ADT index for the specified precision
 */
int decision_datatype_tile_send_potrf_L_dense_mp_gpu_fp8_sp(uint16_t *decisions, int m, int n, int NT) {
    switch( decisions[n*NT+m] ) {
        case DENSE_DP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_sp_FULL_DP_ADT_IDX;
        case DENSE_SP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_sp_FULL_SP_ADT_IDX;
        case DENSE_HP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_sp_FULL_HP_ADT_IDX;
        case DENSE_FP8:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_sp_FULL_FP8_ADT_IDX;
        case LOW_RANK_DP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_sp_UV_DP_ADT_IDX;
        case LOW_RANK_SP:
            return PARSEC_potrf_L_dense_mp_gpu_fp8_sp_UV_SP_ADT_IDX;
        default:
            fprintf(stderr, "The decision is not correct! %d \n", decisions[n*NT+m]);
            return -1;
    }
}


/**
 * @brief Determine the precision of that tile 
 *
 * @return The precision decision 
 */
void hicma_parsec_get_precision_tile(hicma_parsec_params_t *params_tlr,
        hicma_parsec_decision_enum_t *new_decision, double norm_tile, int m, int n) {

    double scalar_factore = (double)params_tlr->NT;

    // Determine precision decisions based on tile norm relative to global norm and tolerance
    // Each decision checks if the normalized tile norm is below the precision-specific threshold
    bool decision_hp_gpu = norm_tile * scalar_factore / params_tlr->norm_global  < params_tlr->fixedacc / EPS_HP_GPU;
    bool decision_hp = norm_tile * scalar_factore / params_tlr->norm_global  < params_tlr->fixedacc / EPS_HP;
    bool decision_sp = norm_tile * scalar_factore / params_tlr->norm_global  < params_tlr->fixedacc / EPS_SP;
    bool decision_fp8 = norm_tile * scalar_factore / params_tlr->norm_global  < params_tlr->fixedacc / EPS_FP8;

    // Debug print statement (commented out)
    //printf("\n (%d, %d), params_tlr->norm_global:%f, tile norm:%f, res:%f %f %f %f\n", m, n, params_tlr->norm_global,  norm_tile);

    /* Make precision decisions based on current tile type */
    // Handle dense tile precision decisions (DP, SP, HP, FP8)
    if( IS_DENSE(m, n) ) {
        // Precision selection hierarchy: FP8 -> HP -> SP -> DP (if supported)
#if HAVE_FP8
        if( decision_fp8 ) {
            *new_decision = DENSE_FP8;
        }
#if HAVE_HP
        else if( decision_hp ) {
            *new_decision = DENSE_HP;
        } else if( decision_sp ) {
#else
        if( decision_sp ) {
#endif // HAVE_HP

#else
        // FP8 not supported, check HP and SP
#if HAVE_HP
        if( decision_hp ) {
            *new_decision = DENSE_HP;
        } else if( decision_sp ) {
#else
        if( decision_sp ) {
#endif // HAVE_HP

#endif // HAVE_FP8

            *new_decision = DENSE_SP;
        } else {
            // Fall back to highest precision based on build configuration
#if GENOMICS
            *new_decision = DENSE_SP;
#else
            *new_decision = DENSE_DP;
#endif
        }
    } else {
        // Handle low-rank tile precision decisions
        if( decision_sp ) {
            *new_decision = LOW_RANK_SP;
        } else {
            *new_decision = LOW_RANK_DP;
        }
    }

    // GPU precision decisions for dense tiles only
    if( params_tlr->gpus > 0 && IS_DENSE(m, n) ) {
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        // CUDA-specific GPU precision selection
        if( decision_hp_gpu && ENABLE_TF16_A16_B16_C16_OP16 ) {
            params_tlr->decisions_gemm_gpu[n*params_tlr->NT+m] = MASK_TF16_A16_B16_C16_OP16;
            //printf("%d %d : %g %g\n", m, n, params_tlr->norm_tile[n*params_tlr->NT+m], params_tlr->norm_global);
        } else if( decision_hp ) {
#elif defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        // HIP-specific GPU precision selection
        if( decision_hp ) {
#else
        // No GPU support
        if( 0 ) {
#endif
            params_tlr->decisions_gemm_gpu[n*params_tlr->NT+m] = MASK_TF16_A16_B16_C32_OP32;
        } else if( decision_sp ) {
            params_tlr->decisions_gemm_gpu[n*params_tlr->NT+m] = MASK_FP32;
        } else {
            // Fall back to highest precision based on build configuration
#if GENOMICS
            params_tlr->decisions_gemm_gpu[n*params_tlr->NT+m] = MASK_FP32;
#else
            params_tlr->decisions_gemm_gpu[n*params_tlr->NT+m] = MASK_FP64;
#endif
        }
    }

}

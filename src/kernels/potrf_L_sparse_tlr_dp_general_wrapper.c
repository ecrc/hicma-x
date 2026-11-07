/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

/**
 * @file potrf_L_sparse_tlr_dp_general_wrapper.c
 * @brief PaRSEC wrapper of potrf_L_sparse_tlr_dp_general.jdf (kind 6, SPARSE_TLR_DP_GENERAL: tile low-rank Cholesky over the sparsity pattern of the analysis)
 *
 * Timing hooks, chore selection, memory pools, GPU discovery and the driver
 * are shared with the other Cholesky wrappers (potrf_common.h); this file
 * keeps what is specific to this kind: the GPU evaluate hooks, the pools, the
 * arenas and the constructor arguments.
 */

#include "hicma_parsec.h"
#include "hicma_kernel_time.h"
#include "potrf_common.h"
#include "potrf_L_sparse_tlr_dp_general.h"

/**
 * @brief Create the taskpool of SPARSE_TLR_DP_GENERAL
 *
 * @param parsec PaRSEC context
 * @param data Matrix descriptors (dcA/dcAd, dcAr, dcRank, dcFake) and GPU workspace
 * @param params Parameters; params->info is reset to 0 and set by the kernels
 * @param analysis Sparsity analysis of the matrix
 * @return the taskpool, or NULL if the parameters are not supported
 */
parsec_taskpool_t*
potrf_L_sparse_tlr_dp_general_New( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t *analysis )
{
    int rank = params->rank;
    int uplo = params->uplo;
    int hmb = params->HNB;
    int compmaxrank = params->compmaxrank;
    int storagemaxrank = params->genmaxrank;

    /* The sparse kinds always work on the band/off-band descriptor */
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&data->dcA;
    parsec_tiled_matrix_t *Ar = (parsec_tiled_matrix_t *)&data->dcAr;
    parsec_tiled_matrix_t *Rank = (parsec_tiled_matrix_t *)&data->dcRank;

    if( uplo != PlasmaLower ) {
        dplasma_error("potrf_L_sparse_tlr_dp_general_New", "only PlasmaLower is supported");
        return NULL;
    }
    if( storagemaxrank > compmaxrank ) {
        dplasma_error("potrf_L_sparse_tlr_dp_general_New", "maxrank for storage larger than maxrank for buffers is not meaningful");
        return NULL;
    }
    if( storagemaxrank > (A->mb/2) && 0 == rank ) {
        fprintf(stderr, RED "Warning: maxrank= %d is larger than half of block size\n" RESET, storagemaxrank);
    }

    params->info = 0;
    parsec_potrf_L_sparse_tlr_dp_general_taskpool_t *tp = parsec_potrf_L_sparse_tlr_dp_general_new( A, Ar, Rank, params, analysis );

    /* Timing hooks on the recursive or CPU chore (these kinds have no GPU bodies) */
    if( 0 != hicma_potrf_install_hooks(&tp->super, params, A, 0, hmb, A->mb, NULL) ) {
        parsec_taskpool_free(&tp->super);
        return NULL;
    }

    /* Per-thread workspace pools */
    tp->_g_p_work = hicma_potrf_pool_new( hicma_potrf_hcore_workspace_doubles(A->mb, compmaxrank) * sizeof(double) );
    tp->_g_p_work_rr = hicma_potrf_pool_new( compmaxrank * compmaxrank * sizeof(double) );
    tp->_g_p_work_mbr = hicma_potrf_pool_new( A->mb * compmaxrank * sizeof(double) );

    tp->_g_PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );
    if( 0 == tp->_g_PRI_CHANGE ) {
        tp->_g_PRI_CHANGE = A->nt;
    }

    /* Arenas: one per datatype the flows exchange */
    /* Set up data type arenas for efficient memory management */
    /* Full matrix arena for dense blocks */
    parsec_add2arena(&tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_FULL_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Default arena for scalar values */
    parsec_add2arena(&tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_DEFAULT_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* UV arena for low-rank factor matrices */
    parsec_add2arena(&tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_UV_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, storagemaxrank*2, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* AR arena for rank information */
    parsec_add2arena(&tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_AR_ADT_IDX],
            parsec_datatype_int_t, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    return (parsec_taskpool_t*)tp;
}

/**
 * @brief Release everything potrf_L_sparse_tlr_dp_general_New allocated, then the taskpool
 */
void potrf_L_sparse_tlr_dp_general_Destruct(parsec_taskpool_t* _tp)
{
    parsec_potrf_L_sparse_tlr_dp_general_taskpool_t *tp = (parsec_potrf_L_sparse_tlr_dp_general_taskpool_t*)_tp;
    parsec_memory_pool_t *pools[] = {
        tp->_g_p_work,
        tp->_g_p_work_rr,
        tp->_g_p_work_mbr
    };

    hicma_potrf_forget(_tp);

    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_FULL_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_DEFAULT_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_UV_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_AR_ADT_IDX] );

    parsec_taskpool_free(_tp);

    /* After the taskpool: its repositories may still reference pool elements */
    for( size_t i = 0; i < sizeof(pools) / sizeof(pools[0]); i++ ) {
        hicma_potrf_pool_free(pools[i]);
    }
}

/**
 * @brief Run SPARSE_TLR_DP_GENERAL on data->dcA (or dcAd); params->info holds the
 *        POTRF info after the reduction done by the caller
 *
 * @return 0, or -1 if the taskpool could not be created
 */
int potrf_L_sparse_tlr_dp_general( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t *analysis )
{
    /* The sparse kinds factorize with a dense band of one tile */
    if( params->band_size_dense != 1 ) {
        if( 0 == params->rank )
            fprintf(stderr, RED "SPARSE_TLR_DP_GENERAL supports band_size_dense = 1 only\n" RESET);
        exit(1);
    }

    return hicma_potrf_run( parsec, potrf_L_sparse_tlr_dp_general_New( parsec, data, params, analysis ),
                            potrf_L_sparse_tlr_dp_general_Destruct, params, "SPARSE_TLR_DP_GENERAL" );
}

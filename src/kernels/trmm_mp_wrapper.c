/*
 * Copyright (c) 2010-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * Adapted for HiCMA TRMM LLN-only path.
 */
#include "hicma_parsec.h"
#include "trmm_mp_LLN.h"

/* 1 if every tile (m, n) of an mt x nt tile matrix, its lower triangle only
 * if lower, has a decision the TRMM supports */
static int hicma_parsec_trmm_decisions_ok(const uint16_t *decisions, int NT, int mt, int nt,
                                          int lower, int myrank, const char *name)
{
    for (int n = 0; n < nt; n++) {
        for (int m = lower ? n : 0; m < mt; m++) {
            uint16_t d = decisions[(size_t)n * NT + m];
            if (DENSE_DP != d && DENSE_SP != d && DENSE_HP != d) {
                if (0 == myrank) {
                    fprintf(stderr, "hicma_parsec_trmm_New: tile (%d, %d) of %s has precision decision %u; "
                            "the TRMM supports DENSE_DP, DENSE_SP and DENSE_HP only\n",
                            m, n, name, (unsigned)d);
                }
                return 0;
            }
        }
    }
    return 1;
}

/**
 * @brief Taskpool of the mixed-precision TRMM B = alpha * A * B
 *
 * Only side = dplasmaLeft, uplo = dplasmaLower and trans = dplasmaNoTrans are
 * supported, with diag = dplasmaUnit or dplasmaNonUnit. Every tile of A and
 * B must be in the storage precision of its decision (params->decisions,
 * params->decisionsB), as parsec_datatype_convert_dense_adaptive leaves it.
 * The computation is not done by this call.
 *
 * @param[in] side, uplo, trans, diag See above
 * @param[in] alpha Scalar alpha
 * @param[in] A N-by-N lower triangular matrix in NB-by-NB tiles, params->NT
 *            tiles per dimension. Its strictly upper part is not referenced,
 *            nor its diagonal if diag = dplasmaUnit.
 * @param[in,out] B N-by-K matrix in NB-by-NB tiles, at most params->KT tile
 *            columns. On exit, alpha * A * B.
 * @param[in] data HiCMA data (GPU workspace)
 * @param[in] params HiCMA parameters and precision decisions: DENSE_DP,
 *            DENSE_SP or DENSE_HP for every tile of A's lower triangle and of B
 *
 * @return The taskpool, to add with parsec_context_add_taskpool() and free
 *         with hicma_parsec_trmm_Destruct(); NULL if the arguments are not
 *         supported.
 */
parsec_taskpool_t*
hicma_parsec_trmm_New( dplasma_enum_t side,  dplasma_enum_t uplo,
                       dplasma_enum_t trans, dplasma_enum_t diag,
                       double alpha,
                       const parsec_tiled_matrix_t *A,
                       parsec_tiled_matrix_t *B,
                       hicma_parsec_data_t *data,
                       hicma_parsec_params_t *params )
{
    if (side != dplasmaLeft || uplo != dplasmaLower ||
        trans != dplasmaNoTrans ||
        (diag != dplasmaUnit && diag != dplasmaNonUnit)) {
        dplasma_error("hicma_parsec_trmm_New", "only Left/Lower/NoTrans with Unit|NonUnit is supported");
        return NULL;
    }

    /* The kernels, arenas and memory pools use NB-by-NB tiles, and the
     * decision arrays NT tile rows and at most KT tile columns */
    if (A->mb != A->nb || B->mb != A->mb || B->nb != A->mb) {
        dplasma_error("hicma_parsec_trmm_New", "A and B must have the same square tiles");
        return NULL;
    }
    if (A->m != A->n || A->n != B->m || A->mt != params->NT || B->nt > params->KT) {
        dplasma_error("hicma_parsec_trmm_New", "A must be N-by-N with NT tile rows, and B N-by-K with at most KT tile columns");
        return NULL;
    }

    /* The kernels and arenas handle FP64, FP32 and FP16 tiles only (no FP8
     * or low-rank decision). Every rank holds all the decisions, so they all
     * return NULL together. */
    if (!hicma_parsec_trmm_decisions_ok(params->decisions, params->NT, A->mt, A->nt, 1, B->super.myrank, "A") ||
        !hicma_parsec_trmm_decisions_ok(params->decisionsB, params->NT, B->mt, B->nt, 0, B->super.myrank, "B")) {
        return NULL;
    }

    /* GPU device management */
    int nb = 0, *dev_index;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Find and initialize available GPU devices */
    hicma_parsec_find_cuda_devices(&dev_index, &nb);

#if !GPU_BUFFER_ONCE
    /* Allocate GPU workspace buffers if not already allocated */
    gpu_temporay_buffer_init(data, params, A->mb, A->nb, 0, 0);
#endif
#endif

    parsec_trmm_mp_LLN_taskpool_t *parsec_trmm = parsec_trmm_mp_LLN_new(
        side, uplo, trans, diag, alpha, A, B, params);

    if (NULL == parsec_trmm) {
        return NULL;
    }

    /* Operand copies of B (see read_B in trmm_mp_LLN.jdf):
     *  2: B(k,n) is stored in another precision than its decision
     *     (--accumulation_fp32); all its GEMMs read a converted copy.
     *  1: some GEMMs reading it run on other ranks; they read a copy, and those
     *     on this rank, the rows congruent to k modulo P, read it in place.
     *     Used only if the distribution matches that rule (block-cyclic with
     *     one tile row per process row).
     *  0: all its GEMMs read it in place. */
    {
        int mt = B->mt, nt = B->nt;
        uint8_t *copy = (uint8_t *)calloc((size_t)mt * nt, sizeof(uint8_t));
        for (int n = 0; n < nt; n++) {
            for (int k = 0; k < mt - 1; k++) {
                size_t idx = (size_t)n * params->NT + k;
                int c = (hicma_parsec_trmm_storage_precision(params, params->decisionsB, idx)
                         != params->decisionsB[idx]) ? 2 : 0;
                if (0 == c) {
                    uint32_t owner = B->super.rank_of(&B->super, k, n);
                    int remote = 0, stride_ok = 1;
                    for (int j = k + 1; j < mt; j++) {
                        int same = (B->super.rank_of(&B->super, j, n) == owner);
                        remote |= !same;
                        stride_ok &= (same == (0 == (j - k) % params->P));
                    }
                    c = (remote && stride_ok) ? 1 : 0;
                }
                copy[(size_t)n * mt + k] = (uint8_t)c;
            }
        }
        parsec_trmm->_g_trmm_copy = copy;
    }

    /* Full matrix memory pools for different precision types */
    parsec_trmm->_g_p_work_full_dp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( parsec_trmm->_g_p_work_full_dp, A->mb * A->mb * sizeof(double) );

    parsec_trmm->_g_p_work_full_sp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( parsec_trmm->_g_p_work_full_sp, A->mb * A->mb * sizeof(float) );

    parsec_trmm->_g_p_work_full_hp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( parsec_trmm->_g_p_work_full_hp, A->mb * A->mb * sizeof(float) / 2 );

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Configure GPU workspace and device information */
    parsec_trmm->_g_ws_gpu = (void *)data->ws_gpu;
    parsec_trmm->_g_nb_cuda_devices = nb;
    parsec_trmm->_g_cuda_device_index = dev_index;
#endif

    /* Double precision full matrix arena */
    parsec_add2arena(&parsec_trmm->arenas_datatypes[PARSEC_trmm_mp_LLN_FULL_DP_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Single precision full matrix arena */
    parsec_add2arena(&parsec_trmm->arenas_datatypes[PARSEC_trmm_mp_LLN_FULL_SP_ADT_IDX],
            parsec_datatype_float_t, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Half precision full matrix arena */
    parsec_add2arena(&parsec_trmm->arenas_datatypes[PARSEC_trmm_mp_LLN_FULL_HP_ADT_IDX],
            MPI_BYTE, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb*2, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    return (parsec_taskpool_t*)parsec_trmm;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_double
 *
 *  hicma_parsec_trmm_Destruct - Free taskpool data structure
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa hicma_parsec_trmm_New
 * @sa hicma_parsec_trmm
 *
 ******************************************************************************/
void
hicma_parsec_trmm_Destruct( parsec_taskpool_t *tp )
{
    parsec_trmm_mp_LLN_taskpool_t *otrmm = (parsec_trmm_mp_LLN_taskpool_t *)tp;

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Clean up GPU resources if devices were used */
    if( otrmm->_g_nb_cuda_devices > 0 ) {
#if !GPU_BUFFER_ONCE
        /* Free GPU workspace memory if not managed globally */
        workspace_memory_free( otrmm->_g_ws_gpu );
#endif

        /* Free GPU device index array */
        if( NULL != otrmm->_g_cuda_device_index )
            free(otrmm->_g_cuda_device_index);
    }
#endif

    parsec_del2arena( &otrmm->arenas_datatypes[PARSEC_trmm_mp_LLN_FULL_DP_ADT_IDX] );
    parsec_del2arena( &otrmm->arenas_datatypes[PARSEC_trmm_mp_LLN_FULL_SP_ADT_IDX] );
    parsec_del2arena( &otrmm->arenas_datatypes[PARSEC_trmm_mp_LLN_FULL_HP_ADT_IDX] );

    parsec_private_memory_fini( otrmm->_g_p_work_full_dp );
    parsec_private_memory_fini( otrmm->_g_p_work_full_sp );
    parsec_private_memory_fini( otrmm->_g_p_work_full_hp );
    free( otrmm->_g_p_work_full_dp );
    free( otrmm->_g_p_work_full_sp );
    free( otrmm->_g_p_work_full_hp );
    free( otrmm->_g_trmm_copy );

    parsec_taskpool_free(tp);
}

/**
 * @brief Mixed-precision TRMM B = alpha * A * B, run to completion
 *
 * Same arguments as hicma_parsec_trmm_New, plus the PaRSEC context.
 *
 * @return 0 on success, -1 for an unsupported side, uplo, trans or diag, -2
 *         if A is not N-by-N with B N-by-K, -101 if the taskpool could not be
 *         created.
 */
int
hicma_parsec_trmm( parsec_context_t *parsec,
                   dplasma_enum_t side,  dplasma_enum_t uplo,
                   dplasma_enum_t trans, dplasma_enum_t diag,
                   double alpha,
                   const parsec_tiled_matrix_t *A,
                   parsec_tiled_matrix_t *B,
                   hicma_parsec_data_t *data,
                   hicma_parsec_params_t *params)
{
    parsec_taskpool_t *parsec_trmm = NULL;

    if (side != dplasmaLeft || uplo != dplasmaLower ||
        trans != dplasmaNoTrans ||
        (diag != dplasmaUnit && diag != dplasmaNonUnit)) {
        dplasma_error("hicma_parsec_trmm", "only Left/Lower/NoTrans with Unit|NonUnit is supported");
        return -1;
    }

    if ( (A->m != A->n) ||
         (A->n != B->m)) {
        dplasma_error("hicma_parsec_trmm", "illegal matrix A");
        return -2;
    }

    parsec_trmm = hicma_parsec_trmm_New(side, uplo, trans, diag, alpha, A, B, data, params);

    if ( parsec_trmm != NULL )
    {
        parsec_context_add_taskpool( parsec, (parsec_taskpool_t*)parsec_trmm);
        dplasma_wait_until_completion(parsec);
        hicma_parsec_trmm_Destruct( parsec_trmm );
        return 0;
    }
    else {
        return -101;
    }
}

/*
 * Copyright (c) 2010-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * Adapted for HiCMA TRMM LLN-only path.
 */
#include "hicma_parsec.h"
#include "trmm_mp_LLN.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_double
 *
 *  hicma_parsec_trmm_New - Generates parsec taskpool to compute:
 *
 *  B = alpha*op( A )*B or B = alpha*B*op( A ).
 *
 *  WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          Specifies whether A appears on the left or on the right of X:
 *          = dplasmaLeft:  A*X = B
 *          = dplasmaRight: X*A = B
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper triangular or lower triangular:
 *          = dplasmaUpper: Upper triangle of A is stored;
 *          = dplasmaLower: Lower triangle of A is stored.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is transposed, not transposed or ugate transposed:
 *          = dplasmaNoTrans:   A is transposed;
 *          = dplasmaTrans:     A is not transposed;
 *          = dplasmaTrans: A is ugate transposed.
 *
 * @param[in] diag
 *          Specifies whether or not A is unit triangular:
 *          = dplasmaNonUnit: A is non unit;
 *          = dplasmaUnit:    A us unit.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha.
 *
 * @param[in] A
 *          Descriptor of the triangular matrix A of size N-by-N.
 *          The triangular matrix A. If uplo = dplasmaUpper, the leading N-by-N upper triangular
 *          part of the array A contains the upper triangular matrix, and the strictly lower
 *          triangular part of A is not referenced. If uplo = dplasmaLower, the leading N-by-N
 *          lower triangular part of the array A contains the lower triangular matrix, and the
 *          strictly upper triangular part of A is not referenced. If diag = dplasmaUnit, the
 *          diagonal elements of A are also not referenced and are assumed to be 1.
 *
 * @param[in,out] B
 *          Descriptor of the N-by-NRHS right hand side B
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if return value = 0, the N-by-NRHS solution matrix X.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroy with dplasma_trmm_mp_Destruct();
 *
 *******************************************************************************
 *
 * @sa hicma_parsec_trmm
 * @sa hicma_parsec_trmm_Destruct
 *
 ******************************************************************************/
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

    if (A->mb != A->nb) {
        dplasma_error("hicma_parsec_trmm_New", "mb should be equal to nb");
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
    //fprintf(stderr, "GPU_BUFFER_ONCE need enable in trmm\n");
    //return NULL;
#endif
#endif

    parsec_trmm_mp_LLN_taskpool_t *parsec_trmm = parsec_trmm_mp_LLN_new(
        side, uplo, trans, diag, alpha, A, B, params);

    if (NULL == parsec_trmm) {
        return NULL;
    }

    /* Operand copies of B (see read_B in trmm_mp_LLN.jdf): 2 if B(k,n) is
     * stored in another precision than its decision (--accumulation_fp32),
     * read converted by all its GEMMs; 1 if a GEMM reading it runs on another
     * rank, which reads a copy; 0 otherwise, all reading it in place. */
    {
        int mt = B->mt, nt = B->nt;
        uint8_t *copy = (uint8_t *)calloc((size_t)mt * nt, sizeof(uint8_t));
        for (int n = 0; n < nt; n++) {
            for (int k = 0; k < mt - 1; k++) {
                size_t idx = (size_t)n * params->NT + k;
                int c = (hicma_parsec_trmm_storage_precision(params, params->decisionsB, idx)
                         != params->decisionsB[idx]) ? 2 : 0;
                uint32_t owner = B->super.rank_of(&B->super, k, n);
                for (int j = k + 1; !c && j < mt; j++) {
                    c = (B->super.rank_of(&B->super, j, n) != owner);
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
 *******************************************************************************
 *
 * @ingroup dplasma_double
 *
 *  hicma_parsec_trmm - Computes:
 *
 *  B = alpha*op( A )*B or B = alpha*B*op( A ).
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] side
 *          Specifies whether A appears on the left or on the right of X:
 *          = dplasmaLeft:  A*X = B
 *          = dplasmaRight: X*A = B
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper triangular or lower triangular:
 *          = dplasmaUpper: Upper triangle of A is stored;
 *          = dplasmaLower: Lower triangle of A is stored.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is transposed, not transposed or ugate transposed:
 *          = dplasmaNoTrans:   A is transposed;
 *          = dplasmaTrans:     A is not transposed;
 *          = dplasmaTrans: A is ugate transposed.
 *
 * @param[in] diag
 *          Specifies whether or not A is unit triangular:
 *          = dplasmaNonUnit: A is non unit;
 *          = dplasmaUnit:    A us unit.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha.
 *
 * @param[in] A
 *          Descriptor of the triangular matrix A of size N-by-N.
 *          The triangular matrix A. If uplo = dplasmaUpper, the leading N-by-N upper triangular
 *          part of the array A contains the upper triangular matrix, and the strictly lower
 *          triangular part of A is not referenced. If uplo = dplasmaLower, the leading N-by-N
 *          lower triangular part of the array A contains the lower triangular matrix, and the
 *          strictly upper triangular part of A is not referenced. If diag = dplasmaUnit, the
 *          diagonal elements of A are also not referenced and are assumed to be 1.
 *
 * @param[in,out] B
 *          Descriptor of the N-by-NRHS right hand side B
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if return value = 0, the N-by-NRHS solution matrix X.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa hicma_parsec_trmm_New
 * @sa hicma_parsec_trmm_Destruct
 *
 ******************************************************************************/
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

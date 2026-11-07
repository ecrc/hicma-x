/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

/**
 * @file potrf_L_dense_mp_gpu_fp8_wrapper.c
 * @brief PaRSEC wrapper of potrf_L_dense_mp_gpu_fp8.jdf (kind 8, DENSE_MP_GPU_FP8: kind 5 with a static FP8 band)
 *
 * Timing hooks, chore selection, memory pools, GPU discovery and the driver
 * are shared with the other Cholesky wrappers (potrf_common.h); this file
 * keeps what is specific to this kind: the GPU evaluate hooks, the pools, the
 * arenas and the constructor arguments.
 */

#include "hicma_parsec.h"
#include "hicma_kernel_time.h"
#include "potrf_common.h"
#include "potrf_L_dense_mp_gpu_fp8.h"

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

/**
 * @brief GPU kernel evaluation function for POTRF operations
 * 
 * This function determines whether to use GPU acceleration for POTRF operations.
 * Currently, it always returns DONE, indicating that GPU kernels should be used
 * for all POTRF operations when GPU support is available.
 * 
 * @param task POTRF task structure
 * @return PARSEC_HOOK_RETURN_DONE to use GPU kernel
 */
static parsec_hook_return_t evaluate_gpu_potrf(parsec_task_t* task) {
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief GPU kernel evaluation function for TRSM operations
 * 
 * This function determines whether to use GPU acceleration for TRSM operations
 * based on the problem size. GPU kernels are used when the matrix size difference
 * (m-k) is smaller than the dense band size, indicating that the operation
 * involves dense matrix regions that benefit from GPU acceleration.
 * 
 * @param task TRSM task structure containing matrix dimensions
 * @return PARSEC_HOOK_RETURN_DONE to use GPU kernel, PARSEC_HOOK_RETURN_NEXT for CPU
 */
static parsec_hook_return_t evaluate_gpu_trsm(parsec_task_t* task) {
    int m = ((__parsec_potrf_L_dense_mp_gpu_fp8_potrf_dtrsm_task_t *)task)->locals.m.value;
    int k = ((__parsec_potrf_L_dense_mp_gpu_fp8_potrf_dtrsm_task_t *)task)->locals.k.value;
    int band_size_dense = ((__parsec_potrf_L_dense_mp_gpu_fp8_potrf_dtrsm_task_t *)task)->locals.band_size_dense_local.value;
    
    // Use GPU for operations within the dense band region
    if( m-k < band_size_dense )
        return PARSEC_HOOK_RETURN_DONE;
    else
        return PARSEC_HOOK_RETURN_NEXT;
}

/**
 * @brief GPU evaluation function for TRSM_SEND operation
 *
 * Uses the same dense-band criterion as TRSM. The CUDA/HIP body of
 * potrf_dtrsm_send returns PARSEC_HOOK_RETURN_NEXT for low-rank tiles, and
 * PaRSEC re-queues a GPU task that returns NEXT on the same GPU stream
 * instead of falling back to the CPU chore, so low-rank tiles must be
 * kept off the GPU here or the factorization hangs.
 *
 * @param task TRSM_SEND task to evaluate
 * @return PARSEC_HOOK_RETURN_DONE for GPU execution, PARSEC_HOOK_RETURN_NEXT for CPU
 */
static parsec_hook_return_t evaluate_gpu_trsm_send(parsec_task_t* task) {
    int m = ((__parsec_potrf_L_dense_mp_gpu_fp8_potrf_dtrsm_send_task_t *)task)->locals.m.value;
    int k = ((__parsec_potrf_L_dense_mp_gpu_fp8_potrf_dtrsm_send_task_t *)task)->locals.k.value;
    int band_size_dense = ((__parsec_potrf_L_dense_mp_gpu_fp8_potrf_dtrsm_send_task_t *)task)->locals.band_size_dense_local.value;

    /* Execute on GPU if within the dense band */
    if( m-k < band_size_dense )
        return PARSEC_HOOK_RETURN_DONE;
    else
        return PARSEC_HOOK_RETURN_NEXT;
}

/**
 * @brief GPU kernel evaluation function for SYRK operations
 * 
 * This function determines whether to use GPU acceleration for SYRK operations.
 * Currently, it always returns DONE, indicating that GPU kernels should be used
 * for all SYRK operations when GPU support is available.
 * 
 * @param task SYRK task structure
 * @return PARSEC_HOOK_RETURN_DONE to use GPU kernel
 */
static parsec_hook_return_t evaluate_gpu_syrk(parsec_task_t* task) {
        return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief GPU kernel evaluation function for GEMM operations
 * 
 * This function determines whether to use GPU acceleration for GEMM operations
 * based on the problem size. GPU kernels are used when the matrix size difference
 * (m-n) is smaller than the dense band size, indicating that the operation
 * involves dense matrix regions that benefit from GPU acceleration.
 * 
 * @param task GEMM task structure containing matrix dimensions
 * @return PARSEC_HOOK_RETURN_DONE to use GPU kernel, PARSEC_HOOK_RETURN_NEXT for CPU
 */
static parsec_hook_return_t evaluate_gpu_gemm(parsec_task_t* task) {
    int m = ((__parsec_potrf_L_dense_mp_gpu_fp8_potrf_dgemm_task_t *)task)->locals.m.value;
    int n = ((__parsec_potrf_L_dense_mp_gpu_fp8_potrf_dgemm_task_t *)task)->locals.n.value;
    int k = ((__parsec_potrf_L_dense_mp_gpu_fp8_potrf_dgemm_task_t *)task)->locals.k.value;
    int band_size_dense = ((__parsec_potrf_L_dense_mp_gpu_fp8_potrf_dgemm_task_t *)task)->locals.band_size_dense_local.value;
    
    // Use GPU for operations within the dense band region
    if( m-n < band_size_dense )
        return PARSEC_HOOK_RETURN_DONE;
    else
        return PARSEC_HOOK_RETURN_NEXT;
}

#endif /* PARSEC_HAVE_DEV_CUDA_SUPPORT */

/**
 * @brief Create the taskpool of DENSE_MP_GPU_FP8
 *
 * @param parsec PaRSEC context
 * @param data Matrix descriptors (dcA/dcAd, dcAr, dcRank, dcFake) and GPU workspace
 * @param params Parameters; params->info is reset to 0 and set by the kernels
 * @return the taskpool, or NULL if the parameters are not supported
 */
parsec_taskpool_t*
potrf_L_dense_mp_gpu_fp8_New( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    int rank = params->rank;
    int uplo = params->uplo;
    int hmb = params->HNB;
    int compmaxrank = params->compmaxrank;
    int storagemaxrank = params->genmaxrank;

    /* The matrix: dcAd when it is dense and contiguous, dcA otherwise */
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&data->dcA;
    if( params->band_size_dense >= params->NT && params->auto_band == 0 && !params->adaptive_memory ) {
        A = (parsec_tiled_matrix_t *)&data->dcAd;
    }
    parsec_tiled_matrix_t *Ar = (parsec_tiled_matrix_t *)&data->dcAr;
    parsec_tiled_matrix_t *Rank = (parsec_tiled_matrix_t *)&data->dcRank;
    parsec_tiled_matrix_t *Fake = (parsec_tiled_matrix_t *)&data->dcFake;

    if( uplo != PlasmaLower ) {
        dplasma_error("potrf_L_dense_mp_gpu_fp8_New", "only PlasmaLower is supported");
        return NULL;
    }
    if( storagemaxrank > compmaxrank ) {
        dplasma_error("potrf_L_dense_mp_gpu_fp8_New", "maxrank for storage larger than maxrank for buffers is not meaningful");
        return NULL;
    }
    if( storagemaxrank > (A->mb/2) && 0 == rank ) {
        fprintf(stderr, RED "Warning: maxrank= %d is larger than half of block size\n" RESET, storagemaxrank);
    }

    int nb = 0, *dev_index = NULL;
    hicma_potrf_gpu_devices(&nb, &dev_index);

    params->info = 0;
    parsec_potrf_L_dense_mp_gpu_fp8_taskpool_t *tp = parsec_potrf_L_dense_mp_gpu_fp8_new( A, Ar, Rank, Fake, params );
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    tp->_g_ws_gpu = (void *)data->ws_gpu;
    tp->_g_nb_cuda_devices = nb;
    tp->_g_cuda_device_index = dev_index;
#endif

    /* Timing hooks on the chore that runs (GPU, recursive or CPU) and the
     * GPU evaluate hooks of this kind */
    {
        hicma_potrf_evaluate_t eval_gpu = { NULL, NULL, NULL, NULL, NULL };
        const hicma_potrf_evaluate_t *eval_gpu_p = NULL;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        eval_gpu.potrf = evaluate_gpu_potrf;
        eval_gpu.trsm = evaluate_gpu_trsm;
        eval_gpu.trsm_send = evaluate_gpu_trsm_send;
        eval_gpu.syrk = evaluate_gpu_syrk;
        eval_gpu.gemm = evaluate_gpu_gemm;
        eval_gpu_p = &eval_gpu;
#endif
        if( 0 != hicma_potrf_install_hooks(&tp->super, params, A, nb, hmb, A->mb, eval_gpu_p) ) {
            parsec_taskpool_free(&tp->super);
            free(dev_index);
            return NULL;
        }
    }

    /* Per-thread workspace pools */
    tp->_g_p_work = hicma_potrf_pool_new( hicma_potrf_hcore_workspace_doubles(A->mb, compmaxrank) * sizeof(double) );
    tp->_g_p_work_rr = hicma_potrf_pool_new( compmaxrank * compmaxrank * sizeof(double) );
    tp->_g_p_work_mbr = hicma_potrf_pool_new( A->mb * compmaxrank * sizeof(double) );
    tp->_g_p_work_full_dp = hicma_potrf_pool_new( (size_t)A->mb * hicma_parsec_max(A->mb, 2 * compmaxrank) * sizeof(double) );
    tp->_g_p_work_full_sp = hicma_potrf_pool_new( (size_t)A->mb * hicma_parsec_max(A->mb, 2 * compmaxrank) * sizeof(float) );
    tp->_g_p_work_full_hp = hicma_potrf_pool_new( (size_t)A->mb * hicma_parsec_max(A->mb, 2 * compmaxrank) * sizeof(float) / 2 );
    tp->_g_p_work_full_fp8 = hicma_potrf_pool_new( (size_t)A->mb * hicma_parsec_max(A->mb, 2 * compmaxrank) * sizeof(float) / 4 );
    tp->_g_p_work_uv_dp = hicma_potrf_pool_new( A->mb * storagemaxrank * 2 * sizeof(double) );
    tp->_g_p_work_uv_sp = hicma_potrf_pool_new( A->mb * storagemaxrank * 2 * sizeof(float) );

    tp->_g_PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );
    if( 0 == tp->_g_PRI_CHANGE ) {
        tp->_g_PRI_CHANGE = A->nt;
    }

    /* Arenas: one per datatype the flows exchange */
    // Configure data type arenas for different precision types and matrix layouts
    parsec_add2arena(&tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_FULL_DP_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    parsec_add2arena(&tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_FULL_SP_ADT_IDX],
            parsec_datatype_float_t, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    parsec_add2arena(&tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_UV_DP_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, storagemaxrank*2, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    parsec_add2arena(&tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_UV_SP_ADT_IDX],
            parsec_datatype_float_t, PARSEC_MATRIX_FULL,
            1, A->mb, storagemaxrank*2, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    parsec_add2arena(&tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_BYTE_ADT_IDX],
            MPI_BYTE, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    parsec_add2arena(&tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_FULL_HP_ADT_IDX],
            MPI_BYTE, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb*2, A->mb, 
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* FP8 tiles sent to FP8 GEMMs, with their per-tile scale (hicma_fp8_tile_bytes) */
    parsec_add2arena(&tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_FULL_FP8_ADT_IDX],
            MPI_BYTE, PARSEC_MATRIX_FULL,
            1, hicma_fp8_tile_bytes(A->mb), 1, hicma_fp8_tile_bytes(A->mb),
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    parsec_add2arena(&tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_AR_ADT_IDX],
            parsec_datatype_int_t, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    if( params->arena_pinned_memory && nb > 0 ) {
        hicma_parsec_arenas_use_pinned_memory(
                tp->arenas_datatypes,
                PARSEC_potrf_L_dense_mp_gpu_fp8_ADT_IDX_MAX);
    }

    return (parsec_taskpool_t*)tp;
}

/**
 * @brief Release everything potrf_L_dense_mp_gpu_fp8_New allocated, then the taskpool
 */
void potrf_L_dense_mp_gpu_fp8_Destruct(parsec_taskpool_t* _tp)
{
    parsec_potrf_L_dense_mp_gpu_fp8_taskpool_t *tp = (parsec_potrf_L_dense_mp_gpu_fp8_taskpool_t*)_tp;
    parsec_memory_pool_t *pools[] = {
        tp->_g_p_work,
        tp->_g_p_work_rr,
        tp->_g_p_work_mbr,
        tp->_g_p_work_full_dp,
        tp->_g_p_work_full_sp,
        tp->_g_p_work_full_hp,
        tp->_g_p_work_full_fp8,
        tp->_g_p_work_uv_dp,
        tp->_g_p_work_uv_sp
    };

    hicma_potrf_forget(_tp);
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    free(tp->_g_cuda_device_index);
    tp->_g_cuda_device_index = NULL;
#endif

    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_FULL_DP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_FULL_SP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_UV_DP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_UV_SP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_BYTE_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_FULL_HP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_FULL_FP8_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_AR_ADT_IDX] );

    parsec_taskpool_free(_tp);

    /* After the taskpool: its repositories may still reference pool elements */
    for( size_t i = 0; i < sizeof(pools) / sizeof(pools[0]); i++ ) {
        hicma_potrf_pool_free(pools[i]);
    }
}

/**
 * @brief Run DENSE_MP_GPU_FP8 on data->dcA (or dcAd); params->info holds the
 *        POTRF info after the reduction done by the caller
 *
 * @return 0, or -1 if the taskpool could not be created
 */
int potrf_L_dense_mp_gpu_fp8( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    return hicma_potrf_run( parsec, potrf_L_dense_mp_gpu_fp8_New( parsec, data, params ),
                            potrf_L_dense_mp_gpu_fp8_Destruct, params, "DENSE_MP_GPU_FP8" );
}

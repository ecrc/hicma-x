/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

/**
 * @file potrf_L_dense_sp_hp_band_wrapper.c
 * @brief PaRSEC wrapper of potrf_L_dense_sp_hp_band.jdf (kind 4, DENSE_SP_HP_BAND: dense Cholesky with single and half precision bands)
 *
 * Timing hooks, chore selection, memory pools, GPU discovery and the driver
 * are shared with the other Cholesky wrappers (potrf_common.h); this file
 * keeps what is specific to this kind: the GPU evaluate hooks, the pools, the
 * arenas and the constructor arguments.
 */

#include "hicma_parsec.h"
#include "hicma_kernel_time.h"
#include "potrf_common.h"
#include "potrf_L_dense_sp_hp_band.h"

/**
 * @brief Create the taskpool of DENSE_SP_HP_BAND
 *
 * @param parsec PaRSEC context
 * @param data Matrix descriptors (dcA/dcAd, dcAr, dcRank, dcFake) and GPU workspace
 * @param params Parameters; params->info is reset to 0 and set by the kernels
 * @return the taskpool, or NULL if the parameters are not supported
 */
parsec_taskpool_t*
parsec_shpotrf_New( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    int rank = params->rank;
    int uplo = params->uplo;

    /* The matrix: dcAd when it is dense and contiguous, dcA otherwise */
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&data->dcA;
    if( params->band_size_dense >= params->NT && params->auto_band == 0 && !params->adaptive_memory ) {
        A = (parsec_tiled_matrix_t *)&data->dcAd;
    }

    if( uplo != PlasmaLower ) {
        dplasma_error("parsec_shpotrf_New", "only PlasmaLower is supported");
        return NULL;
    }
    (void)rank;

    int nb = 0, *dev_index = NULL;
    hicma_potrf_gpu_devices(&nb, &dev_index);

    params->info = 0;
    parsec_potrf_L_dense_sp_hp_band_taskpool_t *tp = parsec_potrf_L_dense_sp_hp_band_new( A, params );
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    tp->_g_ws_gpu = (void *)data->ws_gpu;
    tp->_g_nb_cuda_devices = nb;
    tp->_g_cuda_device_index = dev_index;
#endif

    tp->_g_PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );
    if( 0 == tp->_g_PRI_CHANGE ) {
        tp->_g_PRI_CHANGE = A->nt;
    }

    /* Arenas: one per datatype the flows exchange */
    // Register single precision floating point data type arena for memory management
    // This sets up memory arena for efficient allocation/deallocation of single precision data
    parsec_add2arena(&tp->arenas_datatypes[PARSEC_potrf_L_dense_sp_hp_band_SINGLE_ADT_IDX],
                            parsec_datatype_float_t, PARSEC_MATRIX_FULL,
                            1, A->mb, A->nb, A->mb,
                            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    return (parsec_taskpool_t*)tp;
}

/**
 * @brief Release everything parsec_shpotrf_New allocated, then the taskpool
 */
void parsec_shpotrf_Destruct(parsec_taskpool_t* _tp)
{
    parsec_potrf_L_dense_sp_hp_band_taskpool_t *tp = (parsec_potrf_L_dense_sp_hp_band_taskpool_t*)_tp;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    free(tp->_g_cuda_device_index);
    tp->_g_cuda_device_index = NULL;
#endif

    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_sp_hp_band_SINGLE_ADT_IDX] );

    parsec_taskpool_free(_tp);
}

/**
 * @brief Run DENSE_SP_HP_BAND on data->dcA (or dcAd); params->info holds the
 *        POTRF info after the reduction done by the caller
 *
 * @return 0, or -1 if the taskpool could not be created
 */
int hicma_parsec_shpotrf( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    return hicma_potrf_run( parsec, parsec_shpotrf_New( parsec, data, params ),
                            parsec_shpotrf_Destruct, params, "DENSE_SP_HP_BAND" );
}

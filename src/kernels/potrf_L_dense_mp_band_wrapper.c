/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

/**
 * @file potrf_L_dense_mp_band_wrapper.c
 * @brief Wrapper Functions for Mixed-Precision Cholesky Factorization
 * 
 * This file provides the high-level interface functions for the mixed-precision
 * Cholesky factorization algorithm implemented in the PaRSEC runtime. It includes
 * functions for creating, executing, and destroying the factorization taskpool.
 * 
 * The wrapper functions handle:
 * - Taskpool creation and initialization
 * - Memory management for temporary buffers
 * - GPU device detection and workspace allocation
 * - Data type arena setup for mixed precision
 * - Cleanup and resource deallocation
 * 
 * @version 1.0
 */

#include "hicma_parsec.h"
#include "potrf_L_dense_mp_band.h"


/**
 *******************************************************************************
 *
 * @ingroup dplasma_double
 *
 * @brief Create Mixed-Precision Cholesky Factorization Taskpool
 * 
 * This function creates a PaRSEC taskpool for computing the Cholesky factorization
 * of a symmetric positive definite matrix using a mixed-precision band-based approach.
 * The factorization has the form:
 *
 *    \f[ A = L \times L^T \f]
 *
 * where L is a lower triangular matrix. The algorithm uses different precisions
 * based on the distance from the diagonal:
 * - Double precision for diagonal and near-diagonal blocks
 * - Single precision for medium-distance blocks  
 * - Half precision for far-off-diagonal blocks
 *
 * @details The mixed-precision strategy optimizes the trade-off between accuracy
 * and performance by using higher precision where numerical stability is critical
 * and lower precision where the impact on accuracy is minimal.
 *
 * @warning The computations are not performed by this call. The taskpool must be
 *          executed using parsec_context_add_taskpool() and related functions.
 *
 * @note For recursive DAGs, ensure the recursive tile size is set and synchronize
 *       taskpool IDs after computations since recursive calls are currently local.
 *
 * @reference Hierarchical DAG Scheduling for Hybrid Distributed Systems; Wu, Wei and
 *           Bouteiller, Aurelien and Bosilca, George and Faverge, Mathieu and Dongarra,
 *           Jack. 29th IEEE International Parallel & Distributed Processing Symposium,
 *           May 2015. (https://hal.inria.fr/hal-0107835)
 *
 *******************************************************************************
 *
 * @param[in] parsec
 *          The PaRSEC context for the application.
 *
 * @param[in] data
 *          Data structure containing matrix descriptors and workspace.
 *
 * @param[in] params
 *          Algorithm parameters including:
 *          - uplo: Matrix storage format (PlasmaLower/PlasmaUpper)
 *          - band_size_dense_dp: Double precision band size
 *          - band_size_dense_sp: Single precision band size
 *          - datatype_convert: Enable data type conversion
 *          - lookahead: Lookahead depth for scheduling
 *          - HNB: Recursive threshold size
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given or memory allocation fails.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It must be
 *          destroyed with hicma_parsec_hsdpotrf_Destruct().
 *
 *******************************************************************************
 * @sa hicma_parsec_hsdpotrf
 * @sa hicma_parsec_hsdpotrf_Destruct
 * @sa dplasma_cpotrf_New
 * @sa hicma_parsec_hsdpotrf_New
 * @sa dplasma_spotrf_New
 *
 ******************************************************************************/
parsec_taskpool_t*
hicma_parsec_hsdpotrf_New( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    /* Extract algorithm parameters for clarity */
    int uplo = params->uplo;
    int lookahead = params->lookahead;
    int datatype_convert = params->datatype_convert;
    int band_size_double = params->band_size_dense_dp;
    int band_size_single = params->band_size_dense_sp;
    int maxrank = params->maxrank;
    
    /* Select appropriate matrix descriptor based on memory requirements */
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&data->dcA;
    if( params->band_size_dense >= params->NT && params->auto_band == 0 && !params->adaptive_memory ) {
        A = (parsec_tiled_matrix_t *)&data->dcAd;  /* Use double precision descriptor */
    }

    /* Validate input arguments */
    if (uplo != PlasmaLower) {
        dplasma_error("hicma_parsec_hsdpotrf_New", "illegal value of uplo");
        return NULL /*-1*/;
    }

    /* Ensure band size consistency: double precision band <= single precision band */
    assert( band_size_double <= band_size_single );

    /* Initialize GPU device detection and workspace */
    int nb = 0, *dev_index;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Detect available CUDA/HIP devices for GPU acceleration */
    hicma_parsec_find_cuda_devices( &dev_index, &nb);

#if !GPU_BUFFER_ONCE
    /* Allocate GPU workspace if not already allocated */
    gpu_temporay_buffer_init( data, params, A->mb, A->nb, maxrank, params->kind_of_cholesky );
#endif /* GPU_BUFFER_ONCE */
#endif

    /* Ensure single precision band covers entire matrix when no GPU is available */
    if( 0 == nb )
        assert( band_size_single >= A->lmt );

    /* Initialize error code and create taskpool */
    params->info = 0;
    parsec_potrf_L_dense_mp_band_taskpool_t *parsec_dpotrf = NULL;
    parsec_dpotrf = parsec_potrf_L_dense_mp_band_new( A, params );

    /* Set priority change threshold for task scheduling */
    parsec_dpotrf->_g_PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );
    if(0 == parsec_dpotrf->_g_PRI_CHANGE)
          parsec_dpotrf->_g_PRI_CHANGE = A->nt;

    /* Initialize memory pools for temporary data type conversion buffers */
    parsec_dpotrf->_g_p_work_double = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( parsec_dpotrf->_g_p_work_double, A->mb * A->nb * sizeof(double) );

    parsec_dpotrf->_g_p_work_single = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( parsec_dpotrf->_g_p_work_single, A->mb * A->nb * sizeof(float) );

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Set up GPU workspace and device information */
    parsec_dpotrf->_g_ws_gpu = (void *)data->ws_gpu;
    parsec_dpotrf->_g_nb_cuda_devices = nb;
    parsec_dpotrf->_g_cuda_device_index = dev_index;
#endif

    /* Set up data type arenas for mixed precision support */
    /* Double precision arena */
    parsec_add2arena(&parsec_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_band_DOUBLE_ADT_IDX],
                            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
                            1, A->mb, A->nb, A->mb,
                            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Single precision arena */
    parsec_add2arena(&parsec_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_band_SINGLE_ADT_IDX],
                            parsec_datatype_float_t, PARSEC_MATRIX_FULL,
                            1, A->mb, A->nb, A->mb,
                            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Half precision arena (using MPI_BYTE for 16-bit data) */
    parsec_add2arena(&parsec_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_band_HALF_ADT_IDX],
                            MPI_BYTE, PARSEC_MATRIX_FULL,
                            1, A->mb*2, A->nb, A->mb*2,
                            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    return (parsec_taskpool_t *)parsec_dpotrf;;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_double
 *
 * @brief Destroy Mixed-Precision Cholesky Factorization Taskpool
 * 
 * This function properly deallocates all resources associated with a mixed-precision
 * Cholesky factorization taskpool created by hicma_parsec_hsdpotrf_New(). It handles
 * cleanup of:
 * - GPU workspace and device information
 * - Memory pools for temporary buffers
 * - Data type arenas for mixed precision
 * - Taskpool structure itself
 *
 * @warning The taskpool must not be used after calling this function.
 *
 *******************************************************************************
 *
 * @param[in,out] tp
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore and all associated
 *          resources have been freed.
 *
 *******************************************************************************
 *
 * @sa hicma_parsec_hsdpotrf_New
 * @sa hicma_parsec_hsdpotrf
 *
 ******************************************************************************/
void
hicma_parsec_hsdpotrf_Destruct( parsec_taskpool_t *tp )
{
    parsec_potrf_L_dense_mp_band_taskpool_t *parsec_dpotrf = (parsec_potrf_L_dense_mp_band_taskpool_t *)tp;

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Clean up GPU resources if devices were used */
    if( parsec_dpotrf->_g_nb_cuda_devices > 0 ) {
#if !GPU_BUFFER_ONCE 
        /* Free GPU workspace memory if not managed globally */
        workspace_memory_free( parsec_dpotrf->_g_ws_gpu );
#endif

        /* Free device index array */
        if( NULL != parsec_dpotrf->_g_cuda_device_index )
            free(parsec_dpotrf->_g_cuda_device_index);
    }
#endif

    /* Clean up data type arenas for mixed precision support */
    parsec_del2arena( &parsec_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_band_DOUBLE_ADT_IDX] );
    parsec_del2arena( &parsec_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_band_SINGLE_ADT_IDX] );
    parsec_del2arena( &parsec_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_band_HALF_ADT_IDX] );

    /* Clean up memory pools for temporary data type conversion buffers */
    parsec_private_memory_fini( parsec_dpotrf->_g_p_work_double );
    parsec_private_memory_fini( parsec_dpotrf->_g_p_work_single );

    /* Free the taskpool structure itself */
    parsec_taskpool_free(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_double
 *
 * @brief Execute Mixed-Precision Cholesky Factorization
 * 
 * This function performs the complete mixed-precision Cholesky factorization
 * of a symmetric positive definite matrix A. The factorization has the form:
 *
 *    \f[ A = L \times L^T \f]
 *
 * where L is a lower triangular matrix. The algorithm uses a band-based
 * mixed-precision strategy to optimize the trade-off between accuracy and
 * performance.
 *
 * @details The function creates a taskpool, executes the factorization,
 * and properly cleans up all resources. It handles both regular and
 * recursive factorization modes.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The PaRSEC context of the application that will run the operation.
 *
 * @param[in] data
 *          Data structure containing matrix descriptors and workspace.
 *
 * @param[in] params
 *          Algorithm parameters including precision bands, lookahead depth,
 *          and other configuration options.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameter is incorrect.
 *          \retval 0 on success.
 *          \retval > 0 if the leading minor of order i of A is not positive
 *          definite, so the factorization could not be completed, and the
 *          solution has not been computed. Info will be equal to i on the node
 *          that owns the diagonal element (i,i), and 0 on all other nodes.
 *
 *******************************************************************************
 *
 * @sa hicma_parsec_hsdpotrf_New
 * @sa hicma_parsec_hsdpotrf_Destruct
 * @sa dplasma_cpotrf
 * @sa hicma_parsec_hsdpotrf
 * @sa dplasma_spotrf
 *
 ******************************************************************************/
int hicma_parsec_hsdpotrf( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    parsec_taskpool_t *parsec_dpotrf = NULL;

    /* Print start message on rank 0 if verbose mode is enabled */
    if( 0 == params->rank && params->verbose )
        printf(MAG "DENSE_MP_BAND start\n" RESET);

    /* Create the mixed-precision Cholesky factorization taskpool */
    parsec_dpotrf = hicma_parsec_hsdpotrf_New( parsec, data, params ); 

    if ( parsec_dpotrf != NULL )
    {
        /* Execute the factorization */
        parsec_context_add_taskpool( parsec, (parsec_taskpool_t*)parsec_dpotrf);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        
        /* Clean up resources */
        hicma_parsec_hsdpotrf_Destruct( parsec_dpotrf );

        /* Synchronize taskpool IDs for recursive DAGs if needed */
        if( params->HNB > 0 && params->HNB < params->NB ) {
            parsec_taskpool_sync_ids(); /* recursive DAGs are not synchronous on ids */
        }
    }

    return 0; 
}

/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/
#include "hicma_parsec.h"
#include "potrf_L_dense_sp_hp_band.h"


/**
 *******************************************************************************
 *
 * @ingroup dplasma_float
 *
 * parsec_shpotrf_New - Generates the taskpool that Computes the Cholesky
 * factorization of a symmetric positive definite (or Hermitian positive
 * definite in the complex case) matrix A, with or without recursive calls.
 * The factorization has the form
 *
 *    \f[ A = \{_{L\times L^H, if uplo = PlasmaLower}^{U^H\times U, if uplo = PlasmaUpper} \f]
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *
 * WARNING: The computations are not done by this call.
 *
 * If you want to enable the recursive DAGs, don't forget to set the recursive
 * tile size and to synchonize the taskpool ids after the computations since those
 * are for now local. You can follow the code of parsec_shpotrf_rec() as an
 * example to do this.
 *
 * Hierarchical DAG Scheduling for Hybrid Distributed Systems; Wu, Wei and
 * Bouteiller, Aurelien and Bosilca, George and Faverge, Mathieu and Dongarra,
 * Jack. 29th IEEE International Parallel & Distributed Processing Symposium,
 * May 2015. (https://hal.inria.fr/hal-0107835)
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is referenced;
 *          = PlasmaLower: Lower triangle of A is referenced.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the uplo part of A is overwritten with the factorized
 *          matrix.
 *
 * @param[out] info
 *          Address where to store the output information of the factorization,
 *          this is not synchronized between the nodes, and might not be set
 *          when function exists.
 *          On DAG completion:
 *              - info = 0 on all nodes if successful.
 *              - info > 0 if the leading minor of order i of A is not positive
 *                definite, so the factorization could not be completed, and the
 *                solution has not been computed. Info will be equal to i on the
 *                node that owns the diagonal element (i,i), and 0 on all other
 *                nodes.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroy with parsec_shpotrf_Destruct();
 *
 *******************************************************************************
 * @sa parsec_shpotrf
 * @sa parsec_shpotrf_Destruct
 * @sa dplasma_cpotrf_New
 * @sa parsec_shpotrf_New
 * @sa parsec_shpotrf_New
 *
 ******************************************************************************/
parsec_taskpool_t*
parsec_shpotrf_New( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    // Extract parameters for single precision half precision banded Cholesky factorization
    int uplo = params->uplo;                    // Matrix storage format: PlasmaUpper or PlasmaLower
    int band_size = params->band_size_dense_sp; // Band size for dense single precision operations
    int lookahead = params->lookahead;          // Lookahead parameter for pipeline optimization (number of tasks to schedule ahead)
    int maxrank = params->maxrank;              // Maximum rank for low-rank approximations in hierarchical matrices
    
    // Select appropriate matrix descriptor based on memory constraints and band size
    // Default to single precision matrix descriptor
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&data->dcA;
    // Use double precision matrix if band size is large enough and memory allows
    // This optimization uses higher precision for better numerical stability when possible
    if( params->band_size_dense >= params->NT && params->auto_band == 0 && !params->adaptive_memory ) {
        A = (parsec_tiled_matrix_t *)&data->dcAd;
    } 

    /* Validate input arguments - only lower triangular storage is supported */
    if (uplo != PlasmaLower) {
        dplasma_error("parsec_shpotrf_New", "illegal value of uplo");
        return NULL /*-1*/;
    }

    // Initialize GPU device management variables
    int nb = 0, *dev_index;  // nb: number of GPU devices, dev_index: array of device indices
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /** Discover and enumerate available CUDA/HIP devices for GPU acceleration */
    // This function populates dev_index with available GPU device IDs and sets nb to count
    hicma_parsec_find_cuda_devices( &dev_index, &nb);

#if !GPU_BUFFER_ONCE
    /* Initialize GPU workspace buffers for temporary storage during factorization */
    // Allocate GPU memory buffers for intermediate computations and data transfers
    gpu_temporay_buffer_init( data, params, A->mb, A->nb, maxrank, params->kind_of_cholesky );
#endif /* GPU_BUFFER_ONCE */

#endif

    // Initialize factorization status and create task pool
    params->info = 0;  // Reset info flag to indicate no errors initially
    parsec_potrf_L_dense_sp_hp_band_taskpool_t *parsec_dpotrf = NULL;
    // Create the task pool that will contain all Cholesky factorization tasks
    parsec_dpotrf = parsec_potrf_L_dense_sp_hp_band_new( A, params ); 

    // Configure task priority scheduling for optimal performance
    // Get priority limit for POTRF tasks to optimize task scheduling
    parsec_dpotrf->_g_PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );
    // If no specific priority limit is set, use the number of tiles as fallback
    if(0 == parsec_dpotrf->_g_PRI_CHANGE)
          parsec_dpotrf->_g_PRI_CHANGE = A->nt;

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    // Configure GPU workspace and device information for hybrid CPU-GPU execution
        parsec_dpotrf->_g_ws_gpu = (void *)data->ws_gpu;           // GPU workspace pointer for temporary data
        parsec_dpotrf->_g_nb_cuda_devices = nb;                    // Number of available GPU devices
        parsec_dpotrf->_g_cuda_device_index = dev_index;           // Array of GPU device indices for load balancing
#endif

    // Register single precision floating point data type arena for memory management
    // This sets up memory arena for efficient allocation/deallocation of single precision data
    parsec_add2arena(&parsec_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_sp_hp_band_SINGLE_ADT_IDX],
                            parsec_datatype_float_t, PARSEC_MATRIX_FULL,
                            1, A->mb, A->nb, A->mb,
                            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    return (parsec_taskpool_t *)parsec_dpotrf;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_float
 *
 *  parsec_shpotrf_Destruct - Free the data structure associated to an taskpool
 *  created with parsec_shpotrf_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa parsec_shpotrf_New
 * @sa parsec_shpotrf
 *
 ******************************************************************************/
void
parsec_shpotrf_Destruct( parsec_taskpool_t *tp )
{
    // Cast to specific task pool type for proper cleanup
    // Convert generic taskpool pointer to specific POTRF taskpool type
    parsec_potrf_L_dense_sp_hp_band_taskpool_t *parsec_dpotrf = (parsec_potrf_L_dense_sp_hp_band_taskpool_t *)tp;

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    // Clean up GPU resources if devices were used
    if( parsec_dpotrf->_g_nb_cuda_devices > 0 ) {
#if !GPU_BUFFER_ONCE
        // Free GPU workspace memory if not using persistent buffers
        // Only free if buffers are not reused across multiple factorizations
        workspace_memory_free( parsec_dpotrf->_g_ws_gpu );
#endif

        // Free device index array that was allocated during initialization
        if( NULL != parsec_dpotrf->_g_cuda_device_index )
            free(parsec_dpotrf->_g_cuda_device_index);
    }
#endif

    // Clean up data type arena
    // Remove the single precision floating point arena from memory management
    parsec_del2arena( &parsec_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_sp_hp_band_SINGLE_ADT_IDX] );

    // Free the task pool itself and all associated resources
    parsec_taskpool_free(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_float
 *
 * parsec_shpotrf - Computes the Cholesky factorization of a symmetric positive
 * definite (or Hermitian positive definite in the complex case) matrix A.
 * The factorization has the form
 *
 *    \f[ A = \{_{L\times L^H, if uplo = PlasmaLower}^{U^H\times U, if uplo = PlasmaUpper} \f]
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is referenced;
 *          = PlasmaLower: Lower triangle of A is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the uplo part of A is overwritten with the factorized
 *          matrix.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *          \retval > 0 if the leading minor of order i of A is not positive
 *          definite, so the factorization could not be completed, and the
 *          solution has not been computed. Info will be equal to i on the node
 *          that owns the diagonal element (i,i), and 0 on all other nodes.
 *
 *******************************************************************************
 *
 * @sa parsec_shpotrf_New
 * @sa parsec_shpotrf_Destruct
 * @sa dplasma_cpotrf
 * @sa parsec_shpotrf
 * @sa parsec_shpotrf
 *
 ******************************************************************************/
int hicma_parsec_shpotrf( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    parsec_taskpool_t *parsec_dpotrf = NULL;

    // Print start message on rank 0 if verbose mode is enabled
    // Only the master process (rank 0) prints status messages to avoid output duplication
    if( 0 == params->rank && params->verbose )
        printf(MAG "DENSE_SP_HP_BAND start\n" RESET);

    // Create the task pool for single precision half precision banded Cholesky factorization
    // This initializes all necessary data structures and GPU resources
    parsec_dpotrf = parsec_shpotrf_New( parsec, data, params ); 

    // Execute the factorization if task pool creation was successful
    if ( parsec_dpotrf != NULL )
    {
        // Add task pool to execution context for scheduling
        parsec_context_add_taskpool( parsec, (parsec_taskpool_t*)parsec_dpotrf);
        // Start execution and wait for completion
        // This is a synchronous call that blocks until all tasks complete
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        // Clean up resources after factorization is complete
        parsec_shpotrf_Destruct( parsec_dpotrf );
    }

    // Return 0 to indicate successful completion
    // Note: actual error information is stored in params->info
    return 0; 
}

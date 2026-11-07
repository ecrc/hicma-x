/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"

// Include headers for different GEMMEX operation types
// Currently only TN (Transpose-NoTranspose) operations are implemented
//#include "gemmex_NN.h"      // NoTranspose-NoTranspose (future implementation)
//#include "gemmex_NT.h"      // NoTranspose-Transpose (future implementation)
#include "gemmex_TN.h"      // Transpose-NoTranspose (current implementation)
//#include "gemmex_TT.h"      // Transpose-Transpose (future implementation)

// Include SUMMA (Scalable Universal Matrix Multiplication Algorithm) variants
//#include "gemmex_NN_summa.h"  // NN with SUMMA algorithm (future implementation)
//#include "gemmex_NT_summa.h"  // NT with SUMMA algorithm (future implementation)
#include "gemmex_TN_summa.h"  // TN with SUMMA algorithm (current implementation)
//#include "gemmex_TT_summa.h"  // TT with SUMMA algorithm (future implementation)

/**
 * @file gemmex_wrapper.c
 * @brief HICMA GEMMEX (Extended General Matrix Multiply) wrapper implementation for PaRSEC runtime
 * 
 * This file provides wrapper functions for extended GEMM operations in the HICMA library,
 * supporting both standard and SUMMA (Scalable Universal Matrix Multiplication Algorithm)
 * implementations with double precision arithmetic. It handles GPU device management,
 * memory allocation, and task scheduling for distributed matrix multiplication operations.
 * 
 * Supported operations:
 * - C = α * op(A) * op(B) + β * C (double precision)
 * - Where op(X) can be X, X^T, or X^H (conjugate transpose)
 * 
 * Currently supports TN (Transpose-NoTranspose) operations only.
 * NN, NT, and TT operations are planned for future implementation.
 */

/* Print more debug information flag - controls verbosity of debug output */
static int print_more = 0;

/**
 * @brief GPU evaluation hook for GEMMEX tasks
 * 
 * This function serves as a placeholder for GPU task evaluation in extended GEMM operations.
 * Currently returns immediately, indicating the task is done.
 * Future implementation may include actual GPU computation logic.
 * 
 * @param[in] task Pointer to the PaRSEC task to evaluate
 * @return PARSEC_HOOK_RETURN_DONE indicating task completion
 */
static parsec_hook_return_t evaluate_gpu_gemmex(parsec_task_t* task) {
        return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief Create a new GEMMEX taskpool using SUMMA algorithm
 * 
 * This function creates a taskpool for extended GEMM operations using the SUMMA
 * (Scalable Universal Matrix Multiplication Algorithm) approach. SUMMA provides
 * better scalability for distributed matrix multiplication by reducing communication
 * overhead through a more efficient data distribution pattern.
 * 
 * @param[in] transA Transpose operation for matrix A (dplasmaNoTrans, dplasmaTrans, dplasmaConjTrans)
 * @param[in] transB Transpose operation for matrix B (dplasmaNoTrans, dplasmaTrans, dplasmaConjTrans)
 * @param[in] alpha Scalar multiplier for the matrix product
 * @param[in] A Input matrix A descriptor
 * @param[in] B Input matrix B descriptor
 * @param[in] beta Scalar multiplier for matrix C
 * @param[in] C Output matrix C descriptor (input/output)
 * @param[in] opt Algorithm options (currently unused)
 * @param[in] params_tlr TLR (Tile Low-Rank) parameters for HICMA
 * @param[in] data Additional data including GPU workspace
 * @return Pointer to the created taskpool, or NULL on error
 */
parsec_taskpool_t *
hicma_parsec_gemmex_summa_new(dplasma_enum_t transA, dplasma_enum_t transB,
                        double alpha, parsec_tiled_matrix_t* A, parsec_tiled_matrix_t* B,
                        double beta,  parsec_tiled_matrix_t* C,
                        dplasma_info_t opt, hicma_parsec_params_t * params_tlr, hicma_parsec_data_t *data)
{
    // Local variables for SUMMA algorithm setup
    int P, Q, IP, JQ, m, n;                    // Grid dimensions and offsets
    parsec_taskpool_t *gemmex_tp;              // Taskpool to be created
    parsec_matrix_block_cyclic_t *Cdist;       // Distributed matrix descriptor for C
    void** eval_gpu_gemmex;                    // GPU evaluation function pointer

    // Extract grid parameters from matrix C's block-cyclic distribution
    P = ((parsec_matrix_block_cyclic_t*)C)->grid.rows;   // Number of process rows
    Q = ((parsec_matrix_block_cyclic_t*)C)->grid.cols;   // Number of process columns
    IP = ((parsec_matrix_block_cyclic_t*)C)->grid.ip;    // Process row index
    JQ = ((parsec_matrix_block_cyclic_t*)C)->grid.jq;    // Process column index

    // Calculate maximum dimensions for the distributed matrix
    m = dplasma_imax(C->mt, P);  // Maximum number of tile rows
    n = dplasma_imax(C->nt, Q);  // Maximum number of tile columns

    // Allocate and initialize distributed matrix descriptor for SUMMA algorithm
    Cdist = (parsec_matrix_block_cyclic_t*)malloc(sizeof(parsec_matrix_block_cyclic_t));

    // Initialize block-cyclic distribution for the result matrix C
    parsec_matrix_block_cyclic_init(
            Cdist, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,  // Data type and storage format
            C->super.myrank,                                  // Current process rank
            1, 1,                                             // Block dimensions (1x1 for tiles)
            m, n,                                             // Matrix dimensions
            0, 0,                                             // Starting indices
            m, n,                                             // Full matrix dimensions
            P, Q, 1, 1, IP, JQ);                             // Grid layout parameters
    Cdist->super.super.data_of = NULL;                        // No data ownership
    Cdist->super.super.data_of_key = NULL;                    // No data key

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    /** CUDA device detection and setup for GPU acceleration */
    int nb = 0;  // Number of CUDA devices found
    
    // First pass: count available CUDA devices
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( PARSEC_DEV_CUDA == device->type ) {
            nb++;
        }
    }

    // Handle case where no CUDA devices are found
    if(nb == 0) {
        char hostname[256];
        gethostname(hostname, 256);  // Get hostname for debugging
    }

    // Second pass: collect device indices for available CUDA devices
    int *dev_index = (int*)malloc(nb * sizeof(int));
    nb = 0;  // Reset counter for indexing
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( PARSEC_DEV_CUDA == device->type ) {
            dev_index[nb++] = device->device_index;
        }
    }
#endif

    // Route to appropriate GEMMEX implementation based on transpose operations
    if( dplasmaNoTrans == transA ) {
        // Matrix A is not transposed
        if( dplasmaNoTrans == transB ) {
            // NN case: A^T * B^T (not yet implemented)
            dplasma_error("parsec_gemmex_NN_summa_new",
                          "It will be supported in future");
        } else {
            // NT case: A^T * B (not yet implemented)
            dplasma_error("parsec_gemmex_NT_summa_new",
                          "It will be supported in future");
        }
    } else {
        // Matrix A is transposed
        if( dplasmaNoTrans == transB ) {
            // TN case: A^T * B (currently implemented)
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "gemmex_TN_summa\n");
            parsec_gemmex_TN_summa_taskpool_t* tp;
            tp = parsec_gemmex_TN_summa_new(transA, transB, alpha, beta,
                                           A, B, C, (parsec_data_collection_t*)Cdist, params_tlr);

            // Find the correct taskclass ID for GEMMEX operations
            int gemmex_id, gpu_id = 0;
            for( int i = 0; i < tp->super.nb_task_classes; i++ ) {
                if( !strcmp(tp->super.task_classes_array[i]->name, "gemmex") )
                    gemmex_id = tp->super.task_classes_array[i]->task_class_id;
            }

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    if( 0 == params_tlr->rank ) printf("gemmex_id= %d gpu_id= %d\n", gemmex_id, gpu_id);

        // GPU evaluate of chores 
        eval_gpu_gemmex  = (void *)&tp->super.task_classes_array[gemmex_id]->incarnations[gpu_id].evaluate;
        *eval_gpu_gemmex  = &evaluate_gpu_gemmex;


        tp->_g_ws_gpu = (void *)data->ws_gpu;
        tp->_g_nb_cuda_devices = nb;
        tp->_g_cuda_device_index = dev_index;
#endif

            parsec_add2arena( &((parsec_gemmex_TN_summa_taskpool_t*)tp)->arenas_datatypes[PARSEC_gemmex_TN_DEFAULT_ADT_IDX],
                            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
                             1, C->mb, C->nb, C->mb,
                            PARSEC_ARENA_ALIGNMENT_SSE, -1);
        #if HAVE_I8
            parsec_add2arena( &((parsec_gemmex_TN_summa_taskpool_t*)tp)->arenas_datatypes[PARSEC_gemmex_TN_FULL_I8_ADT_IDX],
                             MPI_BYTE, PARSEC_MATRIX_FULL,
                             1, A->mb, A->nb, A->mb,
                            PARSEC_ARENA_ALIGNMENT_SSE, -1);
        #endif

            gemmex_tp = (parsec_taskpool_t*)tp;

        } else {
            dplasma_error("parsec_gemmex_TT_summa_new",
                          "It will be supported in future");
        }
    }


    (void)opt; //No user-defined options for this algorithm
    return gemmex_tp;
}

/**
 * @brief Create a new GEMMEX taskpool using default algorithm
 * 
 * This function creates a taskpool for extended GEMM operations using the default
 * (non-SUMMA) algorithm. This provides a simpler implementation suitable for
 * smaller problem sizes or when SUMMA overhead is not justified.
 * 
 * @param[in] transA Transpose operation for matrix A (dplasmaNoTrans, dplasmaTrans, dplasmaConjTrans)
 * @param[in] transB Transpose operation for matrix B (dplasmaNoTrans, dplasmaTrans, dplasmaConjTrans)
 * @param[in] alpha Scalar multiplier for the matrix product
 * @param[in] A Input matrix A descriptor
 * @param[in] B Input matrix B descriptor
 * @param[in] beta Scalar multiplier for matrix C
 * @param[in] C Output matrix C descriptor (input/output)
 * @param[in] opt Algorithm options (currently unused)
 * @param[in] params_tlr TLR (Tile Low-Rank) parameters for HICMA
 * @param[in] data Additional data including GPU workspace
 * @return Pointer to the created taskpool, or NULL on error
 */
parsec_taskpool_t *
hicma_parsec_gemmex_default_new(dplasma_enum_t transA, dplasma_enum_t transB,
                          double alpha, parsec_tiled_matrix_t* A, parsec_tiled_matrix_t* B,
                          double beta,  parsec_tiled_matrix_t* C,
                          dplasma_info_t opt, hicma_parsec_params_t * params_tlr, hicma_parsec_data_t *data)
{
    parsec_taskpool_t* gemmex_tp = NULL;
    void** eval_gpu_gemmex;

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    // Find all CUDA devices 
    int nb = 0;
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( PARSEC_DEV_CUDA == device->type ) {
            nb++;
        }
    }

    if(nb == 0) {
        char hostname[256];
        gethostname(hostname, 256);
    }

    int *dev_index = (int*)malloc(nb * sizeof(int));
    nb = 0;
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( PARSEC_DEV_CUDA == device->type ) {
            dev_index[nb++] = device->device_index;
        }
    }
#endif

    if( dplasmaNoTrans == transA ) {
        if( dplasmaNoTrans == transB ) {
            dplasma_error("parsec_gemmex_NN_new",
                          "It will be supported in future");
        } else {
            dplasma_error("parsec_gemmex_NT_new",
                          "It will be supported in future");
        }
    } else {
        if( dplasmaNoTrans == transB ) {
            parsec_gemmex_TN_taskpool_t* tp;
            tp = parsec_gemmex_TN_new(transA, transB, (double)alpha, beta,
                                     A, B, C, params_tlr);
            gemmex_tp = (parsec_taskpool_t*)tp;

        // Find the correct taskclass ID 
    int gemmex_id, gpu_id = 0;
    for( int i = 0; i < tp->super.nb_task_classes; i++ ) {
	    if( !strcmp(tp->super.task_classes_array[i]->name, "gemmex") )
		    gemmex_id = tp->super.task_classes_array[i]->task_class_id;
    }

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    if( 0 == params_tlr->rank ) printf("gemmex_id= %d gpu_id= %d\n", gemmex_id, gpu_id);

        // GPU evaluate of chores 
        eval_gpu_gemmex  = (void *)&tp->super.task_classes_array[gemmex_id]->incarnations[gpu_id].evaluate;
        *eval_gpu_gemmex  = &evaluate_gpu_gemmex;


        tp->_g_ws_gpu = (void *)data->ws_gpu;
        tp->_g_nb_cuda_devices = nb;
        tp->_g_cuda_device_index = dev_index;
#endif

        parsec_add2arena( &((parsec_gemmex_TN_taskpool_t*)tp)->arenas_datatypes[PARSEC_gemmex_TN_DEFAULT_ADT_IDX],
                            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
                             1, C->mb, C->nb, C->mb,
                            PARSEC_ARENA_ALIGNMENT_SSE, -1);
    #if HAVE_I8
        parsec_add2arena( &((parsec_gemmex_TN_taskpool_t*)tp)->arenas_datatypes[PARSEC_gemmex_TN_FULL_I8_ADT_IDX],
                             MPI_BYTE, PARSEC_MATRIX_FULL,
                             1, A->mb, A->nb, A->mb,
                            PARSEC_ARENA_ALIGNMENT_SSE, -1);
    #endif

        }
        else {
            dplasma_error("hicma_parsec_gemmex_TT_new",
                          "It will be supported in future");
        }
    }



    (void)opt; //No user-defined options for this algorithm
    return gemmex_tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_gemmex_New - Generates the taskpool that performs one of the following
 *  matrix-matrix operations. WARNING: The computations are not done by this call.
 *
 *    \f[ C = \alpha [op( A )\times op( B )] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X' or op( X ) = conjg( X' )
 *
 *  alpha and beta are scalars, and A, B and C  are matrices, with op( A )
 *  an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in] transA
 *          Specifies whether the matrix A is transposed, not transposed or conjugate transposed:
 *          = dplasmaNoTrans:   A is not transposed;
 *          = dplasmaTrans:     A is transposed;
 *          = dplasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          Specifies whether the matrix B is transposed, not transposed or conjugate transposed:
 *          = dplasmaNoTrans:   B is not transposed;
 *          = dplasmaTrans:     B is transposed;
 *          = dplasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          Descriptor of the distributed matrix C.
 *          On exit, the data described by C are overwritten by the matrix (
 *          alpha*op( A )*op( B ) + beta*C )
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroy with dplasma_gemmex_Destruct();
 *
 ******************************************************************************/


/**
 * @brief Create a new GEMMEX taskpool with extended options
 * 
 * This is the main entry point for creating GEMMEX taskpools. It automatically
 * selects between SUMMA and default algorithms based on the GEMM_SUMMA compile-time
 * flag and matrix distribution type. The function validates input parameters and
 * routes to the appropriate implementation.
 * 
 * @param[in] transA Transpose operation for matrix A
 * @param[in] transB Transpose operation for matrix B  
 * @param[in] alpha Scalar multiplier for the matrix product
 * @param[in] A Input matrix A descriptor
 * @param[in] B Input matrix B descriptor
 * @param[in] beta Scalar multiplier for matrix C
 * @param[in] C Output matrix C descriptor (input/output)
 * @param[in] opt Algorithm options
 * @param[in] params_tlr TLR parameters for HICMA
 * @param[in] data Additional data including GPU workspace
 * @return Pointer to the created taskpool, or NULL on error
 */
parsec_taskpool_t*
hicma_parsec_gemmex_New_ex( dplasma_enum_t transA, dplasma_enum_t transB,
                      double alpha, parsec_tiled_matrix_t* A, parsec_tiled_matrix_t* B,
                      double beta,  parsec_tiled_matrix_t* C, dplasma_info_t opt, hicma_parsec_params_t * params_tlr,
                      hicma_parsec_data_t *data)
{
    parsec_taskpool_t* gemmex_tp = NULL;
    
    // Validate input arguments for transpose operations
    if ((transA != dplasmaNoTrans) && (transA != dplasmaTrans) && (transA != dplasmaConjTrans)) {
        dplasma_error("dplasma_gemmex_New", "illegal value of transA");
        return NULL;
    }
    if ((transB != dplasmaNoTrans) && (transB != dplasmaTrans) && (transB != dplasmaConjTrans)) {
        dplasma_error("dplasma_gemmex_New", "illegal value of transB");
        return NULL;
    }

    // Algorithm selection based on compile-time flags and matrix type
#if GEMM_SUMMA
    // Use SUMMA algorithm for block-cyclic distributed matrices
    if ( C->dtype & parsec_matrix_block_cyclic_type ) {
        gemmex_tp = hicma_parsec_gemmex_summa_new(transA, transB, alpha, A, B, beta, C, opt, params_tlr, data);
        return gemmex_tp;
    }
#else
    // Use default algorithm (fallback or when SUMMA is disabled)
    gemmex_tp = hicma_parsec_gemmex_default_new(transA, transB, alpha, A, B, beta, C, opt, params_tlr, data);
#endif
    return gemmex_tp;
}

parsec_taskpool_t*
hicma_parsec_gemmex_New( dplasma_enum_t transA, dplasma_enum_t transB,
                   double alpha, parsec_tiled_matrix_t* A, parsec_tiled_matrix_t* B,
                   double beta,  parsec_tiled_matrix_t* C, hicma_parsec_params_t * params_tlr,
               hicma_parsec_data_t *data)
{
    parsec_taskpool_t *tp;
    dplasma_info_t opt;
    dplasma_info_create(&opt);
    tp = hicma_parsec_gemmex_New_ex(transA, transB, alpha, A, B, beta, C, opt, params_tlr, data);
    dplasma_info_free(&opt);
    return tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_gemmex_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_gemmex_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_gemmex_New
 * @sa dplasma_gemmex
 *
 ******************************************************************************/

/**
 * @brief Destructor for GEMMEX taskpool
 * 
 * This function properly cleans up all resources associated with a GEMMEX taskpool,
 * including distributed matrix descriptors, GPU device information, and arena memory.
 * It handles both SUMMA and default algorithm variants.
 * 
 * @param[in,out] tp Taskpool to destroy. On exit, the taskpool cannot be used anymore.
 */
void
hicma_parsec_gemmex_Destruct( parsec_taskpool_t *tp )
{
    parsec_gemmex_TN_taskpool_t *gemmex_tp = (parsec_gemmex_TN_taskpool_t *)tp;

    // Clean up SUMMA algorithm resources
    if(
         gemmex_tp->_g_gemmex_type == HICMA_GEMM_NN_SUMMA ||
        gemmex_tp->_g_gemmex_type == HICMA_GEMM_NT_SUMMA ||
        gemmex_tp->_g_gemmex_type == HICMA_GEMM_TN_SUMMA ||
        gemmex_tp->_g_gemmex_type == HICMA_GEMM_TT_SUMMA) {
        parsec_gemmex_TN_summa_taskpool_t *gemmex_summa_tp = (parsec_gemmex_TN_summa_taskpool_t *)tp;
        parsec_tiled_matrix_t* Cdist = (parsec_tiled_matrix_t*)gemmex_summa_tp->_g_Cdist;
        if ( NULL != Cdist ) {
            parsec_tiled_matrix_destroy( Cdist );
            free( Cdist );
        }

        #if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        if( gemmex_summa_tp->_g_nb_cuda_devices > 0 ) {

        if( NULL != gemmex_summa_tp->_g_cuda_device_index )
            free(gemmex_summa_tp->_g_cuda_device_index);
        }
         #endif 

        parsec_del2arena( &gemmex_summa_tp->arenas_datatypes[PARSEC_gemmex_TN_DEFAULT_ADT_IDX] );
        
#if HAVE_I8
        parsec_del2arena( &gemmex_summa_tp->arenas_datatypes[PARSEC_gemmex_TN_FULL_I8_ADT_IDX] );
#endif

    } else if( gemmex_tp->_g_gemmex_type == HICMA_GEMM_NN ||
               gemmex_tp->_g_gemmex_type == HICMA_GEMM_NT ||
               gemmex_tp->_g_gemmex_type == HICMA_GEMM_TN ||
               gemmex_tp->_g_gemmex_type == HICMA_GEMM_TT) {

        #if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        if( gemmex_tp->_g_nb_cuda_devices > 0 ) {

        if( NULL != gemmex_tp->_g_cuda_device_index )
            free(gemmex_tp->_g_cuda_device_index);
        }
         #endif 

        parsec_del2arena( &gemmex_tp->arenas_datatypes[PARSEC_gemmex_TN_DEFAULT_ADT_IDX] );
        
#if HAVE_I8
        parsec_del2arena( &gemmex_tp->arenas_datatypes[PARSEC_gemmex_TN_FULL_I8_ADT_IDX] );
#endif

    }

    parsec_taskpool_free(tp);

}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_gemmex - Performs one of the following matrix-matrix operations
 *
 *    \f[ C = \alpha [op( A )\times op( B )] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X' or op( X ) = conjg( X' )
 *
 *  alpha and beta are scalars, and A, B and C  are matrices, with op( A )
 *  an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] transA
 *          Specifies whether the matrix A is transposed, not transposed or conjugate transposed:
 *          = dplasmaNoTrans:   A is not transposed;
 *          = dplasmaTrans:     A is transposed;
 *          = dplasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          Specifies whether the matrix B is transposed, not transposed or conjugate transposed:
 *          = dplasmaNoTrans:   B is not transposed;
 *          = dplasmaTrans:     B is transposed;
 *          = dplasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          Descriptor of the distributed matrix C.
 *          On exit, the data described by C are overwritten by the matrix (
 *          alpha*op( A )*op( B ) + beta*C )
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_gemmex_New
 * @sa dplasma_gemmex_Destruct
 * @sa dplasma_cgemmex
 * @sa dplasma_dgemmex
 * @sa dplasma_sgemmex
 *
 ******************************************************************************/
int
hicma_parsec_gemmex( parsec_context_t *parsec,
               dplasma_enum_t transA, dplasma_enum_t transB,
               double alpha, parsec_tiled_matrix_t *A,
                             parsec_tiled_matrix_t *B,
               double beta,  parsec_tiled_matrix_t *C, hicma_parsec_params_t * params_tlr,
               hicma_parsec_data_t *data)
{
    parsec_taskpool_t *parsec_gemmex = NULL;
    int M, N, K;
    int Am, An, Ai, Aj, Amb, Anb;
    int Bm, Bn, Bi, Bj, Bmb, Bnb;

    // Check input arguments 
    if ((transA != dplasmaNoTrans) && (transA != dplasmaTrans) && (transA != dplasmaConjTrans)) {
        dplasma_error("dplasma_gemmex", "illegal value of transA");
        return -1;
    }
    if ((transB != dplasmaNoTrans) && (transB != dplasmaTrans) && (transB != dplasmaConjTrans)) {
        dplasma_error("dplasma_gemmex", "illegal value of transB");
        return -2;
    }

    if ( transA == dplasmaNoTrans ) {
        Am  = A->m;
        An  = A->n;
        Amb = A->mb;
        Anb = A->nb;
        Ai  = A->i;
        Aj  = A->j;
    } else {
        Am  = A->n;
        An  = A->m;
        Amb = A->nb;
        Anb = A->mb;
        Ai  = A->j;
        Aj  = A->i;
    }

    if ( transB == dplasmaNoTrans ) {
        Bm  = B->m;
        Bn  = B->n;
        Bmb = B->mb;
        Bnb = B->nb;
        Bi  = B->i;
        Bj  = B->j;
    } else {
        Bm  = B->n;
        Bn  = B->m;
        Bmb = B->nb;
        Bnb = B->mb;
        Bi  = B->j;
        Bj  = B->i;
    }

    if ( (Amb != C->mb) || (Anb != Bmb) || (Bnb != C->nb) ) {
        dplasma_error("dplasma_gemmex", "tile sizes have to match");
        return -101;
    }
    if ( (Am != C->m) || (An != Bm) || (Bn != C->n) ) {
        dplasma_error("dplasma_gemmex", "sizes of matrices have to match");
        return -101;
    }
    if ( (Ai != C->i) || (Aj != Bi) || (Bj != C->j) ) {
        dplasma_error("dplasma_gemmex", "start indexes have to match");
        return -101;
    }

    M = C->m;
    N = C->n;
    K = An;

    parsec_gemmex = hicma_parsec_gemmex_New(transA, transB,
                                    alpha, A, B,
                                    beta, C, params_tlr, data);

    if ( parsec_gemmex != NULL )
    {
        parsec_context_add_taskpool( parsec, (parsec_taskpool_t*)parsec_gemmex);
        dplasma_wait_until_completion(parsec);
        hicma_parsec_gemmex_Destruct( parsec_gemmex );
        return 0;
    }
    return -101;
}

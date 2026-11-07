/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

/**
 * @file hamming_binary_wrapper.c
 * @brief C Wrapper Functions for Hamming Distance Computation
 * 
 * This file provides high-level C wrapper functions for the Hamming distance
 * computation using binary matrices. It interfaces with the JDF (Just Data Flow)
 * implementations and provides a clean API for users to perform Hamming distance
 * calculations on distributed binary matrices.
 * 
 * Key Features:
 * - High-level interface for Hamming distance computation
 * - Support for both CUDA and HIP GPU backends
 * - Memory management and workspace allocation
 * - Integration with PaRSEC runtime for distributed execution
 * - Comprehensive error checking and validation
 */

#include "hicma_parsec.h"

//#include "gemm_NN.h"
//#include "gemm_NT.h"
#include "hamming_binary.h"
//#include "gemm_TT.h"

/** @brief Debug flag for additional output (currently unused) */
static int print_more = 0;

/**
 * @brief Hook function for GPU task evaluation
 * 
 * This function is used as a hook for GPU task evaluation in the PaRSEC runtime.
 * It always returns DONE, indicating that the task should be executed on GPU.
 * 
 * @param task The task being evaluated
 * @return PARSEC_HOOK_RETURN_DONE
 */
static parsec_hook_return_t evaluate_gpu_gemm(parsec_task_t* task) {
        return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief Create a new Hamming binary computation taskpool with default settings
 * 
 * This function creates and configures a new PaRSEC taskpool for Hamming distance
 * computation between binary matrices. It handles GPU device detection, memory
 * allocation, and taskpool configuration for optimal performance.
 * 
 * @param transA Transpose flag for matrix A (dplasmaTrans or dplasmaNoTrans)
 * @param transB Transpose flag for matrix B (dplasmaTrans or dplasmaNoTrans)
 * @param alpha Scaling factor for the matrix multiplication
 * @param A Input matrix A (binary data)
 * @param B Input matrix B (binary data, typically same as A for Hamming distance)
 * @param beta Scaling factor for accumulation
 * @param C Output matrix C (Hamming distance results)
 * @param opt DPLASMA options (currently unused)
 * @param params_tlr HICMA PaRSEC parameters including algorithm settings
 * @param data HICMA PaRSEC data including GPU workspace
 * @return The configured parsec taskpool object, or NULL on failure
 */
    parsec_taskpool_t *
hicma_parsec_hamming_binary_default_new(dplasma_enum_t transA, dplasma_enum_t transB,
        float alpha, parsec_tiled_matrix_t* A, parsec_tiled_matrix_t* B,
        float beta,  parsec_tiled_matrix_t* C,
        dplasma_info_t opt, hicma_parsec_params_t * params_tlr, hicma_parsec_data_t *data)
{
    parsec_taskpool_t* gemm_tp = NULL;
    void** eval_gpu_gemm;

    //printf("%d %s \n", __LINE__, __FILE__);
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    // Detect and enumerate available CUDA/HIP devices
    int nb = 0;
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( PARSEC_DEV_CUDA == device->type || PARSEC_DEV_HIP == device->type) {
            nb++;
        }
    }

    // Handle case where no GPU devices are found
    if(nb == 0) {
        char hostname[256];
        gethostname(hostname, 256);
    }

    // Allocate array to store device indices
    int *dev_index = (int*)malloc(nb * sizeof(int));
    nb = 0;
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( PARSEC_DEV_CUDA == device->type || PARSEC_DEV_HIP == device->type) {
            dev_index[nb++] = device->device_index;
        }
    }
#endif

    // Handle different transpose combinations (only transA=Trans, transB=NoTrans supported)
    if( dplasmaNoTrans == transA ) {
        if( dplasmaNoTrans == transB ) {
            dplasma_error("parsec_gemm_NN_new",
                    "It will be supported in future");
        } else {
            dplasma_error("parsec_gemm_NT_new",
                    "It will be supported in future");
        }
    } else {
        if( dplasmaNoTrans == transB ) {
            // Create the Hamming binary taskpool (transA=Trans, transB=NoTrans)
            parsec_hamming_binary_taskpool_t* tp;
            tp = parsec_hamming_binary_new(transA, transB, (int)alpha, (int)beta,
                    A, B, C, params_tlr);

            // Allocate and initialize memory pool for workspace A (int8_t)
            tp->_g_p_work_intA = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
            parsec_private_memory_init( tp->_g_p_work_intA, A->mb * A->mb * sizeof(int8_t) );

            // Allocate and initialize memory pool for workspace B (int8_t)
            tp->_g_p_work_intB = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
            parsec_private_memory_init( tp->_g_p_work_intB, B->mb * B->mb * sizeof(int8_t) );

            gemm_tp = (parsec_taskpool_t*)tp;

            // Find the GEMM task class ID for GPU configuration
            int gemm_id, gpu_id = 0;
            for( int i = 0; i < tp->super.nb_task_classes; i++ ) {
                if( !strcmp(tp->super.task_classes_array[i]->name, "GEMM") )
                    gemm_id = tp->super.task_classes_array[i]->task_class_id;
            }

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
            if( DEBUG_INFO && 0 == params_tlr->rank && params_tlr->verbose ) fprintf(stderr, "gemm_id= %d gpu_id= %d\n", gemm_id, gpu_id);

            // Configure GPU task evaluation hook
            eval_gpu_gemm  = (void *)&tp->super.task_classes_array[gemm_id]->incarnations[gpu_id].evaluate;
            *eval_gpu_gemm  = &evaluate_gpu_gemm;

            // Configure GPU workspace and device information
            tp->_g_ws_gpu = (void *)data->ws_gpu;
            tp->_g_nb_cuda_devices = nb;
            tp->_g_cuda_device_index = dev_index;
            
            // Configure lookahead parameters for performance optimization
            if(1 == params_tlr->nodes) {
                tp->_g_lookP = A->mt; 
                tp->_g_lookQ = A->mt; 
            } else if(params_tlr->lookahead < 1) {
                tp->_g_lookP = params_tlr->P;
                tp->_g_lookQ = params_tlr->P;
            } else {
                tp->_g_lookP = params_tlr->lookahead;
                tp->_g_lookQ = params_tlr->lookahead;
            }
            params_tlr->lookahead = tp->_g_lookP;
            if( 0 == params_tlr->rank && params_tlr->verbose ) fprintf(stderr, RED"Set lookP= %d lookQ= %d lookahead= %d\n"RESET,
                    tp->_g_lookP, tp->_g_lookQ, params_tlr->lookahead);
#endif

            // Configure memory arenas for different data types
            // Arena for float data type (unused but required for compatibility)
            parsec_add2arena( &((parsec_hamming_binary_taskpool_t*)tp)->arenas_datatypes[PARSEC_hamming_binary_FULL_SP_ADT_IDX],
                    parsec_datatype_float_t, PARSEC_MATRIX_FULL,
                    1, C->mb, C->nb, C->mb,
                    PARSEC_ARENA_ALIGNMENT_SSE, -1);
            
            // Arena for int8_t data type (input binary matrices A and B)
            parsec_add2arena( &((parsec_hamming_binary_taskpool_t*)tp)->arenas_datatypes[PARSEC_hamming_binary_FULL_I8_ADT_IDX],
                    parsec_datatype_int8_t, PARSEC_MATRIX_FULL,
                    1, A->mb, A->nb, A->mb,
                    PARSEC_ARENA_ALIGNMENT_SSE, -1);

            // Arena for int32_t data type (output Hamming distance matrix C)
            parsec_add2arena( &((parsec_hamming_binary_taskpool_t*)tp)->arenas_datatypes[PARSEC_hamming_binary_FULL_I32_ADT_IDX],
                    parsec_datatype_int32_t, PARSEC_MATRIX_FULL,
                    1, A->mb, A->nb, A->mb,
                    PARSEC_ARENA_ALIGNMENT_SSE, -1);

        }
        else {
            dplasma_error("hicma_parsec_hamming_binary_TT_new",
                    "It will be supported in future");
        }
    }

    (void)opt; //No user-defined options for this algorithm
    return gemm_tp;
}

/**
 *******************************************************************************
 *
 * @brief Create a new Hamming binary computation taskpool with extended options
 *
 * This function creates and configures a new PaRSEC taskpool for Hamming distance
 * computation between binary matrices. It provides the same functionality as
 * hicma_parsec_hamming_binary_New but with additional DPLASMA options support.
 *
 * The computation performs: C = alpha * A^T * B + beta * C
 * where A and B are binary matrices and C contains the Hamming distances.
 *
 *******************************************************************************
 *
 * @param[in] transA
 *          Specifies whether the matrix A is transposed:
 *          = dplasmaNoTrans:   A is not transposed;
 *          = dplasmaTrans:     A is transposed;
 *          = dplasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          Specifies whether the matrix B is transposed:
 *          = dplasmaNoTrans:   B is not transposed;
 *          = dplasmaTrans:     B is transposed;
 *          = dplasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] alpha
 *          Scaling factor for the matrix multiplication
 *
 * @param[in] A
 *          Descriptor of the distributed binary matrix A
 *
 * @param[in] B
 *          Descriptor of the distributed binary matrix B (typically same as A)
 *
 * @param[in] beta
 *          Scaling factor for accumulation
 *
 * @param[in,out] C
 *          Descriptor of the distributed matrix C for Hamming distance results
 *
 * @param[in] opt
 *          DPLASMA options (currently unused)
 *
 * @param[in] params_tlr
 *          HICMA PaRSEC parameters including algorithm settings
 *
 * @param[in] data
 *          HICMA PaRSEC data including GPU workspace
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroyed with hicma_parsec_hamming_binary_Destruct();
 *
 ******************************************************************************/


parsec_taskpool_t*
hicma_parsec_hamming_binary_New_ex( dplasma_enum_t transA, dplasma_enum_t transB,
                      float alpha, parsec_tiled_matrix_t* A, parsec_tiled_matrix_t* B,
                      float beta,  parsec_tiled_matrix_t* C, dplasma_info_t opt, hicma_parsec_params_t * params_tlr,
               hicma_parsec_data_t *data)
{
    parsec_taskpool_t* gemm_tp = NULL;
    
    // Validate input arguments
    if ((transA != dplasmaNoTrans) && (transA != dplasmaTrans) && (transA != dplasmaConjTrans)) {
        dplasma_error("dplasma_gemm_New", "illegal value of transA");
        return NULL;
    }
    if ((transB != dplasmaNoTrans) && (transB != dplasmaTrans) && (transB != dplasmaConjTrans)) {
        dplasma_error("dplasma_gemm_New", "illegal value of transB");
        return NULL;
    }
    
    // Create the taskpool using the default configuration
    gemm_tp = hicma_parsec_hamming_binary_default_new(transA, transB, alpha, A, B, beta, C, opt, params_tlr, data);

    return gemm_tp;
}

/**
 * @brief Create a new Hamming binary computation taskpool (simplified interface)
 * 
 * This function provides a simplified interface for creating Hamming binary
 * computation taskpools without requiring DPLASMA options.
 * 
 * @param transA Transpose flag for matrix A
 * @param transB Transpose flag for matrix B  
 * @param alpha Scaling factor for the matrix multiplication
 * @param A Input matrix A (binary data)
 * @param B Input matrix B (binary data)
 * @param beta Scaling factor for accumulation
 * @param C Output matrix C (Hamming distance results)
 * @param params_tlr HICMA PaRSEC parameters
 * @param data HICMA PaRSEC data including GPU workspace
 * @return The configured parsec taskpool object, or NULL on failure
 */
parsec_taskpool_t*
hicma_parsec_hamming_binary_New( dplasma_enum_t transA, dplasma_enum_t transB,
                   float alpha, parsec_tiled_matrix_t* A, parsec_tiled_matrix_t* B,
                   float beta,  parsec_tiled_matrix_t* C, hicma_parsec_params_t * params_tlr,
               hicma_parsec_data_t *data)
{
    parsec_taskpool_t *tp;
    dplasma_info_t opt;
    dplasma_info_create(&opt);
    tp = hicma_parsec_hamming_binary_New_ex(transA, transB, alpha, A, B, beta, C, opt, params_tlr, data);
    dplasma_info_free(&opt);
    return tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_gemm_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_gemm_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_gemm_New
 * @sa dplasma_gemm
 *
 ******************************************************************************/

/**
 * @brief Destroy a Hamming binary computation taskpool
 * 
 * This function properly cleans up and destroys a Hamming binary computation
 * taskpool, including freeing all associated memory arenas, GPU resources,
 * and workspace allocations.
 * 
 * @param tp The parsec taskpool object to destroy
 */
void hicma_parsec_hamming_binary_Destruct( parsec_taskpool_t *tp )
{
    parsec_hamming_binary_taskpool_t *gemm_tp = (parsec_hamming_binary_taskpool_t *)tp;

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        // Clean up GPU device information
        if( gemm_tp->_g_nb_cuda_devices > 0 ) {
            if( NULL != gemm_tp->_g_cuda_device_index )
                free(gemm_tp->_g_cuda_device_index);
        }
#endif 

        // Clean up memory arenas
        parsec_del2arena( &gemm_tp->arenas_datatypes[PARSEC_hamming_binary_FULL_SP_ADT_IDX] );
        parsec_del2arena( &gemm_tp->arenas_datatypes[PARSEC_hamming_binary_FULL_I8_ADT_IDX] );
        parsec_del2arena( &gemm_tp->arenas_datatypes[PARSEC_hamming_binary_FULL_I32_ADT_IDX] );

        // Clean up private memory pools
        parsec_private_memory_fini( gemm_tp->_g_p_work_intA );
        parsec_private_memory_fini( gemm_tp->_g_p_work_intB );
        free( gemm_tp->_g_p_work_intA );
        free( gemm_tp->_g_p_work_intB );

        // Free the taskpool
        parsec_taskpool_free(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_gemm - Performs one of the following matrix-matrix operations
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
 * @sa dplasma_gemm_New
 * @sa dplasma_gemm_Destruct
 * @sa dplasma_cgemm
 * @sa dplasma_dgemm
 * @sa dplasma_sgemm
 *
 ******************************************************************************/
/**
 * @brief High-level function to perform Hamming distance computation
 * 
 * This function provides a high-level interface for performing Hamming distance
 * computation between binary matrices. It creates the taskpool, executes the
 * computation tasks, and properly cleans up resources.
 * 
 * @param parsec The PaRSEC context for distributed execution
 * @param transA Transpose flag for matrix A
 * @param transB Transpose flag for matrix B
 * @param alpha Scaling factor for the matrix multiplication
 * @param A Input matrix A (binary data)
 * @param B Input matrix B (binary data)
 * @param beta Scaling factor for accumulation
 * @param C Output matrix C (Hamming distance results)
 * @param params_tlr HICMA PaRSEC parameters
 * @param data HICMA PaRSEC data including GPU workspace
 * @return 0 on success, negative value on failure
 */
int
hicma_parsec_hamming_binary( parsec_context_t *parsec,
               dplasma_enum_t transA, dplasma_enum_t transB,
               float alpha, parsec_tiled_matrix_t *A,
                            parsec_tiled_matrix_t *B,
               float beta,  parsec_tiled_matrix_t *C, hicma_parsec_params_t * params_tlr,
               hicma_parsec_data_t *data)
{
    parsec_taskpool_t *parsec_gemm = NULL;
    int M, N, K;
    int Am, An, Ai, Aj, Amb, Anb;
    int Bm, Bn, Bi, Bj, Bmb, Bnb;

    // Validate input arguments
    if ((transA != dplasmaNoTrans) && (transA != dplasmaTrans) && (transA != dplasmaConjTrans)) {
        dplasma_error("dplasma_gemm", "illegal value of transA");
        return -1;
    }
    if ((transB != dplasmaNoTrans) && (transB != dplasmaTrans) && (transB != dplasmaConjTrans)) {
        dplasma_error("dplasma_gemm", "illegal value of transB");
        return -2;
    }

    // Calculate effective dimensions based on transpose flags
    if ( transA == dplasmaNoTrans ) {
        Am  = A->m;   // Number of rows
        An  = A->n;   // Number of columns
        Amb = A->mb;  // Row block size
        Anb = A->nb;  // Column block size
        Ai  = A->i;   // Starting row index
        Aj  = A->j;   // Starting column index
    } else {
        Am  = A->n;   // Transposed: rows become columns
        An  = A->m;   // Transposed: columns become rows
        Amb = A->nb;  // Transposed: row block size becomes column block size
        Anb = A->mb;  // Transposed: column block size becomes row block size
        Ai  = A->j;   // Transposed: starting row index becomes starting column index
        Aj  = A->i;   // Transposed: starting column index becomes starting row index
    }

    if ( transB == dplasmaNoTrans ) {
        Bm  = B->m;   // Number of rows
        Bn  = B->n;   // Number of columns
        Bmb = B->mb;  // Row block size
        Bnb = B->nb;  // Column block size
        Bi  = B->i;   // Starting row index
        Bj  = B->j;   // Starting column index
    } else {
        Bm  = B->n;   // Transposed: rows become columns
        Bn  = B->m;   // Transposed: columns become rows
        Bmb = B->nb;  // Transposed: row block size becomes column block size
        Bnb = B->mb;  // Transposed: column block size becomes row block size
        Bi  = B->j;   // Transposed: starting row index becomes starting column index
        Bj  = B->i;   // Transposed: starting column index becomes starting row index
    }

    // Validate tile size compatibility
    if ( (Amb != C->mb) || (Anb != Bmb) || (Bnb != C->nb) ) {
        dplasma_error("dplasma_gemm", "tile sizes have to match");
        return -101;
    }
    
    // Validate matrix dimension compatibility
    if ( (Am != C->m) || (An != Bm) || (Bn != C->n) ) {
        dplasma_error("dplasma_gemm", "sizes of matrices have to match");
        return -101;
    }
    
    // Validate starting index compatibility
    if ( (Ai != C->i) || (Aj != Bi) || (Bj != C->j) ) {
        dplasma_error("dplasma_gemm", "start indexes have to match");
        return -101;
    }

    // Set up matrix dimensions for the computation
    M = C->m;  // Number of rows in result matrix C
    N = C->n;  // Number of columns in result matrix C
    K = An;    // Inner dimension (columns of A, rows of B)

    // Create the Hamming binary computation taskpool
    parsec_gemm = hicma_parsec_hamming_binary_New(transA, transB,
                                    alpha, A, B,
                                    beta, C, params_tlr, data);

    if ( parsec_gemm != NULL )
    {
        // Execute the computation tasks
        parsec_context_add_taskpool( parsec, (parsec_taskpool_t*)parsec_gemm);
        dplasma_wait_until_completion(parsec);
        
        // Clean up resources
        hicma_parsec_hamming_binary_Destruct( parsec_gemm );
        return 0;
    }
    return -101;
}

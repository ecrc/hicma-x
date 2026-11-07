/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"

//#include "gemm_NN.h"
//#include "gemm_NT.h"
#include "gemm_TN.h"
//#include "gemm_TT.h"

//#include "gemm_NN_summa.h"
//#include "gemm_NT_summa.h"
//#include "gemm_TT_summa.h"

/**
 * @file gemm_wrapper.c
 * @brief HICMA GEMM (General Matrix Multiply) wrapper implementation for PaRSEC runtime
 * 
 * This file provides wrapper functions for GEMM operations in the HICMA library,
 * supporting both standard and SUMMA (Scalable Universal Matrix Multiplication Algorithm)
 * implementations. It handles GPU device management, memory allocation, and task
 * scheduling for distributed matrix multiplication operations.
 * 
 * Supported operations:
 * - C = α * op(A) * op(B) + β * C
 * - Where op(X) can be X, X^T, or X^H (conjugate transpose)
 * 
 * Currently supports TN (Transpose-NoTranspose) operations only.
 * NN, NT, and TT operations are planned for future implementation.
 */

/* Print more debug information flag */
static int print_more = 0;

/**
 * @brief GPU evaluation hook for GEMM tasks
 * 
 * This function serves as a placeholder for GPU task evaluation.
 * Currently returns immediately, indicating the task is done.
 * Future implementation may include actual GPU computation logic.
 * 
 * @param[in] task Pointer to the PaRSEC task to evaluate
 * @return PARSEC_HOOK_RETURN_DONE indicating task completion
 */
static parsec_hook_return_t evaluate_gpu_gemm(parsec_task_t* task) {
        return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief Create a new default GEMM taskpool for distributed matrix multiplication
 * 
 * This function creates a taskpool for GEMM operations using the default algorithm,
 * which provides standard distributed matrix multiplication capabilities. It handles
 * GPU device discovery, memory allocation, and task scheduling setup.
 * 
 * @param[in] transA Transpose operation for matrix A (dplasmaNoTrans, dplasmaTrans, dplasmaConjTrans)
 * @param[in] transB Transpose operation for matrix B (dplasmaNoTrans, dplasmaTrans, dplasmaConjTrans)
 * @param[in] alpha Scalar multiplier for the matrix product
 * @param[in] A Input matrix A descriptor
 * @param[in] B Input matrix B descriptor
 * @param[in] beta Scalar multiplier for matrix C
 * @param[in] C Output matrix C descriptor (input/output)
 * @param[in] opt Algorithm options (currently unused)
 * @param[in] params_tlr HICMA PaRSEC parameters including rank, nodes, etc.
 * @param[in] data HICMA PaRSEC data structures including GPU workspace
 * @return Pointer to the created taskpool, or NULL on error
 */
parsec_taskpool_t *
hicma_parsec_syrk_default_new(dplasma_enum_t transA, dplasma_enum_t transB,
        float alpha, parsec_tiled_matrix_t* A, parsec_tiled_matrix_t* B,
        float beta,  parsec_tiled_matrix_t* C,
        dplasma_info_t opt, hicma_parsec_params_t * params_tlr, hicma_parsec_data_t *data)
{
    parsec_taskpool_t* gemm_tp = NULL;
    void** eval_gpu_gemm;

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Discover and enumerate available CUDA/HIP GPU devices */
    int nb = 0;
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( PARSEC_DEV_CUDA == device->type ) {
            nb++;
        }
    }

    /* Handle case where no GPU devices are found */
    if(nb == 0) {
        char hostname[256];
        gethostname(hostname, 256);
    }

    /* Allocate array to store device indices */
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
            dplasma_error("parsec_gemm_NN_new",
                    "It will be supported in future");
        } else {
            dplasma_error("parsec_gemm_NT_new",
                    "It will be supported in future");
        }
    } else {
        if( dplasmaNoTrans == transB ) {
            parsec_gemm_TN_taskpool_t* tp;
            tp = parsec_gemm_TN_new(transA, transB, alpha, beta,
                    A, B, C, params_tlr);

            tp->_g_p_work_int = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
            parsec_private_memory_init( tp->_g_p_work_int, C->mb * C->mb * sizeof(int) );

            gemm_tp = (parsec_taskpool_t*)tp;

            // Find the correct taskclass ID 
            int gemm_id, gpu_id = 0;
            for( int i = 0; i < tp->super.nb_task_classes; i++ ) {
                if( !strcmp(tp->super.task_classes_array[i]->name, "GEMM") )
                    gemm_id = tp->super.task_classes_array[i]->task_class_id;
            }

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
            if( DEBUG_INFO && 0 == params_tlr->rank && params_tlr->verbose ) fprintf(stderr, "gemm_id= %d gpu_id= %d\n", gemm_id, gpu_id);

            // GPU evaluate of chores 
            eval_gpu_gemm  = (void *)&tp->super.task_classes_array[gemm_id]->incarnations[gpu_id].evaluate;
            *eval_gpu_gemm  = &evaluate_gpu_gemm;

            tp->_g_ws_gpu = (void *)data->ws_gpu;
            tp->_g_nb_cuda_devices = nb;
            tp->_g_cuda_device_index = dev_index;
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
            if( 0 == params_tlr->rank && params_tlr->verbose) fprintf(stderr, RED"Set lookP= %d lookQ= %d lookahead= %d\n"RESET,
                    tp->_g_lookP, tp->_g_lookQ, params_tlr->lookahead);
#endif

            parsec_add2arena( &((parsec_gemm_TN_taskpool_t*)tp)->arenas_datatypes[PARSEC_gemm_TN_FULL_SP_ADT_IDX],
                    parsec_datatype_float_t, PARSEC_MATRIX_FULL,
                    1, C->mb, C->nb, C->mb,
                    PARSEC_ARENA_ALIGNMENT_SSE, -1);
            parsec_add2arena( &((parsec_gemm_TN_taskpool_t*)tp)->arenas_datatypes[PARSEC_gemm_TN_FULL_I8_ADT_IDX],
                    parsec_datatype_int8_t, PARSEC_MATRIX_FULL,
                    1, A->mb, A->nb, A->mb,
                    PARSEC_ARENA_ALIGNMENT_SSE, -1);

            parsec_add2arena( &((parsec_gemm_TN_taskpool_t*)tp)->arenas_datatypes[PARSEC_gemm_TN_FULL_I32_ADT_IDX],
                    parsec_datatype_int32_t, PARSEC_MATRIX_FULL,
                    1, A->mb, A->nb, A->mb,
                    PARSEC_ARENA_ALIGNMENT_SSE, -1);

        }
        else {
            dplasma_error("hicma_parsec_syrk_TT_new",
                    "It will be supported in future");
        }
    }



    (void)opt; //No user-defined options for this algorithm
    return gemm_tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_gemm_New - Generates the taskpool that performs one of the following
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
 *          destroy with dplasma_gemm_Destruct();
 *
 ******************************************************************************/


/**
 * @brief Create a new GEMM taskpool with extended options
 * 
 * This is the main interface function for creating GEMM taskpools. It validates
 * input parameters and routes to the appropriate implementation (SUMMA or default).
 * 
 * @param[in] transA Transpose operation for matrix A
 * @param[in] transB Transpose operation for matrix B  
 * @param[in] alpha Scalar multiplier for the matrix product
 * @param[in] A Input matrix A descriptor
 * @param[in] B Input matrix B descriptor
 * @param[in] beta Scalar multiplier for matrix C
 * @param[in] C Output matrix C descriptor (input/output)
 * @param[in] opt Algorithm options
 * @param[in] params_tlr HICMA PaRSEC parameters
 * @param[in] data HICMA PaRSEC data structures
 * @return Pointer to the created taskpool, or NULL on error
 */
parsec_taskpool_t*
hicma_parsec_syrk_New_ex( dplasma_enum_t transA, dplasma_enum_t transB,
                      float alpha, parsec_tiled_matrix_t* A, parsec_tiled_matrix_t* B,
                      float beta,  parsec_tiled_matrix_t* C, dplasma_info_t opt, hicma_parsec_params_t * params_tlr,
               hicma_parsec_data_t *data)
{
    parsec_taskpool_t* gemm_tp = NULL;
    
    /* Validate input arguments */
    if ((transA != dplasmaNoTrans) && (transA != dplasmaTrans) && (transA != dplasmaConjTrans)) {
        dplasma_error("dplasma_gemm_New", "illegal value of transA");
        return NULL;
    }
    if ((transB != dplasmaNoTrans) && (transB != dplasmaTrans) && (transB != dplasmaConjTrans)) {
        dplasma_error("dplasma_gemm_New", "illegal value of transB");
        return NULL;
    }

    /* Currently using default implementation - SUMMA commented out for future use */
    //if ( C->dtype & parsec_matrix_block_cyclic_type ) {
    //    printf("\n %s %d\n", __FILE__, __LINE__);
    //    gemm_tp = hicma_parsec_syrk_summa_new(transA, transB, alpha, A, B, beta, C, opt, params_tlr, data);
    //    return gemm_tp;
    //}

    /* Create taskpool using default GEMM implementation */
    gemm_tp = hicma_parsec_syrk_default_new(transA, transB, alpha, A, B, beta, C, opt, params_tlr, data);

    return gemm_tp;
}

/**
 * @brief Create a new GEMM taskpool (simplified interface)
 * 
 * Simplified interface that creates default options and calls the extended version.
 * 
 * @param[in] transA Transpose operation for matrix A
 * @param[in] transB Transpose operation for matrix B
 * @param[in] alpha Scalar multiplier for the matrix product
 * @param[in] A Input matrix A descriptor
 * @param[in] B Input matrix B descriptor
 * @param[in] beta Scalar multiplier for matrix C
 * @param[in] C Output matrix C descriptor (input/output)
 * @param[in] params_tlr HICMA PaRSEC parameters
 * @param[in] data HICMA PaRSEC data structures
 * @return Pointer to the created taskpool, or NULL on error
 */
parsec_taskpool_t*
hicma_parsec_syrk_New( dplasma_enum_t transA, dplasma_enum_t transB,
                   float alpha, parsec_tiled_matrix_t* A, parsec_tiled_matrix_t* B,
                   float beta,  parsec_tiled_matrix_t* C, hicma_parsec_params_t * params_tlr,
               hicma_parsec_data_t *data)
{
    parsec_taskpool_t *tp;
    dplasma_info_t opt;
    /* Create default options and call extended interface */
    dplasma_info_create(&opt);
    tp = hicma_parsec_syrk_New_ex(transA, transB, alpha, A, B, beta, C, opt, params_tlr, data);
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
 * @brief Destroy a GEMM taskpool and free associated resources
 * 
 * This function properly cleans up all resources associated with a GEMM taskpool,
 * including memory pools, GPU device arrays, and data type arenas.
 * 
 * @param[in,out] tp Taskpool to destroy
 */
void hicma_parsec_syrk_Destruct( parsec_taskpool_t *tp )
{
    parsec_gemm_TN_taskpool_t *gemm_tp = (parsec_gemm_TN_taskpool_t *)tp;

    if( gemm_tp->_g_gemm_type == HICMA_GEMM_NN ||
            gemm_tp->_g_gemm_type == HICMA_GEMM_NT ||
            gemm_tp->_g_gemm_type == HICMA_GEMM_TN ||
            gemm_tp->_g_gemm_type == HICMA_GEMM_TT) {
        /* Handle default GEMM taskpools */

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        /* Free GPU device index array */
        if( gemm_tp->_g_nb_cuda_devices > 0 ) {
            if( NULL != gemm_tp->_g_cuda_device_index )
                free(gemm_tp->_g_cuda_device_index);
        }
#endif 

        /* Clean up data type arenas for different precisions */
        parsec_del2arena( &gemm_tp->arenas_datatypes[PARSEC_gemm_TN_FULL_SP_ADT_IDX] );
        parsec_del2arena( &gemm_tp->arenas_datatypes[PARSEC_gemm_TN_FULL_I8_ADT_IDX] );
        parsec_del2arena( &gemm_tp->arenas_datatypes[PARSEC_gemm_TN_FULL_I32_ADT_IDX] );

        /* Finalize private memory pool */
        parsec_private_memory_fini( gemm_tp->_g_p_work_int );
        free( gemm_tp->_g_p_work_int );
    }

    /* Free the taskpool itself */
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
 * @brief Execute GEMM operation: C = α * op(A) * op(B) + β * C
 * 
 * This is the main execution function that creates a GEMM taskpool, schedules it
 * for execution, waits for completion, and cleans up resources.
 * 
 * @param[in,out] parsec PaRSEC context for task execution
 * @param[in] transA Transpose operation for matrix A
 * @param[in] transB Transpose operation for matrix B
 * @param[in] alpha Scalar multiplier for the matrix product
 * @param[in] A Input matrix A descriptor
 * @param[in] B Input matrix B descriptor
 * @param[in] beta Scalar multiplier for matrix C
 * @param[in,out] C Output matrix C descriptor (input/output)
 * @param[in] params_tlr HICMA PaRSEC parameters
 * @param[in] data HICMA PaRSEC data structures
 * @return 0 on success, negative value on error
 */
int
hicma_parsec_syrk( parsec_context_t *parsec,
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

    /* Validate input arguments */
    if ((transA != dplasmaNoTrans) && (transA != dplasmaTrans) && (transA != dplasmaConjTrans)) {
        dplasma_error("dplasma_gemm", "illegal value of transA");
        return -1;
    }
    if ((transB != dplasmaNoTrans) && (transB != dplasmaTrans) && (transB != dplasmaConjTrans)) {
        dplasma_error("dplasma_gemm", "illegal value of transB");
        return -2;
    }

    /* Determine effective dimensions of matrix A based on transpose operation */
    if ( transA == dplasmaNoTrans ) {
        Am  = A->m;
        An  = A->n;
        Amb = A->mb;
        Anb = A->nb;
        Ai  = A->i;
        Aj  = A->j;
    } else {
        /* Transpose: swap dimensions and indices */
        Am  = A->n;
        An  = A->m;
        Amb = A->nb;
        Anb = A->mb;
        Ai  = A->j;
        Aj  = A->i;
    }

    /* Determine effective dimensions of matrix B based on transpose operation */
    if ( transB == dplasmaNoTrans ) {
        Bm  = B->m;
        Bn  = B->n;
        Bmb = B->mb;
        Bnb = B->nb;
        Bi  = B->i;
        Bj  = B->j;
    } else {
        /* Transpose: swap dimensions and indices */
        Bm  = B->n;
        Bn  = B->m;
        Bmb = B->nb;
        Bnb = B->mb;
        Bi  = B->j;
        Bj  = B->i;
    }

    /* Validate matrix dimension compatibility */
    if ( (Amb != C->mb) || (Anb != Bmb) || (Bnb != C->nb) ) {
        dplasma_error("dplasma_gemm", "tile sizes have to match");
        return -101;
    }
    if ( (Am != C->m) || (An != Bm) || (Bn != C->n) ) {
        dplasma_error("dplasma_gemm", "sizes of matrices have to match");
        return -101;
    }
    if ( (Ai != C->i) || (Aj != Bi) || (Bj != C->j) ) {
        dplasma_error("dplasma_gemm", "start indexes have to match");
        return -101;
    }

    /* Set up matrix dimensions for GEMM operation */
    M = C->m;
    N = C->n;
    K = An;

    /* Create GEMM taskpool */
    parsec_gemm = hicma_parsec_syrk_New(transA, transB,
                                    alpha, A, B,
                                    beta, C, params_tlr, data);

    if ( parsec_gemm != NULL )
    {
        /* Schedule taskpool for execution and wait for completion */
        parsec_context_add_taskpool( parsec, (parsec_taskpool_t*)parsec_gemm);
        dplasma_wait_until_completion(parsec);
        /* Clean up resources */
        hicma_parsec_syrk_Destruct( parsec_gemm );
        return 0;
    }
    return -101;
}

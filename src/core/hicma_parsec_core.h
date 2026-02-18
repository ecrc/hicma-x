/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 * 
 * @version 1.0.0
 */

#ifndef HICMA_PARSEC_CORE_H
#define HICMA_PARSEC_CORE_H

/* ============================================================================
 * System includes
 * ============================================================================ */
#include <stddef.h>
#include <stdbool.h>

/* ============================================================================
 * PaRSEC includes
 * ============================================================================ */
#include "parsec.h"
#include "dplasma.h"

/* ============================================================================
 * HICMA PaRSEC includes
 * ============================================================================ */
#include "hicma_parsec_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Memory management functions
 * ============================================================================ */

/**
 * @brief Allocate memory for matrix data based on allocation type
 * 
 * This function allocates memory for matrix data using different allocation strategies
 * (CPU, CUDA, HIP) based on the specified allocation type.
 * 
 * @param[out] A Pointer to pointer that will hold the allocated memory address
 * @param[in] nb_elements Number of elements to allocate
 * @param[in] allocate_type String specifying allocation type:
 *                         - "cpu" -> malloc
 *                         - "cuda" -> cudaMallocHost
 *                         - "hip" -> hipHostMalloc
 * @param[out] data_size Size of allocated memory in bytes
 */
void hicma_parsec_core_memory_allocation(void **A, size_t nb_elements, char *allocate_type, size_t *data_size);

/**
 * @brief Configure initialized taskpool arenas to allocate pinned host memory.
 *
 * Uninitialized entries are ignored. This must be called before an arena makes
 * its first allocation so that allocation and release always use a matched pair.
 */
void hicma_parsec_arenas_use_pinned_memory(parsec_arena_datatype_t *arenas,
                                           int arena_count);

/* ============================================================================
 * Utility and debugging functions
 * ============================================================================ */

/**
 * @brief Print SYRK process timing information
 * 
 * @param[in] NT Number of tiles
 * @param[in] k Current step in factorization
 * @param[in] start_time_syrk Start time for SYRK operation
 */
void hicma_parsec_print_process_syrk(int NT, int k, double start_time_syrk);

/**
 * @brief Print general process timing information
 * 
 * @param[in] NT Number of tiles
 * @param[in] k Current step in factorization
 * @param[in] start_time_potrf Start time for POTRF operation
 */
void hicma_parsec_print_process(int NT, int k, double start_time_potrf);

/* ============================================================================
 * Operation counting functions
 * ============================================================================ */

/**
 * @brief Count operations for TRSM (triangular solve)
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] k Current step in factorization
 * @param[in] params_tlr HICMA parameters
 * @param[in] m Number of rows
 * @param[in] th_id Thread ID
 * @param[in] tempmm Temporary matrix size
 * @param[in] Arank Rank of matrix A
 */
void hicma_parsec_op_count_trsm(parsec_tiled_matrix_t *descA,
                                hicma_parsec_params_t *params_tlr,
                                int m, int k, int th_id, int tempmm, int Arank);

/**
 * @brief Count operations for SYRK (symmetric rank-k update)
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] k Current step in factorization
 * @param[in] params_tlr HICMA parameters
 * @param[in] m Number of rows
 * @param[in] th_id Thread ID
 * @param[in] tempmm Temporary matrix size
 * @param[in] rank Rank of the matrix
 */
void hicma_parsec_op_count_syrk(parsec_tiled_matrix_t *descA,
                                hicma_parsec_params_t *params_tlr,
                                int m, int k, int th_id, int tempmm, int rank);

/**
 * @brief Count operations for dense GEMM
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] k Current step in factorization
 * @param[in] params_tlr HICMA parameters
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] th_id Thread ID
 * @param[in] tempmm Temporary matrix size
 * @param[in] Crank Rank of matrix C
 * @param[in] Arank Rank of matrix A
 * @param[in] Brank Rank of matrix B
 */
void hicma_parsec_op_count_gemm_dense(parsec_tiled_matrix_t *descA,
                                      hicma_parsec_params_t *params_tlr,
                                      int m, int n, int k, int th_id, int tempmm,
                                      int Crank, int Arank, int Brank);

/* ============================================================================
 * Rank information gathering functions
 * ============================================================================ */

/**
 * @brief Gather initial rank information during Cholesky factorization
 * 
 * This function collects and stores initial rank information for matrix tiles
 * during the Cholesky factorization process. It is called at the beginning
 * of triangular solve operations to establish baseline rank data.
 * 
 * @param[in] descA Matrix descriptor containing matrix metadata
 * @param[in] descRank Rank matrix descriptor for storing rank information
 * @param[in] params_tlr HICMA parameters containing factorization settings
 * @param[in] m Row index of the current tile
 * @param[in] n Column index of the current tile
 * @param[in] k Current step in the factorization process
 * @param[in] Crank Rank of matrix C (1 for dense, actual rank for low-rank)
 * 
 * @note This function is only active when PRINT_RANK > 1
 * @note Rank information is stored in a buffer with multiple fields for tracking
 * @note The function allocates memory for rank tracking if not already allocated
 */
void hicma_parsec_gather_rank_initial(parsec_tiled_matrix_t *descA,
                                      parsec_tiled_matrix_t *descRank,
                                      hicma_parsec_params_t *params_tlr,
                                      int m, int n, int k, int Crank);

/**
 * @brief Gather final rank information during Cholesky factorization
 * 
 * This function collects and updates final rank information for matrix tiles
 * during the Cholesky factorization process. It tracks minimum, maximum, and
 * final ranks for performance analysis and debugging.
 * 
 * @param[in] descA Matrix descriptor containing matrix metadata
 * @param[in] descRank Rank matrix descriptor for storing rank information
 * @param[in] params_tlr HICMA parameters containing factorization settings
 * @param[in] m Row index of the current tile
 * @param[in] n Column index of the current tile
 * @param[in] k Current step in the factorization process
 * @param[in] Crank Rank of matrix C (1 for dense, actual rank for low-rank)
 * 
 * @note This function is only active when PRINT_RANK > 1
 * @note The function updates minimum and maximum rank values during factorization
 * @note Final rank is recorded when the tile processing is complete (n-1 == k)
 */
void hicma_parsec_gather_rank_final(parsec_tiled_matrix_t *descA,
                                    parsec_tiled_matrix_t *descRank,
                                    hicma_parsec_params_t *params_tlr,
                                    int m, int n, int k, int Crank);

/* ============================================================================
 * CPU implementation functions
 * ============================================================================ */

/**
 * @brief CPU implementation of Cholesky factorization (POTRF)
 * 
 * Performs Cholesky factorization of a symmetric positive definite matrix
 * using CPU computation.
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] params_tlr HICMA parameters
 * @param[in] es Execution stream
 * @param[in] T Workspace array
 * @param[in] k Current step in factorization
 */
void hicma_parsec_core_potrf_cpu(parsec_tiled_matrix_t* descA,
                                 hicma_parsec_params_t *params_tlr,
                                 parsec_execution_stream_t *es,
                                 void *T, int k);

/**
 * @brief CPU implementation of triangular solve (TRSM)
 * 
 * Solves triangular system of equations using CPU computation.
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] descRank Rank matrix descriptor
 * @param[in] params_tlr HICMA parameters
 * @param[in] es Execution stream
 * @param[in] p_work_full_sp Memory pool for single precision workspace
 * @param[in] T Workspace array
 * @param[in] C Result matrix
 * @param[in] m Number of rows
 * @param[in] k Current step in factorization
 * @param[in] Crank Rank of matrix C
 */
void hicma_parsec_core_trsm_cpu(parsec_tiled_matrix_t* descA,
                                parsec_tiled_matrix_t* descRank,
                                hicma_parsec_params_t *params_tlr,
                                parsec_execution_stream_t *es,
                                parsec_memory_pool_t *p_work_full_sp,
                                void *T, void *C, int m, int k, int Crank);

/**
 * @brief CPU implementation of symmetric rank-k update (SYRK)
 * 
 * Performs symmetric rank-k update using CPU computation.
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] descRank Rank matrix descriptor
 * @param[in] params_tlr HICMA parameters
 * @param[in] es Execution stream
 * @param[in] p_work Memory pool for workspace
 * @param[in] p_work_full_dp Memory pool for double precision workspace
 * @param[in] p_work_uv_dp Memory pool for UV double precision workspace
 * @param[in] p_work_mbr Memory pool for MBR workspace
 * @param[in] p_work_rr Memory pool for RR workspace
 * @param[in] T Workspace array
 * @param[in] A Input matrix
 * @param[in] m Number of rows
 * @param[in] k Current step in factorization
 * @param[in] rank Rank of matrix A
 */
void hicma_parsec_core_syrk_cpu(parsec_tiled_matrix_t* descA,
                                parsec_tiled_matrix_t* descRank,
                                hicma_parsec_params_t *params_tlr,
                                parsec_execution_stream_t *es,
                                parsec_memory_pool_t *p_work,
                                parsec_memory_pool_t *p_work_full_dp,
                                parsec_memory_pool_t *p_work_uv_dp,
                                parsec_memory_pool_t *p_work_mbr,
                                parsec_memory_pool_t *p_work_rr,
                                void *T, void *A, int m, int k, int rank);

/**
 * @brief CPU implementation of symmetric rank-k update (SYRK)
 *        Adaptive decision during runtime
 * 
 * Performs symmetric rank-k update using CPU computation.
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] descRank Rank matrix descriptor
 * @param[in] params_tlr HICMA parameters
 * @param[in] es Execution stream
 * @param[in] p_work Memory pool for workspace
 * @param[in] p_work_full_dp Memory pool for double precision workspace
 * @param[in] p_work_uv_dp Memory pool for UV double precision workspace
 * @param[in] p_work_mbr Memory pool for MBR workspace
 * @param[in] p_work_rr Memory pool for RR workspace
 * @param[in] T Workspace array
 * @param[in] A Input matrix
 * @param[in] m Number of rows
 * @param[in] k Current step in factorization
 * @param[in] rank Rank of matrix A
 */
void hicma_parsec_core_syrk_runtime_decision_cpu(parsec_tiled_matrix_t* descA,
                                parsec_tiled_matrix_t* descRank,
                                hicma_parsec_params_t *params_tlr,
                                parsec_execution_stream_t *es,
                                parsec_memory_pool_t *p_work,
                                parsec_memory_pool_t *p_work_full_dp,
                                parsec_memory_pool_t *p_work_uv_dp,
                                parsec_memory_pool_t *p_work_mbr,
                                parsec_memory_pool_t *p_work_rr,
                                void *T, void *A, int m, int k, int rank, void *A_norm);

/* ============================================================================
 * Data type conversion functions
 * ============================================================================ */

/**
 * @brief Convert matrix to FP8 bit format
 * 
 * @param[in] params_tlr HICMA parameters
 * @param[in] A Input matrix
 * @param[in] A_use Output matrix in FP8 format
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] mb Block size for rows
 * @param[in] nb Block size for columns
 */
void hicma_parsec_convert_2fp8_bit(hicma_parsec_params_t *params_tlr,
                                   float *A, float *A_use, int m, int n, int mb, int nb);

void hicma_parsec_count_datatype_conversion(hicma_parsec_params_t *params_tlr,
                                            int counter_id);

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
int hicma_parsec_gpu_counter_id(parsec_device_cuda_module_t *cuda_device,
                                parsec_cuda_exec_stream_t *cuda_stream);
#endif

/**
 * @brief Print tile matrix in new format
 * 
 * @param[in] A Matrix data
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 */
void parsec_print_tilenew(float *A, int m, int n);

/**
 * @brief CPU implementation of dense GEMM with dense C, dense A, dense B
 * 
 * Performs matrix multiplication C = alpha * A * B + beta * C where all matrices are dense.
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] descRank Rank matrix descriptor
 * @param[in] params_tlr HICMA parameters
 * @param[in] es Execution stream
 * @param[in] p_work Memory pool for workspace
 * @param[in] p_work_full_dp Memory pool for double precision workspace
 * @param[in] p_work_full_sp Memory pool for single precision workspace
 * @param[in] p_work_full_hp Memory pool for half precision workspace
 * @param[in] p_work_uv_dp Memory pool for UV double precision workspace
 * @param[in] p_work_uv_sp Memory pool for UV single precision workspace
 * @param[in] p_work_mbr Memory pool for MBR workspace
 * @param[in] p_work_rr Memory pool for RR workspace
 * @param[in] C Result matrix (dense)
 * @param[in] A Input matrix A (dense)
 * @param[in] B Input matrix B (dense)
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] k Inner dimension
 * @param[in] Crank Rank of matrix C
 * @param[in] Arank Rank of matrix A
 * @param[in] Brank Rank of matrix B
 */
void hicma_parsec_core_gemm_denseC_denseA_denseB_cpu(parsec_tiled_matrix_t* descA,
                                                     parsec_tiled_matrix_t* descRank,
                                                     hicma_parsec_params_t *params_tlr,
                                                     parsec_execution_stream_t *es,
                                                     parsec_memory_pool_t *p_work,
                                                     parsec_memory_pool_t *p_work_full_dp,
                                                     parsec_memory_pool_t *p_work_full_sp,
                                                     parsec_memory_pool_t *p_work_full_hp,
                                                     parsec_memory_pool_t *p_work_uv_dp,
                                                     parsec_memory_pool_t *p_work_uv_sp,
                                                     parsec_memory_pool_t *p_work_mbr,
                                                     parsec_memory_pool_t *p_work_rr,
                                                     void *C, void *A, void *B,
                                                     int m, int n, int k,
                                                     int Crank, int Arank, int Brank,
                                                     double Anorm, double Bnorm);
/**
 * @brief CPU implementation of dense GEMM with dense C, dense A, dense B
 *        Adaptive decision during runtime
 * 
 * Performs matrix multiplication C = alpha * A * B + beta * C where all matrices are dense.
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] descRank Rank matrix descriptor
 * @param[in] params_tlr HICMA parameters
 * @param[in] es Execution stream
 * @param[in] p_work Memory pool for workspace
 * @param[in] p_work_full_dp Memory pool for double precision workspace
 * @param[in] p_work_full_sp Memory pool for single precision workspace
 * @param[in] p_work_full_hp Memory pool for half precision workspace
 * @param[in] p_work_uv_dp Memory pool for UV double precision workspace
 * @param[in] p_work_uv_sp Memory pool for UV single precision workspace
 * @param[in] p_work_mbr Memory pool for MBR workspace
 * @param[in] p_work_rr Memory pool for RR workspace
 * @param[in] C Result matrix (dense)
 * @param[in] A Input matrix A (dense)
 * @param[in] B Input matrix B (dense)
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] k Inner dimension
 * @param[in] Crank Rank of matrix C
 * @param[in] Arank Rank of matrix A
 * @param[in] Brank Rank of matrix B
 */
void hicma_parsec_core_gemm_denseC_denseA_denseB_runtime_decision_cpu(parsec_tiled_matrix_t* descA,
                                                     parsec_tiled_matrix_t* descRank,
                                                     hicma_parsec_params_t *params_tlr,
                                                     parsec_execution_stream_t *es,
                                                     parsec_memory_pool_t *p_work,
                                                     parsec_memory_pool_t *p_work_full_dp,
                                                     parsec_memory_pool_t *p_work_full_sp,
                                                     parsec_memory_pool_t *p_work_full_hp,
                                                     parsec_memory_pool_t *p_work_uv_dp,
                                                     parsec_memory_pool_t *p_work_uv_sp,
                                                     parsec_memory_pool_t *p_work_mbr,
                                                     parsec_memory_pool_t *p_work_rr,
                                                     void *C, void *A, void *B,
                                                     int m, int n, int k,
                                                     int Crank, int Arank, int Brank,
                                                     void *Anorm, void *Bnorm);


/**
 * @brief CPU implementation of GEMM with dense C, low-rank A, dense B
 * 
 * Performs matrix multiplication C = alpha * A * B + beta * C where A is low-rank.
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] descRank Rank matrix descriptor
 * @param[in] params_tlr HICMA parameters
 * @param[in] es Execution stream
 * @param[in] p_work Memory pool for workspace
 * @param[in] p_work_full_dp Memory pool for double precision workspace
 * @param[in] p_work_full_sp Memory pool for single precision workspace
 * @param[in] p_work_full_hp Memory pool for half precision workspace
 * @param[in] p_work_uv_dp Memory pool for UV double precision workspace
 * @param[in] p_work_uv_sp Memory pool for UV single precision workspace
 * @param[in] p_work_mbr Memory pool for MBR workspace
 * @param[in] p_work_rr Memory pool for RR workspace
 * @param[in] C Result matrix (dense)
 * @param[in] A Input matrix A (low-rank)
 * @param[in] B Input matrix B (dense)
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] k Inner dimension
 * @param[in] Crank Rank of matrix C
 * @param[in] Arank Rank of matrix A
 * @param[in] Brank Rank of matrix B
 */
void hicma_parsec_core_gemm_denseC_lrA_denseB_cpu(parsec_tiled_matrix_t* descA,
                                                  parsec_tiled_matrix_t* descRank,
                                                  hicma_parsec_params_t *params_tlr,
                                                  parsec_execution_stream_t *es,
                                                  parsec_memory_pool_t *p_work,
                                                  parsec_memory_pool_t *p_work_full_dp,
                                                  parsec_memory_pool_t *p_work_full_sp,
                                                  parsec_memory_pool_t *p_work_full_hp,
                                                  parsec_memory_pool_t *p_work_uv_dp,
                                                  parsec_memory_pool_t *p_work_uv_sp,
                                                  parsec_memory_pool_t *p_work_mbr,
                                                  parsec_memory_pool_t *p_work_rr,
                                                  void *C, void *A, void *B,
                                                  int m, int n, int k,
                                                  int Crank, int Arank, int Brank);

/**
 * @brief CPU implementation of GEMM with dense C, low-rank A, low-rank B
 * 
 * Performs matrix multiplication C = alpha * A * B + beta * C where A and B are low-rank.
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] descRank Rank matrix descriptor
 * @param[in] params_tlr HICMA parameters
 * @param[in] es Execution stream
 * @param[in] p_work Memory pool for workspace
 * @param[in] p_work_full_dp Memory pool for double precision workspace
 * @param[in] p_work_full_sp Memory pool for single precision workspace
 * @param[in] p_work_full_hp Memory pool for half precision workspace
 * @param[in] p_work_uv_dp Memory pool for UV double precision workspace
 * @param[in] p_work_uv_sp Memory pool for UV single precision workspace
 * @param[in] p_work_mbr Memory pool for MBR workspace
 * @param[in] p_work_rr Memory pool for RR workspace
 * @param[in] C Result matrix (dense)
 * @param[in] A Input matrix A (low-rank)
 * @param[in] B Input matrix B (low-rank)
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] k Inner dimension
 * @param[in] Crank Rank of matrix C
 * @param[in] Arank Rank of matrix A
 * @param[in] Brank Rank of matrix B
 */
void hicma_parsec_core_gemm_denseC_lrA_lrB_cpu(parsec_tiled_matrix_t* descA,
                                               parsec_tiled_matrix_t* descRank,
                                               hicma_parsec_params_t *params_tlr,
                                               parsec_execution_stream_t *es,
                                               parsec_memory_pool_t *p_work,
                                               parsec_memory_pool_t *p_work_full_dp,
                                               parsec_memory_pool_t *p_work_full_sp,
                                               parsec_memory_pool_t *p_work_full_hp,
                                               parsec_memory_pool_t *p_work_uv_dp,
                                               parsec_memory_pool_t *p_work_uv_sp,
                                               parsec_memory_pool_t *p_work_mbr,
                                               parsec_memory_pool_t *p_work_rr,
                                               void *C, void *A, void *B,
                                               int m, int n, int k,
                                               int Crank, int Arank, int Brank);

/**
 * @brief SVD-based low-rank GEMM implementation
 * 
 * Performs matrix multiplication using SVD decomposition for low-rank matrices.
 * 
 * @param[in] C Result matrix
 * @param[in] A Input matrix A
 * @param[in] B Input matrix B
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] k Inner dimension
 * @param[in] Crank Rank of matrix C
 * @param[in] Arank Rank of matrix A
 * @param[in] Brank Rank of matrix B
 */
void hicma_parsec_core_dgemm_lr_svd(void *C, void *A, void *B, int m, int n, int k,
                                    int Crank, int Arank, int Brank);

/**
 * @brief SVD-based low-rank GEMM with low-rank result
 * 
 * Performs matrix multiplication using SVD decomposition where result is also low-rank.
 * 
 * @param[in] C Result matrix (low-rank)
 * @param[in] A Input matrix A (low-rank)
 * @param[in] B Input matrix B (low-rank)
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] k Inner dimension
 * @param[in] Crank Rank of matrix C
 * @param[in] Arank Rank of matrix A
 * @param[in] Brank Rank of matrix B
 */
void hicma_parsec_core_dgemm_lrA_lrB_lrC_svd(void *C, void *A, void *B, int m, int n, int k,
                                              int Crank, int Arank, int Brank);

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

/**
 * @brief GPU implementation of double precision GEMM
 * 
 * @param[in] handle cuBLAS handle
 * @param[in] transa Operation on matrix A
 * @param[in] transb Operation on matrix B
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] k Inner dimension
 * @param[in] alpha Scalar multiplier
 * @param[in] A Input matrix A
 * @param[in] lda Leading dimension of A
 * @param[in] B Input matrix B
 * @param[in] ldb Leading dimension of B
 * @param[in] beta Scalar multiplier
 * @param[out] C Result matrix
 * @param[in] ldc Leading dimension of C
 * @return cuBLAS status
 */
cublasStatus_t hicma_parsec_dgemm_gpu(cublasHandle_t handle,
                                      cublasOperation_t transa, cublasOperation_t transb,
                                      int m, int n, int k,
                                      const double *alpha,
                                      const double *A, int lda,
                                      const double *B, int ldb,
                                      const double *beta,
                                      double *C, int ldc);

/**
 * @brief GPU implementation of single precision GEMM
 * 
 * @param[in] handle cuBLAS handle
 * @param[in] transa Operation on matrix A
 * @param[in] transb Operation on matrix B
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] k Inner dimension
 * @param[in] alpha Scalar multiplier
 * @param[in] A Input matrix A
 * @param[in] lda Leading dimension of A
 * @param[in] B Input matrix B
 * @param[in] ldb Leading dimension of B
 * @param[in] beta Scalar multiplier
 * @param[out] C Result matrix
 * @param[in] ldc Leading dimension of C
 * @param[in] tensor_gemm Tensor core GEMM flag
 * @return cuBLAS status
 */
cublasStatus_t hicma_parsec_sgemm_gpu(cublasHandle_t handle,
                                      cublasOperation_t transa, cublasOperation_t transb,
                                      int m, int n, int k,
                                      const float *alpha,
                                      const float *A, int lda,
                                      const float *B, int ldb,
                                      const float *beta,
                                      float *C, int ldc,
                                      int tensor_gemm);

/**
 * @brief GPU implementation of Cholesky factorization (POTRF)
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] params_tlr HICMA parameters
 * @param[in] ws_handle_cusolver cuSOLVER workspace handle
 * @param[in] cuda_device CUDA device module
 * @param[in] gpu_task GPU task descriptor
 * @param[in] cuda_stream CUDA execution stream
 * @param[in] T Workspace array
 * @param[in] k Current step in factorization
 */
void hicma_parsec_core_potrf_gpu(parsec_tiled_matrix_t* descA,
                                 hicma_parsec_params_t *params_tlr,
                                 parsec_potrf_workspace_t *ws_handle_cusolver,
                                 parsec_device_cuda_module_t *cuda_device,
                                 parsec_gpu_task_t *gpu_task,
                                 parsec_cuda_exec_stream_t *cuda_stream,
                                 void *T, int k);

/**
 * @brief GPU implementation of triangular solve (TRSM)
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] params_tlr HICMA parameters
 * @param[in] ws_gpu GPU workspace
 * @param[in] cuda_device CUDA device module
 * @param[in] gpu_task GPU task descriptor
 * @param[in] cuda_stream CUDA execution stream
 * @param[in] T Workspace array
 * @param[in] C Result matrix
 * @param[in] m Number of rows
 * @param[in] k Current step in factorization
 */
void hicma_parsec_core_trsm_gpu(parsec_tiled_matrix_t* descA,
                                hicma_parsec_params_t *params_tlr,
                                parsec_potrf_stream_workspace_t *stream_found,
                                parsec_device_cuda_module_t *cuda_device,
                                parsec_gpu_task_t *gpu_task,
                                parsec_cuda_exec_stream_t *cuda_stream,
                                void *T, void *C, int m, int k);

/**
 * @brief GPU implementation of symmetric rank-k update (SYRK)
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] params_tlr HICMA parameters
 * @param[in] ws_gpu GPU workspace
 * @param[in] cuda_device CUDA device module
 * @param[in] gpu_task GPU task descriptor
 * @param[in] cuda_stream CUDA execution stream
 * @param[in] T Workspace array
 * @param[in] A Input matrix
 * @param[in] m Number of rows
 * @param[in] k Current step in factorization
 * @param[in] Arank Rank of matrix A
 */
void hicma_parsec_core_syrk_gpu(parsec_tiled_matrix_t* descA,
                                hicma_parsec_params_t *params_tlr,
                                parsec_potrf_workspace_t *ws_gpu,
                                parsec_device_cuda_module_t *cuda_device,
                                parsec_gpu_task_t *gpu_task,
                                parsec_cuda_exec_stream_t *cuda_stream,
                                void *T, void *A, int m, int k, int Arank);

void hicma_parsec_core_syrk_runtime_decision_gpu(parsec_tiled_matrix_t* descA,
                                                 hicma_parsec_params_t *params_tlr,
                                                 parsec_potrf_workspace_t *ws_gpu,
                                                 parsec_device_cuda_module_t *cuda_device,
                                                 parsec_gpu_task_t *gpu_task,
                                                 parsec_cuda_exec_stream_t *cuda_stream,
                                                 void *T, void *A, int m, int k, int Arank, void *A_norm);

/**
 * @brief GPU implementation of dense GEMM with dense C, dense A, dense B
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] params_tlr HICMA parameters
 * @param[in] ws_gpu GPU workspace
 * @param[in] cuda_device CUDA device module
 * @param[in] gpu_task GPU task descriptor
 * @param[in] cuda_stream CUDA execution stream
 * @param[in] C Result matrix (dense)
 * @param[in] A Input matrix A (dense)
 * @param[in] B Input matrix B (dense)
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] k Inner dimension
 * @param[in] Crank Rank of matrix C
 * @param[in] Arank Rank of matrix A
 * @param[in] Brank Rank of matrix B
 */
void hicma_parsec_core_gemm_denseC_denseA_denseB_gpu(parsec_tiled_matrix_t* descA,
                                                     hicma_parsec_params_t *params_tlr,
                                                     parsec_potrf_workspace_t *ws_gpu,
                                                     parsec_device_cuda_module_t *cuda_device,
                                                     parsec_gpu_task_t *gpu_task,
                                                     parsec_cuda_exec_stream_t *cuda_stream,
                                                     void *C, void *A, void *B, int m, int n, int k,
                                                     int Crank, int Arank, int Brank);

void hicma_parsec_core_gemm_denseC_denseA_denseB_runtime_decision_gpu(void *this_task_,
                                                                      parsec_tiled_matrix_t* descA,
                                                                      hicma_parsec_params_t *params_tlr,
                                                                      parsec_potrf_workspace_t *ws_gpu,
                                                                      parsec_device_cuda_module_t *cuda_device,
                                                                      parsec_gpu_task_t *gpu_task,
                                                                      parsec_cuda_exec_stream_t *cuda_stream,
                                                                      void *C, void *A, void *B, int m, int n, int k,
                                                                      int Crank, int Arank, int Brank,
                                                                      void *A_norm, void *B_norm);

/**
 * @brief GPU implementation of GEMM with dense C, low-rank A, dense B
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] params_tlr HICMA parameters
 * @param[in] ws_gpu GPU workspace
 * @param[in] cuda_device CUDA device module
 * @param[in] gpu_task GPU task descriptor
 * @param[in] cuda_stream CUDA execution stream
 * @param[in] C Result matrix (dense)
 * @param[in] A Input matrix A (low-rank)
 * @param[in] B Input matrix B (dense)
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] k Inner dimension
 * @param[in] Crank Rank of matrix C
 * @param[in] Arank Rank of matrix A
 * @param[in] Brank Rank of matrix B
 */
void hicma_parsec_core_gemm_denseC_lrA_denseB_gpu(parsec_tiled_matrix_t* descA,
                                                  hicma_parsec_params_t *params_tlr,
                                                  parsec_potrf_workspace_t *ws_gpu,
                                                  parsec_device_cuda_module_t *cuda_device,
                                                  parsec_gpu_task_t *gpu_task,
                                                  parsec_cuda_exec_stream_t *cuda_stream,
                                                  void *C, void *A, void *B, int m, int n, int k,
                                                  int Crank, int Arank, int Brank);

/**
 * @brief GPU implementation of GEMM with dense C, low-rank A, low-rank B
 * 
 * @param[in] descA Matrix descriptor
 * @param[in] params_tlr HICMA parameters
 * @param[in] ws_gpu GPU workspace
 * @param[in] cuda_device CUDA device module
 * @param[in] gpu_task GPU task descriptor
 * @param[in] cuda_stream CUDA execution stream
 * @param[in] C Result matrix (dense)
 * @param[in] A Input matrix A (low-rank)
 * @param[in] B Input matrix B (low-rank)
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] k Inner dimension
 * @param[in] Crank Rank of matrix C
 * @param[in] Arank Rank of matrix A
 * @param[in] Brank Rank of matrix B
 */
void hicma_parsec_core_gemm_denseC_lrA_lrB_gpu(parsec_tiled_matrix_t* descA,
                                               hicma_parsec_params_t *params_tlr,
                                               parsec_potrf_workspace_t *ws_gpu,
                                               parsec_device_cuda_module_t *cuda_device,
                                               parsec_gpu_task_t *gpu_task,
                                               parsec_cuda_exec_stream_t *cuda_stream,
                                               void *C, void *A, void *B, int m, int n, int k,
                                               int Crank, int Arank, int Brank);

#endif /* HICMA_PARSEC_HAVE_CUDA */

/**
 * @brief Calculate matrix norm for a tile based on datatype
 * 
 * This function calculates the norm of a matrix tile based on the specified datatype.
 * It supports both row-major and column-major memory layouts.
 * 
 * @param[in] data Pointer to the matrix tile data
 * @param[in] tempmm Number of rows in the tile
 * @param[in] tempnn Number of columns in the tile
 * @param[in] lda Leading dimension
 * @param[in] datatype_str String representation of the datatype. Supported values:
 *                       - "double", "d" -> double precision floating point
 *                       - "float", "single", "s" -> single precision floating point
 *                       - "int8", "i8" -> 8-bit signed integer
 *                       - "int16", "i16" -> 16-bit signed integer
 *                       - "int32", "int", "i32", "i" -> 32-bit signed integer
 *                       - "int64", "i64" -> 64-bit signed integer
 *                       - "uint8", "u8" -> 8-bit unsigned integer
 *                       - "uint16", "u16" -> 16-bit unsigned integer
 *                       - "uint32", "uint", "u32", "u" -> 32-bit unsigned integer
 *                       - "uint64", "u64" -> 64-bit unsigned integer
 *                       - "half", "fp16", "h" -> 16-bit floating point (half precision)
 *                       - "fp8" -> 8-bit floating point representation
 *                       - "fp4" -> 4-bit floating point representation
 *                       - "int4" -> 4-bit integer representation
 *                       - "1bit" -> 1-bit representation
 * @param[in] is_row_major true for row-major layout, false for column-major layout
 * @param[in] is_lower true if lower triangular, false otherwise
 * @param[in] is_upper true if upper triangular, false otherwise
 * @param[in] is_diagonal true if diagonal tile (m == n), false otherwise
 * 
 * @return The calculated norm value (sum of squares, not square root)
 * 
 * @note This function returns the sum of squares of matrix elements, not the square root.
 *       The caller should take the square root if the actual norm is needed.
 * 
 * @note For triangular matrices, only the specified triangular part is included in the computation.
 *       For diagonal tiles, the triangular pattern depends on the is_lower and is_upper flags.
 */
double hicma_parsec_core_matrix_norm_get(const void *data, int tempmm, int tempnn, int lda,
                                        const char *datatype_str, bool is_row_major,
                                        bool is_lower, bool is_upper, bool is_diagonal);

#ifdef __cplusplus
}
#endif

#endif /* HICMA_PARSEC_CORE_H */

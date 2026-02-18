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

#ifndef HICMA_PARSEC_DECISION_H
#define HICMA_PARSEC_DECISION_H

/* ============================================================================
 * HICMA PaRSEC includes
 * ============================================================================ */
#include "hicma_parsec_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Decision update and management functions
 * ============================================================================ */

/**
 * @brief Update decisions for each tile
 * 
 * Updates the decision matrix for each tile based on current computation state.
 * This decision is used for matrix generation and Cholesky factorization.
 * The function analyzes the current matrix state and updates precision decisions
 * for optimal performance and accuracy.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 */
void hicma_parsec_decisions_update(parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params);

/**
 * @brief Initialize decisions for each tile
 * 
 * Initializes the decision matrix for each tile. This decision is used for
 * matrix generation and Cholesky factorization. The function sets up initial
 * precision decisions based on tile position and configuration parameters.
 * 
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_decision_init(hicma_parsec_params_t *params);

/* ============================================================================
 * Decision making functions
 * ============================================================================ */

/**
 * @brief Make decisions based on norm approach
 * 
 * Makes decisions about computation strategy based on matrix norm analysis.
 * This function determines the optimal precision for each tile based on
 * numerical properties and performance requirements.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] A Matrix descriptor
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_decision_make_comp(parsec_context_t *parsec,
                         parsec_tiled_matrix_t *A,
                         hicma_parsec_params_t *params);

/**
 * @brief Make decisions about datatype sending
 * 
 * Makes decisions about which datatype to use when sending data between processes.
 * This function optimizes communication overhead by selecting appropriate
 * precision levels for data transmission.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] A Matrix descriptor
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_decision_make_send(parsec_context_t *parsec,
                         parsec_tiled_matrix_t *A,
                         hicma_parsec_params_t *params);

/* ============================================================================
 * Datatype decision functions
 * ============================================================================ */

/**
 * @brief Get datatype for POTRF dense TLR mixed precision tile
 * 
 * Determines the optimal datatype for a specific tile in dense TLR mixed precision
 * Cholesky factorization (POTRF) operations. The decision is based on the tile's
 * position and the overall computation strategy.
 * 
 * @param[in] decisions Decision array containing datatype choices for all tiles
 * @param[in] m Row index of the tile
 * @param[in] n Column index of the tile
 * @param[in] NT Total number of tiles in the matrix
 * @return Datatype identifier for the specified tile
 */
int decision_datatype_tile_potrf_L_dense_tlr_mp(uint16_t *decisions, int m, int n, int NT);

/**
 * @brief Get datatype for POTRF dense mixed precision GPU tile
 * 
 * Determines the optimal datatype for a specific tile in dense mixed precision
 * Cholesky factorization (POTRF) operations on GPU. The decision considers
 * GPU memory constraints and performance characteristics.
 * 
 * @param[in] decisions Decision array containing datatype choices for all tiles
 * @param[in] m Row index of the tile
 * @param[in] n Column index of the tile
 * @param[in] NT Total number of tiles in the matrix
 * @return Datatype identifier for the specified tile
 */
int decision_datatype_tile_potrf_L_dense_mp_gpu(uint16_t *decisions, int m, int n, int NT);

/**
 * @brief Get datatype for POTRF dense mixed precision GPU FP8 tile
 * 
 * Determines the optimal datatype for a specific tile in dense mixed precision
 * Cholesky factorization (POTRF) operations on GPU using FP8 arithmetic.
 * The decision optimizes for memory efficiency while maintaining numerical stability.
 * 
 * @param[in] decisions Decision array containing datatype choices for all tiles
 * @param[in] m Row index of the tile
 * @param[in] n Column index of the tile
 * @param[in] NT Total number of tiles in the matrix
 * @return Datatype identifier for the specified tile
 */
int decision_datatype_tile_potrf_L_dense_mp_gpu_fp8(uint16_t *decisions, int m, int n, int NT);


/**
 * @brief Get datatype for POTRF dense mixed precision GPU FP8 tile adaptive
 * 
 * Determines the optimal datatype for a specific tile in dense mixed precision
 * Cholesky factorization (POTRF) operations on GPU using FP8 arithmetic.
 * The decision optimizes for memory efficiency while maintaining numerical stability.
 * 
 * @param[in] decisions Decision array containing datatype choices for all tiles
 * @param[in] m Row index of the tile
 * @param[in] n Column index of the tile
 * @param[in] NT Total number of tiles in the matrix
 * @return Datatype identifier for the specified tile
 */
int decision_datatype_tile_potrf_L_dense_mp_gpu_fp8_adaptive(uint16_t *decisions, int m, int n, int NT);

/**
 * @brief Get datatype for POTRF dense mixed precision GPU FP8 single precision tile
 * 
 * Determines the optimal datatype for a specific tile in dense mixed precision
 * Cholesky factorization (POTRF) operations on GPU using FP8 and single precision
 * arithmetic. This combination provides a balance between memory efficiency and
 * numerical accuracy.
 * 
 * @param[in] decisions Decision array containing datatype choices for all tiles
 * @param[in] m Row index of the tile
 * @param[in] n Column index of the tile
 * @param[in] NT Total number of tiles in the matrix
 * @return Datatype identifier for the specified tile
 */
int decision_datatype_tile_potrf_L_dense_mp_gpu_fp8_sp(uint16_t *decisions, int m, int n, int NT);

/**
 * @brief Get datatype for GEMM mixed precision TN tile
 * 
 * Determines the optimal datatype for a specific tile in mixed precision
 * General Matrix Multiply (GEMM) operations with transposed first matrix (TN).
 * The decision optimizes for performance and memory usage.
 * 
 * @param[in] decisions Decision array containing datatype choices for all tiles
 * @param[in] m Row index of the tile
 * @param[in] n Column index of the tile
 * @param[in] NT Total number of tiles in the matrix
 * @return Datatype identifier for the specified tile
 */
int decision_datatype_tile_gemmmp_TN(uint16_t *decisions, int m, int n, int NT);

/**
 * @brief Get datatype for conversion tile
 * 
 * Determines the optimal datatype for a specific tile during datatype conversion
 * operations. The decision considers the source and target datatypes and
 * conversion efficiency requirements.
 * 
 * @param[in] decisions Decision array containing datatype choices for all tiles
 * @param[in] m Row index of the tile
 * @param[in] n Column index of the tile
 * @param[in] NT Total number of tiles in the matrix
 * @return Datatype identifier for the specified tile
 */
int decision_datatype_tile_convert(uint16_t *decisions, int m, int n, int NT);

/**
 * @brief Get output datatype for a tile
 * 
 * Determines the output datatype for a specific tile based on the decision array.
 * This function is used to determine the final precision of computation results.
 * 
 * @param[in] decisions Decision array containing datatype choices for all tiles
 * @param[in] m Row index of the tile
 * @param[in] n Column index of the tile
 * @param[in] NT Total number of tiles in the matrix
 * @return Datatype identifier for the output
 */
int decision_datatype_out(uint16_t *decisions, int m, int n, int NT);

/**
 * @brief Get output size for a tile
 * 
 * Calculates the output size in bytes for a specific tile based on the decision
 * array and block dimensions. This is used for memory allocation and data transfer.
 * 
 * @param[in] decisions Decision array containing datatype choices for all tiles
 * @param[in] m Row index of the tile
 * @param[in] n Column index of the tile
 * @param[in] NT Total number of tiles in the matrix
 * @param[in] mb Block size for rows
 * @param[in] nb Block size for columns
 * @param[in] size Base size in bytes
 * @return Output size in bytes
 */
int decision_size_out(uint16_t *decisions, int m, int n, int NT,
        int mb, int nb, int size);

/**
 * @brief Get datatype for sending POTRF dense mixed precision GPU tile
 * 
 * Determines the optimal datatype for sending a specific tile during dense mixed
 * precision Cholesky factorization (POTRF) operations on GPU. The decision
 * considers communication overhead and memory bandwidth.
 * 
 * @param[in] decisions Decision array containing datatype choices for all tiles
 * @param[in] m Row index of the tile
 * @param[in] n Column index of the tile
 * @param[in] NT Total number of tiles in the matrix
 * @return Datatype identifier for the specified tile
 */
int decision_datatype_tile_send_potrf_L_dense_mp_gpu(uint16_t *decisions, int m, int n, int NT);

/**
 * @brief Get datatype for sending POTRF dense mixed precision GPU FP8 tile
 * 
 * Determines the optimal datatype for sending a specific tile during dense mixed
 * precision Cholesky factorization (POTRF) operations on GPU using FP8 arithmetic.
 * The decision optimizes for communication efficiency and memory usage.
 * 
 * @param[in] decisions Decision array containing datatype choices for all tiles
 * @param[in] m Row index of the tile
 * @param[in] n Column index of the tile
 * @param[in] NT Total number of tiles in the matrix
 * @return Datatype identifier for the specified tile
 */
int decision_datatype_tile_send_potrf_L_dense_mp_gpu_fp8(uint16_t *decisions, int m, int n, int NT);

/**
 * @brief Get datatype for sending POTRF dense mixed precision GPU FP8 adaptive
 * 
 * Determines the optimal datatype for sending a specific tile during dense mixed
 * precision Cholesky factorization (POTRF) operations on GPU using FP8 arithmetic.
 * The decision optimizes for communication efficiency and memory usage.
 * 
 * @param[in] decisions Decision array containing datatype choices for all tiles
 * @param[in] m Row index of the tile
 * @param[in] n Column index of the tile
 * @param[in] NT Total number of tiles in the matrix
 * @return Datatype identifier for the specified tile
 */
int decision_datatype_tile_send_potrf_L_dense_mp_gpu_fp8_adaptive(uint16_t *decisions, int m, int n, int NT);

/**
 * @brief Get datatype for sending POTRF dense mixed precision GPU FP8 single precision tile
 * 
 * Determines the optimal datatype for sending a specific tile during dense mixed
 * precision Cholesky factorization (POTRF) operations on GPU using FP8 and single
 * precision arithmetic. The decision balances communication efficiency with
 * numerical accuracy requirements.
 * 
 * @param[in] decisions Decision array containing datatype choices for all tiles
 * @param[in] m Row index of the tile
 * @param[in] n Column index of the tile
 * @param[in] NT Total number of tiles in the matrix
 * @return Datatype identifier for the specified tile
 */
int decision_datatype_tile_send_potrf_L_dense_mp_gpu_fp8_sp(uint16_t *decisions, int m, int n, int NT);

/* ============================================================================
 * Decision analysis and counting functions
 * ============================================================================ */

/**
 * @brief Count decisions for matrix
 * 
 * Counts the number of tiles with different precision decisions across the matrix.
 * This function provides statistics about the distribution of precision choices
 * for performance analysis and debugging.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] A Matrix descriptor
 * @param[in] params HICMA PaRSEC parameters
 * @return Number of decisions counted
 */
int hicma_parsec_decision_count(parsec_context_t *parsec,
        parsec_tiled_matrix_t *A,
        hicma_parsec_params_t *params);

/**
 * @brief Initialize decisions for GEMM GPU operations
 * 
 * Initializes the decision matrix specifically for GEMM operations on GPU.
 * This function sets up precision decisions optimized for GPU-based matrix
 * multiplication operations.
 * 
 * @param[in] params HICMA PaRSEC parameters
 */
void hicma_parsec_decisions_gemm_gpu_init(hicma_parsec_params_t *params);

/* ============================================================================
 * Decision printing and analysis functions
 * ============================================================================ */

/**
 * @brief Print decisions matrix
 * 
 * Prints the decision matrix in a human-readable format with color coding.
 * This function is useful for debugging and understanding the precision
 * distribution across the matrix.
 * 
 * @param[in] params HICMA PaRSEC parameters
 */
void print_decisions(hicma_parsec_params_t *params);
void print_first_local_gemm_decisions(const hicma_parsec_params_t *params);
void print_last_local_gemm_decisions(const hicma_parsec_params_t *params);
void print_final_storage_decisions(const hicma_parsec_params_t *params);


void get_decisions(uint16_t *decisions, size_t size,
        parsec_tiled_matrix_t *descA );

/**
 * @brief Print convert type decisions
 * 
 * Prints the decision matrix for data conversion operations with color coding.
 * This helps visualize how data types are chosen for conversion operations.
 * 
 * @param[in] params HICMA PaRSEC parameters
 */
void print_decisions_send(hicma_parsec_params_t *params);

/**
 * @brief Print GEMM GPU decisions
 * 
 * Prints the decision matrix for GEMM operations on GPU with detailed mask
 * information. This is useful for debugging GPU computation strategies.
 * 
 * @param[in] params HICMA PaRSEC parameters
 */
void print_decisions_gemm_gpu(hicma_parsec_params_t *params);

/**
 * @brief Analyze decisions for sending data
 * 
 * Analyzes the decision matrix to determine optimal data types for communication.
 * This function optimizes the trade-off between precision and communication
 * overhead based on the computation requirements.
 * 
 * @param[in] params HICMA PaRSEC parameters
 */
void hicma_parsec_decisions_send_analysis(hicma_parsec_params_t *params);

/* ============================================================================
 * Matrix sum functions
 * ============================================================================ */

/**
 * @brief Calculate double precision matrix sum
 * 
 * Computes the sum of all matrix elements using double precision arithmetic.
 * This function is used for verification and analysis of computation results.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] A Matrix descriptor
 * @param[in] Ar Rank matrix descriptor
 * @param[in] decisions Decision array
 * @return Sum of matrix elements in double precision
 */
double hicma_parsec_matrix_dsum(parsec_context_t *parsec,
                         parsec_tiled_matrix_t *A,
                         parsec_tiled_matrix_t *Ar,
                         uint16_t *decisions);

/**
 * @brief Calculate single precision matrix sum
 * 
 * Computes the sum of all matrix elements using single precision arithmetic.
 * This function is used for verification and analysis of computation results.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] A Matrix descriptor
 * @param[in] Ar Rank matrix descriptor
 * @param[in] decisions Decision array
 * @return Sum of matrix elements in single precision
 */
float hicma_parsec_matrix_ssum(parsec_context_t *parsec,
                         parsec_tiled_matrix_t *A,
                         parsec_tiled_matrix_t *Ar,
                         uint16_t *decisions);

/**
 * @brief Determine the precision of that tile 
 *
 * @return The precision decision 
 */
void hicma_parsec_get_precision_tile(hicma_parsec_params_t *params_tlr,
        hicma_parsec_decision_enum_t *new_decision, double norm_tile, int m, int n);

#ifdef __cplusplus
}
#endif

#endif /* HICMA_PARSEC_DECISION_H */ 

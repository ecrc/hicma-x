/**
 * @file hicma_parsec_datatype_convert.h
 * @brief Comprehensive datatype conversion interface for HICMA PaRSEC operations
 * 
 * This header file provides a complete interface for datatype conversions in the HICMA
 * (Hierarchical Computations for Manycore Architectures) library using PaRSEC runtime.
 * It supports conversions between various numeric precisions and formats commonly used
 * in high-performance linear algebra computations.
 * 
 * **Key Features:**
 * - Unified conversion functions that handle all datatype conversions based on type strings
 * - Individual conversion functions for specific datatype pairs with optimized implementations
 * - Both unary (in-place) and binary (separate source/target) conversion variants
 * - Support for floating-point precisions: double, single, half (if available)
 * - Support for integer types: 8-bit, 16-bit, 32-bit, 64-bit (signed and unsigned)
 * - PaRSEC tiled matrix to LAPACK format conversions and vice versa
 * - Utility functions for datatype size queries, array initialization, and debugging
 * 
 * **Supported Conversion Types:**
 * - Floating-point: double ↔ single ↔ half precision
 * - Integer: 8-bit, 16-bit, 32-bit, 64-bit (signed/unsigned)
 * - Mixed: floating-point ↔ integer conversions
 * - Special formats: FP8, FP4, INT4, 1-bit representations
 * 
 * **Performance Considerations:**
 * - Optimized CPU implementations for each conversion type
 * - Memory-efficient in-place conversions where possible
 * - PaRSEC task-based parallel execution for large matrices
 * - Bit manipulation optimizations for half precision conversions
 * 
 * @version 0.2.0
 * @see hicma_parsec_internal.h for internal HICMA PaRSEC definitions
 */

#ifndef HICMA_PARSEC_DATATYPE_CONVERT_H
#define HICMA_PARSEC_DATATYPE_CONVERT_H

#include "hicma_parsec_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omnicast/omnicast.h"


#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
   HICMA PaRSEC High-Level Conversion Functions
   
   These functions provide the main interface for datatype conversions in HICMA
   applications. They handle the complete conversion workflow including memory
   management, task scheduling, and error handling through the PaRSEC runtime.
   ============================================================================ */

/**
 * @brief Convert double precision matrix to single precision using HICMA PaRSEC
 * 
 * This function performs a high-level conversion of a double precision matrix to single
 * precision format using the HICMA PaRSEC framework. It handles memory management,
 * task scheduling, and parallel execution automatically.
 * 
 * **Conversion Process:**
 * - Analyzes the input matrix structure and dimensions
 * - Allocates appropriate memory for single precision data if needed
 * - Schedules conversion tasks across available compute resources
 * - Updates the data structure to reflect the new precision
 * 
 * **Memory Management:**
 * - May reallocate memory if the target precision requires different storage
 * - Preserves matrix structure and metadata during conversion
 * - Handles both dense and sparse matrix formats
 * 
 * **Error Handling:**
 * - Returns 0 on successful completion
 * - Returns non-zero error code on failure (memory allocation, invalid parameters, etc.)
 * - Provides detailed error information through HICMA error reporting system
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution.
 *                    Must be a valid, initialized PaRSEC context.
 * @param[in] data HICMA PaRSEC data structure containing matrix information.
 *                 Must contain valid double precision matrix data.
 * @param[in] params_tlr HICMA PaRSEC parameters for the conversion.
 *                       Controls conversion behavior and performance settings.
 * @return 0 on success, non-zero error code on failure
 * 
 * @note This function is blocking and will not return until conversion is complete
 * @note Memory usage may increase temporarily during conversion
 * @note The original double precision data may be freed after successful conversion
 * 
 * @see hicma_parsec_convert_s2d() for reverse conversion
 * @see hicma_parsec_params_t for parameter structure details
 */
int hicma_parsec_convert_d2s(parsec_context_t *parsec,
                         hicma_parsec_data_t *data,
                         hicma_parsec_params_t *params_tlr);

/**
 * @brief Convert double precision matrix to integer format using PaRSEC
 * 
 * This function converts a double precision tiled matrix to the specified integer format
 * using the PaRSEC runtime for parallel execution. The conversion applies truncation
 * to the floating-point values to obtain integer representations.
 * 
 * **Supported Integer Types:**
 * - 8: Convert to int8_t (8-bit signed integer, range: -128 to 127)
 * - 16: Convert to int16_t (16-bit signed integer, range: -32,768 to 32,767)
 * - 32: Convert to int32_t (32-bit signed integer, range: -2^31 to 2^31-1)
 * - 64: Convert to int64_t (64-bit signed integer, range: -2^63 to 2^63-1)
 * 
 * **Conversion Behavior:**
 * - Floating-point values are truncated (not rounded) to the nearest integer
 * - Values outside the target integer range are clamped to the range limits
 * - NaN and infinity values are converted to 0
 * - The conversion is performed in parallel across matrix tiles
 * 
 * **Performance Notes:**
 * - Uses PaRSEC task scheduling for optimal parallel execution
 * - Memory usage is reduced when converting to smaller integer types
 * - Conversion is performed in-place to minimize memory overhead
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution.
 *                    Must be a valid, initialized PaRSEC context.
 * @param[in] A Tiled matrix to convert. Must contain valid double precision data.
 *              The matrix is modified in-place to contain integer data.
 * @param[in] intype Target integer type specification:
 *                   - 8: int8_t (1 byte per element)
 *                   - 16: int16_t (2 bytes per element)
 *                   - 32: int32_t (4 bytes per element)
 *                   - 64: int64_t (8 bytes per element)
 * @return 0 on success, non-zero error code on failure
 * 
 * @note This function modifies the input matrix in-place
 * @note Values are truncated, not rounded, during conversion
 * @note Out-of-range values are clamped to the target integer range
 * 
 * @see parsec_convert_i2d() for reverse conversion
 * @see parsec_convert_s2i() for single precision to integer conversion
 */
int parsec_convert_d2i(parsec_context_t *parsec,
                         parsec_tiled_matrix_t *A, int intype);

/**
 * @brief Convert integer matrix to double precision format
 * 
 * Converts an integer matrix to double precision format.
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution
 * @param[in] A Tiled matrix to convert
 * @param[in] intype Source integer type (e.g., 8, 16, 32, 64 for int8_t, int16_t, int32_t, int64_t)
 * @return 0 on success, non-zero on failure
 */
int parsec_convert_i2d(parsec_context_t *parsec,
                         parsec_tiled_matrix_t *A, int intype);

/**
 * @brief Convert single precision matrix to integer format
 * 
 * Converts a single precision matrix to the specified integer format.
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution
 * @param[in] A Tiled matrix to convert
 * @param[in] intype Target integer type (e.g., 8, 16, 32, 64 for int8_t, int16_t, int32_t, int64_t)
 * @return 0 on success, non-zero on failure
 */
int parsec_convert_s2i(parsec_context_t *parsec,
                         parsec_tiled_matrix_t *A, int intype);

/**
 * @brief Convert single precision matrix to double precision (for correctness checking)
 * 
 * Converts a single precision matrix to double precision format.
 * This function is primarily used for correctness checking and validation.
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution
 * @param[in] data HICMA PaRSEC data structure containing matrix information
 * @param[in] params_tlr HICMA PaRSEC parameters for the conversion
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_convert_s2d(parsec_context_t *parsec,
                         hicma_parsec_data_t *data,
                         hicma_parsec_params_t *params_tlr);

/**
 * @brief Create a new dense static datatype conversion taskpool
 * 
 * **Supported Conversion Types:**
 * - "d2s": Double precision to single precision
 * - "s2d": Single precision to double precision
 * - "d2i8": Double precision to 8-bit integer
 * - "s2i8": Single precision to 8-bit integer
 * - "i8d": 8-bit integer to double precision
 * - "i8s": 8-bit integer to single precision
 * - "d2h": Double precision to half precision
 * - "h2d": Half precision to double precision
 * - "s2h": Single precision to half precision
 * - "h2s": Half precision to single precision
 * 
 * **Error Handling:**
 * - Returns NULL on failure (invalid parameters, unsupported conversion type)
 * - Returns valid taskpool pointer on success
 * - Common failure modes:
 *   * Invalid matrix descriptor
 *   * Unsupported conversion type
 *   * Memory allocation failure
 *   * Invalid matrix region specification
 * 
 * **Usage Example:**
 * @code
 * // Create conversion taskpool
 * parsec_taskpool_t *convert_tp = parsec_datatype_convert_dense_static_New(
 *     dplasmaGeneral,           // Use entire matrix
 *     matrix_A,                 // Input matrix
 *     "d2s"                     // Conversion type
 * );
 * 
 * if (convert_tp != NULL) {
 *     // Submit taskpool for execution
 *     parsec_context_add_taskpool(parsec, convert_tp);
 *     parsec_context_wait(parsec);
 *     
 *     // Clean up
 *     parsec_datatype_convert_dense_static_Destruct(convert_tp);
 * }
 * @endcode
 * 
 * @param[in] uplo Specifies which part of the matrix to use:
 *                 - dplasmaUpper: Use upper triangular part (for symmetric matrices)
 *                 - dplasmaLower: Use lower triangular part (for symmetric matrices)
 *                 - dplasmaGeneral: Use entire matrix (default for general matrices)
 * @param[in] A Tiled matrix to convert. Must be a valid, initialized PaRSEC
 *              tiled matrix descriptor. The matrix dimensions and tile sizes
 *              are automatically determined from this descriptor.
 * @param[in] convert_type String specifying the conversion type. Must be one of
 *                         the supported conversion strings listed above.
 *                         The string is case-sensitive and must be null-terminated.
 * @return Pointer to the created taskpool on success, NULL on failure.
 *         The returned taskpool must be submitted to the PaRSEC context
 *         for execution and later destroyed using the Destruct function.
 * 
 * @see parsec_datatype_convert_dense_static() for blocking conversion execution
 * @see parsec_datatype_convert_dense_static_Destruct() for taskpool cleanup
 * @see parsec_context_add_taskpool() for submitting taskpools to execution
 * @see parsec_tiled_matrix_t for matrix descriptor structure
 * @see dplasma_enum_t for matrix region specification constants
 */
parsec_taskpool_t* parsec_datatype_convert_dense_static_New( dplasma_enum_t uplo,
        parsec_tiled_matrix_t *A,
        char *convert_type);

/**
 * @brief Destroy a dense static datatype conversion taskpool
 * 
 * @param[in] taskpool Taskpool to destroy. Can be NULL (function does nothing).
 *                    After this call, the taskpool handle becomes invalid
 *                    and should not be used for any operations.
 * 
 * @note This function should be called after the taskpool has completed
 *       execution to ensure proper cleanup of all resources.
 * 
 * @note The function is idempotent - calling it multiple times on the
 *       same taskpool is safe, though unnecessary.
 * 
 * @note After calling this function, the taskpool handle becomes invalid
 *       and should be set to NULL to prevent accidental use.
 * 
 * @warning Do not call this function on a taskpool that is currently
 *          executing or has pending tasks, as this may lead to undefined
 *          behavior or memory corruption.
 * 
 * @see parsec_datatype_convert_dense_static_New() for creating conversion taskpools
 * @see parsec_datatype_convert_dense_static() for blocking conversion execution
 * @see parsec_context_add_taskpool() for submitting taskpools to execution
 * @see parsec_context_wait() for waiting for taskpool completion
 */
void parsec_datatype_convert_dense_static_Destruct(parsec_taskpool_t *taskpool);

/**
 * @brief Execute dense static datatype conversion
 * 
 * **Supported Conversion Types:**
 * - "d2s": Double precision to single precision (may lose precision)
 * - "s2d": Single precision to double precision (maintains precision)
 * - "d2i8": Double precision to 8-bit integer (truncation occurs)
 * - "s2i8": Single precision to 8-bit integer (truncation occurs)
 * - "i8d": 8-bit integer to double precision (maintains precision)
 * - "i8s": 8-bit integer to single precision (maintains precision)
 * - "d2h": Double precision to half precision (may lose precision)
 * - "h2d": Half precision to double precision (maintains precision)
 * - "s2h": Single precision to half precision (may lose precision)
 * - "h2s": Half precision to single precision (maintains precision)
 * 
 * **Error Handling:**
 * - Returns 0 on successful completion
 * - Returns non-zero value on failure
 * - Common failure modes include:
 *   * Invalid conversion type string
 *   * Matrix dimension mismatches
 *   * Memory allocation failures
 *   * PaRSEC runtime errors
 * 
 * **Usage Example:**
 * @code
 * // Convert double precision matrix to single precision
 * int result = parsec_datatype_convert_dense_static(
 *     parsec,                    // PaRSEC context
 *     dplasmaGeneral,           // Use entire matrix
 *     matrix_A,                 // Input matrix
 *     "d2s"                     // Conversion type
 * );
 * if (result != 0) {
 *     // Handle conversion error
 * }
 * @endcode
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution.
 *                    Must be a valid, initialized PaRSEC context.
 * @param[in] uplo Specifies which part of the matrix to use:
 *                 - dplasmaUpper: Use upper triangular part (for symmetric matrices)
 *                 - dplasmaLower: Use lower triangular part (for symmetric matrices)
 *                 - dplasmaGeneral: Use entire matrix (default for general matrices)
 * @param[in] A Tiled matrix to convert. Must be a valid, initialized PaRSEC
 *              tiled matrix descriptor. The matrix dimensions and tile sizes
 *              are automatically determined from this descriptor.
 * @param[in] convert_type String specifying the conversion type. Must be one of
 *                         the supported conversion strings listed above.
 *                         The string is case-sensitive and must be null-terminated.
 * @return 0 on successful completion of all conversion tasks, non-zero on failure.
 *         The return value indicates the overall success/failure status, not
 *         the number of tiles processed.
 * 
 * @note This function is blocking and will not return until all conversion
 *       tasks have completed or failed. For non-blocking operation, use the
 *       taskpool-based approach with parsec_datatype_convert_dense_static_New().
 * 
 * @see parsec_datatype_convert_dense_static_New() for creating conversion taskpools
 * @see parsec_datatype_convert_dense_static_Destruct() for cleaning up taskpools
 * @see parsec_tiled_matrix_t for matrix descriptor structure
 * @see dplasma_enum_t for matrix region specification constants
 */
int parsec_datatype_convert_dense_static(parsec_context_t *parsec,
        dplasma_enum_t uplo,
        parsec_tiled_matrix_t *A,
        char *convert_type);

/**
 * @brief Copy between a tiled matrix and the local column-major ("LAPACK")
 *        buffer of the calling process
 *
 * Collective over the processes of dcA. The buffer holds the tiles this
 * process owns, tile (m, n) at local tile position (m / P, n / Q), column-major
 * with leading dimension dcA->llm (local rows of a 2D block-cyclic descriptor,
 * global rows of a symmetric one); on a 1 x 1 grid it is the whole matrix in
 * LAPACK layout. The named functions below are this one with fixed arguments.
 *
 * @param[in]    parsec      PaRSEC context
 * @param[inout] dcA         tiled matrix, 2D block-cyclic or symmetric block-cyclic over P x Q
 * @param[inout] lapack      buffer of at least dcA->llm x dcA->lln elements of @p type
 * @param[in]    P           process grid rows of dcA
 * @param[in]    Q           process grid columns of dcA
 * @param[in]    type        element type of the tiles and of the buffer
 *                           (parsec_datatype_double_t, _float_t, _int_t, _int8_t, ...)
 * @param[in]    lower       1: only the tiles on and below the diagonal
 * @param[in]    into_tiles  1: buffer -> tiles, 0: tiles -> buffer
 * @return 0 on success; -1 on every process when the arguments or the
 *         distribution do not fit the layout (nothing is copied)
 */
int hicma_parsec_lapack_tile_convert(parsec_context_t *parsec,
        parsec_tiled_matrix_t *dcA, void *lapack, int P, int Q,
        parsec_datatype_t type, int lower, int into_tiles);

/**
 * @brief Convert PaRSEC tiled matrix to LAPACK double precision format
 * 
 * Converts a PaRSEC tiled matrix to LAPACK-compatible double precision format.
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution
 * @param[in] dcA Source tiled matrix
 * @param[out] Zobs Output LAPACK matrix in double precision
 * @param[in] P Number of processes in row dimension
 * @param[in] Q Number of processes in column dimension
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_Tile_to_Lapack(parsec_context_t *parsec,
                parsec_tiled_matrix_t *dcA, double *Zobs, int P, int Q);

/**
 * @brief Convert PaRSEC tiled matrix to LAPACK single precision format
 * 
 * Converts a PaRSEC tiled matrix to LAPACK-compatible single precision format.
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution
 * @param[in] dcA Source tiled matrix
 * @param[out] Zobs Output LAPACK matrix in single precision
 * @param[in] P Number of processes in row dimension
 * @param[in] Q Number of processes in column dimension
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_Tile_to_Lapack_single(parsec_context_t *parsec,
                parsec_tiled_matrix_t *dcA, float *Zobs, int P, int Q);

/**
 * @brief Convert PaRSEC tiled matrix to LAPACK int8 format
 * 
 * Converts a PaRSEC tiled matrix to LAPACK-compatible 8-bit integer format.
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution
 * @param[in] dcA Source tiled matrix
 * @param[out] Zobs Output LAPACK matrix in int8 format
 * @param[in] P Number of processes in row dimension
 * @param[in] Q Number of processes in column dimension
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_Tile_to_Lapack_int8(parsec_context_t *parsec,
                parsec_tiled_matrix_t *dcA, int8_t *Zobs, int P, int Q);

/**
 * @brief Convert PaRSEC tiled matrix to LAPACK integer format
 * 
 * Converts the tiles on and below the diagonal of a PaRSEC tiled matrix to
 * LAPACK-compatible integer format (the tiles above are not copied).
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution
 * @param[in] dcA Source tiled matrix
 * @param[out] Zobs Output LAPACK matrix in integer format
 * @param[in] P Number of processes in row dimension
 * @param[in] Q Number of processes in column dimension
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_Tile_to_Lapack_int(parsec_context_t *parsec,
                parsec_tiled_matrix_t *dcA, int *Zobs, int P, int Q);

/**
 * @brief Convert PaRSEC tiled symmetric matrix to LAPACK double precision format
 * 
 * Converts a PaRSEC tiled symmetric matrix to LAPACK-compatible double precision format.
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution
 * @param[in] dcA Source tiled symmetric matrix
 * @param[out] Zobs Output LAPACK matrix in double precision
 * @param[in] P Number of processes in row dimension
 * @param[in] Q Number of processes in column dimension
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_Tile_to_Lapack_sym(parsec_context_t *parsec,
                parsec_tiled_matrix_t *dcA, double *Zobs, int P, int Q);

/**
 * @brief Convert PaRSEC tiled symmetric matrix to LAPACK single precision format
 * 
 * Converts a PaRSEC tiled symmetric matrix to LAPACK-compatible single precision format.
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution
 * @param[in] dcA Source tiled symmetric matrix
 * @param[out] Zobs Output LAPACK matrix in single precision
 * @param[in] P Number of processes in row dimension
 * @param[in] Q Number of processes in column dimension
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_Tile_to_Lapack_sym_single(parsec_context_t *parsec,
                parsec_tiled_matrix_t *dcA, float *Zobs, int P, int Q);

/**
 * @brief Convert LAPACK double precision matrix to PaRSEC tiled format
 * 
 * Converts a LAPACK-compatible double precision matrix to PaRSEC tiled format.
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution
 * @param[in] dcA Output tiled matrix
 * @param[in] Zobs Input LAPACK matrix in double precision
 * @param[in] P Number of processes in row dimension
 * @param[in] Q Number of processes in column dimension
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_Lapack_to_Tile(parsec_context_t *parsec,
		parsec_tiled_matrix_t *dcA, double *Zobs, int P, int Q);

/**
 * @brief Convert LAPACK single precision matrix to PaRSEC tiled format
 * 
 * Converts a LAPACK-compatible single precision matrix to PaRSEC tiled format.
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution
 * @param[in] dcA Output tiled matrix
 * @param[in] Zobs Input LAPACK matrix in single precision
 * @param[in] P Number of processes in row dimension
 * @param[in] Q Number of processes in column dimension
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_Lapack_to_Tile_Single(parsec_context_t *parsec,
		parsec_tiled_matrix_t *dcA, float *Zobs, int P, int Q);

/**
 * @brief Convert LAPACK symmetric double precision matrix to PaRSEC tiled format
 * 
 * Converts a LAPACK-compatible symmetric double precision matrix to PaRSEC tiled format.
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution
 * @param[in] dcA Output tiled symmetric matrix
 * @param[in] Zobs Input LAPACK symmetric matrix in double precision
 * @param[in] P Number of processes in row dimension
 * @param[in] Q Number of processes in column dimension
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_Lapack_to_Tile_sym(parsec_context_t *parsec,
		parsec_tiled_matrix_t *dcA, double *Zobs, int P, int Q);

/* ============================================================================
   Element conversions: OmniCast

   Every conversion between element types (FP64, FP32, FP16, BF16, FP8 and
   integers), on the CPU and on the GPU, goes through OmniCast (submodule
   omnicast/, header omnicast/omnicast.h): SIMD kernels chosen at run time on
   the CPU, vectorized CUDA/HIP kernels on the GPU, single-pass in-place
   conversion, transposed/padded FP8 operands, and stochastic rounding.

   Call sites use the OmniCast API directly, for instance

       HICMA_OMC( omc_gpu_convert_2d(OMC_FP64, A, lda, OMC_FP16, A_h, lda,
                                     m, n, HICMA_OMC_SR(sr_on, seed), stream) );

   The helpers below map HiCMA-X tile precisions to OmniCast types and carry
   the HiCMA-X conventions (seed per tile, FP8 operand layout of cuBLASLt).
   ============================================================================ */

/** Abort with the call site when an OmniCast call fails: a failure means a
 *  wrong argument or a dead GPU context, not a condition to recover from. */
#define HICMA_OMC(call) do {                                                       \
        omc_status_t hicma_omc_status_ = (call);                                   \
        if( OMC_SUCCESS != hicma_omc_status_ ) {                                   \
            fprintf(stderr, "%s:%d: OmniCast: %s\n", __FILE__, __LINE__,           \
                    omc_status_string(hicma_omc_status_));                         \
            abort();                                                               \
        }                                                                          \
    } while(0)

/** OmniCast options for stochastic rounding with the given seed when `on`,
 *  NULL (round to nearest even) otherwise. C only (compound literal). */
#define HICMA_OMC_SR(on_, seed_) \
    ((on_) ? &(omc_options_t){ .rounding = OMC_ROUND_STOCHASTIC, .seed = (seed_) } \
          : (const omc_options_t *)NULL)

/** FP32 -> FP16 (round to nearest even) of a host scalar, stored at h (2 bytes):
 *  the alpha/beta of FP16-compute cuBLAS calls */
static inline void hicma_parsec_fp32_to_fp16_host(float x, void *h)
{
    const uint16_t b = omc_fp32_to_fp16(x);
    memcpy(h, &b, sizeof(b));
}

/** OmniCast options that multiply by `s` before converting. C only. */
#define HICMA_OMC_SCALED(s_) (&(omc_options_t){ .scale_mode = OMC_SCALE_HOST, .scale = (s_) })

/**
 * @brief OmniCast element type of a HiCMA-X tile precision
 *
 * DENSE_DP and LOW_RANK_DP: OMC_FP64; DENSE_SP and LOW_RANK_SP: OMC_FP32;
 * DENSE_HP: OMC_FP16; DENSE_FP8: OMC_FP8_E4M3. Exits on anything else.
 */
omc_dtype_t hicma_parsec_omc_dtype(uint16_t precision);

/**
 * @brief Unified datatype conversion function (unary, in place)
 *
 * Converts the mb-by-nb matrix A (leading dimension lda, the same in the
 * elements of both types) in place, the conversion given by a type string:
 * "d2s", "s2d", "d2h", "h2d", "s2h", "h2s", "d2i8", "d2i16", "d2i",
 * "s2i8", "s2i16", "s2i", "i82s", "i82d", "i162s", "i162d", "i2s", "i2d"
 * (case-insensitive). The float -> integer conversions truncate and saturate.
 *
 * @param[in,out] A    The data, converted in place
 * @param[in]     mb   Number of rows
 * @param[in]     nb   Number of columns
 * @param[in]     lda  Leading dimension
 * @param[in]     type Conversion type string
 * @param[out]    size Set to mb * nb * (size of the target type) for "d2s"
 * @return 0 on success, -1 on an unknown type or a NULL argument
 */
int convert_datatype_unary_CPU(void *A, int mb, int nb, int lda, char *type, size_t *size);

/**
 * @brief Unified datatype conversion function (binary)
 *
 * Same conversions as convert_datatype_unary_CPU, from source into target
 * (both with leading dimension lda).
 *
 * @return 0 on success, -1 on an unknown type or a NULL argument
 */
int convert_datatype_binary_CPU(void *target, void *source, int mb, int nb, int lda, char *type);

/* ============================================================================
   Utility Functions
   
   These functions provide essential utilities for datatype conversion operations,
   including size queries, array initialization, debugging support, and specialized
   conversion helpers. They are designed to support the main conversion functions
   and provide additional functionality for application developers.
   ============================================================================ */

/**
 * @brief Get the byte size of a datatype based on input string
 * 
 * This function returns the size in bytes of various datatypes based on the input string.
 * It is useful for memory allocation and buffer size calculations.
 * 
 * **Supported Datatypes:**
 * - "double", "d" -> sizeof(double) (typically 8 bytes)
 * - "float", "single", "s" -> sizeof(float) (typically 4 bytes)
 * - "int8", "i8" -> sizeof(int8_t) (1 byte)
 * - "int16", "i16" -> sizeof(int16_t) (2 bytes)
 * - "int32", "int", "i32", "i" -> sizeof(int) (typically 4 bytes)
 * - "int64", "i64" -> sizeof(int64_t) (8 bytes)
 * - "uint8", "u8" -> sizeof(uint8_t) (1 byte)
 * - "uint16", "u16" -> sizeof(uint16_t) (2 bytes)
 * - "uint32", "uint", "u32", "u" -> sizeof(unsigned int) (typically 4 bytes)
 * - "uint64", "u64" -> sizeof(uint64_t) (8 bytes)
 * - "half", "fp16", "h" -> sizeof(__fp16) (2 bytes, if available)
 * - "fp8" -> sizeof(uint8_t) (1 byte, for FP8 representation)
 * - "fp4" -> sizeof(uint8_t) (1 byte, for FP4 representation, 2 values per byte)
 * - "int4" -> sizeof(uint8_t) (1 byte, for INT4 representation, 2 values per byte)
 * - "1bit" -> sizeof(uint8_t) (1 byte, for 1-bit representation, 8 values per byte)
 * 
 * @param[in] datatype_str String representation of the datatype (case-insensitive)
 * 
 * @return Size in bytes of the datatype, or 0 if datatype is not recognized
 * 
 * @note The function is case-insensitive for better usability
 * @note Returns 0 for unsupported datatypes to indicate error
 * @note Useful for dynamic memory allocation based on datatype requirements
 * 
 * @see hicma_parsec_init_array_to_zero() for array initialization
 * @see hicma_parsec_print_array_values() for array printing
 */
size_t get_datatype_size(const char *datatype_str);

/**
 * @brief Initialize an array to zero values based on datatype
 * 
 * This function initializes a 2D array of mb * nb elements with leading dimension lda
 * to zero values based on the specified datatype. The function supports both row-major
 * and column-major memory layouts. Column-major is the default layout.
 * 
 * **Supported Datatypes:**
 * - "double", "d" -> double precision floating point (0.0)
 * - "float", "single", "s" -> single precision floating point (0.0f)
 * - "int8", "i8" -> 8-bit signed integer (0)
 * - "int16", "i16" -> 16-bit signed integer (0)
 * - "int32", "int", "i32", "i" -> 32-bit signed integer (0)
 * - "int64", "i64" -> 64-bit signed integer (0)
 * - "uint8", "u8" -> 8-bit unsigned integer (0)
 * - "uint16", "u16" -> 16-bit unsigned integer (0)
 * - "uint32", "uint", "u32", "u" -> 32-bit unsigned integer (0)
 * - "uint64", "u64" -> 64-bit unsigned integer (0)
 * - "half", "fp16", "h" -> 16-bit floating point (0.0, if available)
 * - "fp8" -> 8-bit floating point representation (0)
 * - "fp4" -> 4-bit floating point representation (0)
 * - "int4" -> 4-bit integer representation (0)
 * - "1bit" -> 1-bit representation (0)
 * 
 * @param[out] array Pointer to the array to initialize
 * @param[in] mb Number of rows in the matrix
 * @param[in] nb Number of columns in the matrix
 * @param[in] lda Leading dimension of the matrix
 * @param[in] datatype_str String representation of the datatype (case-insensitive)
 * @param[in] is_row_major 1 for row-major layout, 0 for column-major layout (default: 0)
 * 
 * @return 0 on successful initialization, -1 on error
 * 
 * @note The function is case-insensitive for better usability
 * @note Memory layout affects how the array is traversed during initialization
 * @note Useful for initializing matrices before computation
 * @note Error occurs if datatype is not recognized or parameters are invalid
 * 
 * @see get_datatype_size() for datatype size information
 * @see hicma_parsec_print_array_values() for array printing
 */
int hicma_parsec_init_array_to_zero(void *array, int mb, int nb, int lda, const char *datatype_str, int is_row_major);

/**
 * @brief Print values in a 2D array based on datatype
 * 
 * This function prints the values of a 2D array of mb * nb elements with leading dimension lda
 * in a formatted matrix layout (mb rows and nb columns) based on the specified datatype.
 * The function supports both row-major and column-major memory layouts. Column-major is the default layout.
 * 
 * **Supported Datatypes:**
 * - "double", "d" -> double precision floating point
 * - "float", "single", "s" -> single precision floating point
 * - "int8", "i8" -> 8-bit signed integer
 * - "int16", "i16" -> 16-bit signed integer
 * - "int32", "int", "i32", "i" -> 32-bit signed integer
 * - "int64", "i64" -> 64-bit signed integer
 * - "uint8", "u8" -> 8-bit unsigned integer
 * - "uint16", "u16" -> 16-bit unsigned integer
 * - "uint32", "uint", "u32", "u" -> 32-bit unsigned integer
 * - "uint64", "u64" -> 64-bit unsigned integer
 * - "half", "fp16", "h" -> 16-bit floating point (half precision)
 * - "fp8" -> 8-bit floating point representation
 * - "fp4" -> 4-bit floating point representation
 * - "int4" -> 4-bit integer representation
 * - "1bit" -> 1-bit representation
 * 
 * @param[in] array Pointer to the array to print
 * @param[in] mb Number of rows in the matrix
 * @param[in] nb Number of columns in the matrix
 * @param[in] lda Leading dimension of the matrix
 * @param[in] datatype_str String representation of the datatype (case-insensitive)
 * @param[in] array_name Optional name for the array (can be NULL)
 * @param[in] is_row_major 1 for row-major layout, 0 for column-major layout (default: 0)
 * 
 * @return 0 on successful printing, -1 on error
 * 
 * @note The function is case-insensitive for better usability
 * @note Memory layout affects how the array is traversed during printing
 * @note Useful for debugging and verification of matrix contents
 * @note Error occurs if datatype is not recognized or parameters are invalid
 * @note Output format is optimized for readability with proper spacing and alignment
 * 
 * @see get_datatype_size() for datatype size information
 * @see hicma_parsec_init_array_to_zero() for array initialization
 */
int hicma_parsec_print_array_values(const void *array, int mb, int nb, int lda, const char *datatype_str, const char *array_name, int is_row_major);

/**
 * @brief Check whether to convert datatype in TRSM operations
 * 
 * Determines if datatype conversion is needed for the specified matrix element
 * during triangular solve (TRSM) operations based on the current parameters.
 * This function is used to optimize TRSM operations by avoiding unnecessary conversions.
 * 
 * @param[in] params_tlr HICMA PaRSEC parameters structure containing conversion settings
 * @param[in] m Row index of the matrix element
 * @param[in] n Column index of the matrix element
 * 
 * @return true if conversion is needed, false otherwise
 * 
 * @note This function is used internally by TRSM algorithms
 * @note The decision is based on the current conversion parameters and matrix structure
 * @note Useful for performance optimization in triangular solve operations
 * 
 * @see hicma_parsec_params_t for parameter structure details
 */
bool hicma_parsec_convert_in_trsm(hicma_parsec_params_t *params_tlr, int m, int n);

/**
 * @brief The values of tile A rounded to FP16 and back, in FP32
 *
 * A_use[i + j * mb] = (float)(fp16)A(i, j) for the mb-by-nb tile A, which is
 * FP64 when decision is DENSE_DP or LOW_RANK_DP and FP32 otherwise: the
 * rounding error of an FP16 copy, for the accuracy checks.
 *
 * @param[in]  A        Input tile (contiguous)
 * @param[out] A_use    mb * nb floats
 * @param[in]  mb       Number of rows
 * @param[in]  nb       Number of columns
 * @param[in]  decision Precision of A
 */
void hicma_parsec_convert_2h_bit(void *A, float *A_use, int mb, int nb, uint16_t decision);

/**
 * @brief Convert a tile to (convert_direction 0) or from (1) the precision
 *        `decision`, in place, on the CPU
 *
 * Direction 0 converts an FP64 tile to decision (DENSE_SP, DENSE_HP), with
 * stochastic rounding when enable_stochastic_rounding (FP32 only, as before);
 * direction 1 converts it back to FP64. DENSE_DP does nothing. Exits on a
 * precision without a CPU conversion.
 *
 * @return 0 on success, -1 on a NULL argument
 */
int convert_datatype_adaptive_unary_CPU(void *A, int mb, int nb, int lda, uint16_t decision, int convert_direction, int enable_stochastic_rounding, uint64_t sr_seed, size_t *size);

/**
 * @brief Seed of the stochastic rounding of the elements of tile (m, n)
 *
 * Stochastic rounding (OmniCast, OMC_ROUND_STOCHASTIC) takes the seed of the
 * tile its input belongs to (HICMA_OMC_SR(on, hicma_parsec_sr_seed(m, n))). The random number of each element is a hash of its
 * position in the tile, the seed and its value, so that the elements of
 * different tiles round independently, even where their values are equal,
 * and the results are reproducible.
 */
static inline uint64_t hicma_parsec_sr_seed(int m, int n)
{
    uint64_t x = (uint64_t)(uint32_t)m | ((uint64_t)(uint32_t)n << 32);
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

/**
 * @brief Perform datatype conversion on a dense matrix
 *
 * @param [in] parsec: The PaRSEC context
 * @param [in] uplo: Upper or lower triangular part to convert
 * @param [inout] A: The matrix to convert, already distributed and allocated
 * @param [in] decisions: Type of conversion to perform
 * @param [in] convert_direction: 0: high->low; 1: low->high
 * @return 0 on success
 */
int parsec_datatype_convert_dense_adaptive(parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        dplasma_enum_t uplo,
        parsec_tiled_matrix_t *A,
        uint16_t *decisions,
        int convert_direction);


#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
/**
 * @brief GPU version of convert_datatype_adaptive_unary_CPU: one single-pass
 *        in-place OmniCast kernel on the stream (FP32 and FP16, with
 *        stochastic rounding when enabled)
 */
int convert_datatype_adaptive_unary_GPU(void *A, int mb, int nb, int lda,
        uint16_t decision, int convert_direction, int enable_stochastic_rounding, uint64_t sr_seed,
        size_t *size, cudaStream_t stream);

/**
 * @brief FP8 (E4M3) operand of a cuBLASLt matmul from a tile
 *
 * T (leading dimension ldt) receives the transpose of the nrows-by-ncols tile
 * S (leading dimension lds, element type src_type), the layout cuBLASLt takes
 * for FP8 (TN), zero-padded to padded_cols rows and padded_rows columns and
 * saturated. With
 * scaling, S is first multiplied by the largest power of two s with
 * s * max|S| <= 448, and scale[0..2] (device) receives {max|S|, s, 1/s};
 * without, scale[0..2] = {0, 1, 1}. cuBLASLt reads 1/s from scale[2].
 */
void hicma_parsec_fp8_operand_GPU(omc_dtype_t src_type, int nrows, int ncols,
        const void *S, int lds, void *T, int ldt, int padded_rows, int padded_cols,
        float *scale, int scaling, cudaStream_t stream);
#endif


#ifdef __cplusplus
}
#endif

#endif /* HICMA_PARSEC_DATATYPE_CONVERT_H */ 

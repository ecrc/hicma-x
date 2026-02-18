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
 * @version 1.0.0
 * @see hicma_parsec_internal.h for internal HICMA PaRSEC definitions
 */

#ifndef HICMA_PARSEC_DATATYPE_CONVERT_H
#define HICMA_PARSEC_DATATYPE_CONVERT_H

#include "hicma_parsec_internal.h"


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
 * @brief Convert integer matrix to single precision format
 * 
 * Converts an integer matrix to single precision format.
 * 
 * @param[in] parsec PaRSEC context for task scheduling and execution
 * @param[in] A Tiled matrix to convert
 * @param[in] intype Source integer type (e.g., 8, 16, 32, 64 for int8_t, int16_t, int32_t, int64_t)
 * @return 0 on success, non-zero on failure
 */
int parsec_convert_i2s(parsec_context_t *parsec,
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
 * Converts a PaRSEC tiled matrix to LAPACK-compatible integer format.
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
   Unified Conversion Functions
   
   These functions provide a unified interface for datatype conversions using
   type string parameters. They automatically dispatch to the appropriate
   conversion function based on the conversion type string, simplifying the
   API for applications that need to perform various conversions dynamically.
   ============================================================================ */

/**
 * @brief Unified datatype conversion function (unary version)
 * 
 * This function provides a unified interface for in-place datatype conversions
 * based on a type string parameter. It automatically dispatches to the appropriate
 * conversion function and handles memory management for the conversion process.
 * 
 * **Supported Conversion Types:**
 * - "d2s": Double precision to single precision
 * - "s2d": Single precision to double precision  
 * - "d2i8": Double precision to 8-bit integer
 * - "d2i16": Double precision to 16-bit integer
 * - "d2i32": Double precision to 32-bit integer
 * - "d2i64": Double precision to 64-bit integer
 * - "s2i8": Single precision to 8-bit integer
 * - "s2i16": Single precision to 16-bit integer
 * - "s2i32": Single precision to 32-bit integer
 * - "s2i64": Single precision to 64-bit integer
 * - "i8d": 8-bit integer to double precision
 * - "i16d": 16-bit integer to double precision
 * - "i32d": 32-bit integer to double precision
 * - "i64d": 64-bit integer to double precision
 * - "i8s": 8-bit integer to single precision
 * - "i16s": 16-bit integer to single precision
 * - "i32s": 32-bit integer to single precision
 * - "i64s": 64-bit integer to single precision
 * - "d2h": Double precision to half precision (if available)
 * - "h2d": Half precision to double precision (if available)
 * - "s2h": Single precision to half precision (if available)
 * - "h2s": Half precision to single precision (if available)
 * 
 * **Memory Management:**
 * - Performs in-place conversion when possible to minimize memory usage
 * - May reallocate memory if the target type requires different storage size
 * - Updates the size parameter to reflect the new data size in bytes
 * 
 * **Error Handling:**
 * - Returns 0 on successful conversion
 * - Returns -1 on error (invalid type string, memory allocation failure, etc.)
 * - Provides detailed error information through standard error reporting
 * 
 * @param[in,out] A Pointer to the data buffer to be converted.
 *                  The buffer is modified in-place to contain the converted data.
 * @param[in] mb Number of rows in the matrix
 * @param[in] nb Number of columns in the matrix
 * @param[in] lda Leading dimension of the matrix (stride between rows)
 * @param[in] type Conversion type string specifying source and target types.
 *                 Must be one of the supported conversion strings listed above.
 * @param[out] size Pointer to store the output size in bytes after conversion.
 *                  This helps with memory management and buffer size calculations.
 * 
 * @return 0 on success, -1 on error
 * 
 * @note The conversion type string is case-sensitive
 * @note Memory may be reallocated if the target type has different size requirements
 * @note The function is optimized for CPU execution
 * 
 * @see convert_datatype_binary_CPU() for binary conversion with separate buffers
 * @see get_datatype_size() for getting datatype sizes
 */
int convert_datatype_unary_CPU(void *A, int mb, int nb, int lda, char *type, size_t *size);

/**
 * @brief Unified datatype conversion function (binary version)
 * 
 * This function provides a unified interface for datatype conversions using separate
 * source and target buffers. It automatically dispatches to the appropriate conversion
 * function based on the type string parameter, preserving the original data.
 * 
 * **Supported Conversion Types:**
 * Same as convert_datatype_unary_CPU() - see that function for the complete list.
 * 
 * **Memory Management:**
 * - Uses separate source and target buffers to preserve original data
 * - Target buffer must be pre-allocated with sufficient size for the converted data
 * - No memory reallocation is performed during conversion
 * - Source buffer remains unchanged after conversion
 * 
 * **Performance Benefits:**
 * - Preserves original data for comparison or rollback operations
 * - Enables parallel processing of source and target data
 * - Reduces memory fragmentation compared to in-place conversion
 * - Allows for streaming conversions of large datasets
 * 
 * **Error Handling:**
 * - Returns 0 on successful conversion
 * - Returns -1 on error (invalid type string, buffer size mismatch, etc.)
 * - Validates buffer sizes before performing conversion
 * 
 * @param[out] target Pointer to the target buffer for converted data.
 *                    Must be pre-allocated with sufficient size for the target type.
 * @param[in] source Pointer to the source buffer containing data to convert.
 *                   This buffer remains unchanged after conversion.
 * @param[in] mb Number of rows in the matrix
 * @param[in] nb Number of columns in the matrix
 * @param[in] lda Leading dimension of the matrix (stride between rows)
 * @param[in] type Conversion type string specifying source and target types.
 *                 Must be one of the supported conversion strings.
 * 
 * @return 0 on success, -1 on error
 * 
 * @note The conversion type string is case-sensitive
 * @note Target buffer must be pre-allocated with correct size
 * @note Source buffer is not modified during conversion
 * @note Function is optimized for CPU execution
 * 
 * @see convert_datatype_unary_CPU() for in-place conversion
 * @see get_datatype_size() for calculating required buffer sizes
 */
int convert_datatype_binary_CPU(void *target, void *source, int mb, int nb, int lda, char *type);

/* ============================================================================
   Individual Conversion Functions - Floating Point
   
   These functions provide optimized implementations for specific floating-point
   datatype conversions. They are called by the unified conversion functions
   and can also be used directly for performance-critical applications that
   know the exact conversion type at compile time.
   ============================================================================ */

// Single to Double precision conversions
/**
 * @brief Convert single precision to double precision (unary version)
 * 
 * This function performs an in-place conversion of a single precision matrix
 * to double precision format. The conversion expands each 32-bit float value
 * to a 64-bit double value, potentially increasing memory usage by a factor of 2.
 * 
 * **Conversion Process:**
 * - Each single precision value is converted to double precision
 * - Precision is maintained (no loss of information)
 * - Memory is reallocated to accommodate the larger data type
 * - Matrix structure and metadata are preserved
 * 
 * **Memory Considerations:**
 * - Memory usage doubles after conversion
 * - Original single precision data is overwritten
 * - Matrix dimensions remain unchanged
 * - Leading dimension is adjusted for the new data type
 * 
 * **Performance Notes:**
 * - Optimized for CPU execution with vectorized operations
 * - Memory bandwidth intensive due to data expansion
 * - Suitable for applications requiring higher precision
 * 
 * @param[in,out] data Pointer to the matrix data to convert.
 *                     The buffer is modified in-place to contain double precision data.
 * @param[in] mb Number of rows in the matrix
 * @param[in] nb Number of columns in the matrix
 * 
 * @note This function modifies the input data in-place
 * @note Memory usage will increase by approximately 2x
 * @note No precision is lost during conversion
 * @note Function is optimized for CPU execution
 * 
 * @see convert_s2d_binary_CPU() for binary conversion preserving original data
 * @see convert_d2s_unary_CPU() for reverse conversion
 */
void convert_s2d_unary_CPU(float *data, int mb, int nb);

/**
 * @brief Convert single precision to double precision (binary version)
 * 
 * Converts a single precision matrix to double precision with separate
 * source and target buffers.
 * 
 * @param[out] _target Pointer to the target double precision buffer
 * @param[in] _source Pointer to the source single precision buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 */
void convert_s2d_binary_CPU(double *_target, float *_source, int mb, int nb);

/**
 * @brief Convert double precision to single precision (unary version)
 * 
 * Converts a double precision matrix to single precision in-place.
 * 
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 */
void convert_d2s_unary_CPU(double *data, int mb, int nb);

/**
 * @brief Convert double precision to single precision (binary version)
 * 
 * Converts a double precision matrix to single precision with separate
 * source and target buffers.
 * 
 * @param[out] _target Pointer to the target single precision buffer
 * @param[in] _source Pointer to the source double precision buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 */
void convert_d2s_binary_CPU(float *_target, double *_source, int mb, int nb);

/* ============================================================================
   Individual Conversion Functions - Half Precision (if available)
   
   These functions provide optimized implementations for half precision (16-bit)
   floating-point conversions. Half precision support is conditional and depends
   on hardware and compiler support for the __fp16 type. These functions are
   only available when HAVE_HP_CPU is defined.
   ============================================================================ */

#if HAVE_HP_CPU
/**
 * @brief Convert double precision to half precision (binary version)
 * 
 * Converts a double precision matrix to half precision with separate
 * source and target buffers.
 * 
 * @param[out] _target Pointer to the target half precision buffer
 * @param[in] _source Pointer to the source double precision buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 */
void convert_d2h_binary_CPU(__fp16 *_target, double *_source, int mb, int nb);

/**
 * @brief Convert double precision to half precision (unary version)
 *
 * Converts a double precision matrix to half precision in-place.
 *
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 */
void convert_d2h_unary_CPU(double *data, int mb, int nb);

/**
 * @brief Convert half precision to double precision (binary version)
 * 
 * Converts a half precision matrix to double precision with separate
 * source and target buffers.
 * 
 * @param[out] _target Pointer to the target double precision buffer
 * @param[in] _source Pointer to the source half precision buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 */
void convert_h2d_binary_CPU(double *_target, __fp16 *_source, int mb, int nb);

/**
 * @brief Convert half precision to double precision (unary version)
 *
 * Converts a half precision matrix to double precision in-place.
 *
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 */
void convert_h2d_unary_CPU(__fp16 *data, int mb, int nb);

/**
 * @brief Convert single precision to half precision (binary version)
 * 
 * Converts a single precision matrix to half precision with separate
 * source and target buffers.
 * 
 * @param[out] _target Pointer to the target half precision buffer
 * @param[in] _source Pointer to the source single precision buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 */
void convert_s2h_binary_CPU(__fp16 *_target, float *_source, int mb, int nb);

/**
 * @brief Convert single precision to half precision (unary version)
 * 
 * Converts a single precision matrix to half precision in-place.
 * 
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 */
void convert_s2h_unary_CPU(float *data, int mb, int nb);

/**
 * @brief Convert half precision to single precision (binary version)
 * 
 * Converts a half precision matrix to single precision with separate
 * source and target buffers.
 * 
 * @param[out] _target Pointer to the target single precision buffer
 * @param[in] _source Pointer to the source half precision buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 */
void convert_h2s_binary_CPU(float *_target, __fp16 *_source, int mb, int nb);

/**
 * @brief Convert half precision to single precision (unary version)
 * 
 * Converts a half precision matrix to single precision in-place.
 * 
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 */
void convert_h2s_unary_CPU(__fp16 *data, int mb, int nb);
#endif

/* ============================================================================
   Individual Conversion Functions - 8-bit Integer
   
   These functions provide optimized implementations for 8-bit integer conversions.
   They handle conversions between floating-point types and 8-bit signed integers
   (int8_t), which are commonly used for quantization and reduced precision computations
   in machine learning and signal processing applications.
   ============================================================================ */

/**
 * @brief Convert single precision to 8-bit integer (unary version)
 * 
 * Converts a single precision matrix to 8-bit integer in-place.
 * 
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension of the matrix
 */
void convert_s2i8_unary_CPU(float *data, int mb, int nb, int lda);

/**
 * @brief Convert single precision to 8-bit integer (binary version)
 * 
 * Converts a single precision matrix to 8-bit integer with separate
 * source and target buffers.
 * 
 * @param[out] _target Pointer to the target 8-bit integer buffer
 * @param[in] _source Pointer to the source single precision buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension of the matrix
 */
void convert_s2i8_binary_CPU(int8_t *_target, float *_source, int mb, int nb, int lda);

/**
 * @brief Convert double precision to 8-bit integer (unary version)
 * 
 * Converts a double precision matrix to 8-bit integer in-place.
 * 
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension of the matrix
 */
void convert_d2i8_unary_CPU(double *data, int mb, int nb, int lda);

/**
 * @brief Convert double precision to 8-bit integer (binary version)
 * 
 * Converts a double precision matrix to 8-bit integer with separate
 * source and target buffers.
 * 
 * @param[out] _target Pointer to the target 8-bit integer buffer
 * @param[in] _source Pointer to the source double precision buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension of the matrix
 */
void convert_d2i8_binary_CPU(int8_t *_target, double *_source, int mb, int nb, int lda);

/**
 * @brief Convert 8-bit integer to single precision (unary version)
 * 
 * Converts an 8-bit integer matrix to single precision in-place.
 * 
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension of the matrix
 */
void convert_8i2s_unary_CPU(int8_t *data, int mb, int nb, int lda);

/**
 * @brief Convert 8-bit integer to single precision (binary version)
 * 
 * Converts an 8-bit integer matrix to single precision with separate
 * source and target buffers.
 * 
 * @param[out] _target Pointer to the target single precision buffer
 * @param[in] _source Pointer to the source 8-bit integer buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension of the matrix
 */
void convert_8i2s_binary_CPU(float *_target, int8_t *_source, int mb, int nb, int lda);

/**
 * @brief Convert 8-bit integer to double precision (unary version)
 * 
 * Converts an 8-bit integer matrix to double precision in-place.
 * 
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension of the matrix
 */
void convert_8i2d_unary_CPU(int8_t *data, int mb, int nb, int lda);

/**
 * @brief Convert 8-bit integer to double precision (binary version)
 * 
 * Converts an 8-bit integer matrix to double precision with separate
 * source and target buffers.
 * 
 * @param[out] _target Pointer to the target double precision buffer
 * @param[in] _source Pointer to the source 8-bit integer buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension of the matrix
 */
void convert_8i2d_binary_CPU(double *_target, int8_t *_source, int mb, int nb, int lda);

/* ============================================================================
   Individual Conversion Functions - 16-bit Integer
   
   These functions provide optimized implementations for 16-bit integer conversions.
   They handle conversions between floating-point types and 16-bit signed integers
   (int16_t), which offer a good balance between precision and memory efficiency
   for many numerical computations.
   ============================================================================ */

// Single to 16-bit integer conversions
/**
 * @brief Convert single precision to 16-bit integer (unary version)
 * 
 * Converts a single precision matrix to 16-bit integer in-place with truncation.
 * 
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_s2i16_unary_CPU(float *data, int mb, int nb, int lda);

/**
 * @brief Convert single precision to 16-bit integer (binary version)
 * 
 * Converts a single precision matrix to 16-bit integer with separate buffers.
 * 
 * @param[out] _target Pointer to the target 16-bit integer buffer
 * @param[in] _source Pointer to the source single precision buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_s2i16_binary_CPU(int16_t *_target, float *_source, int mb, int nb, int lda);

// Double to 16-bit integer conversions
/**
 * @brief Convert double precision to 16-bit integer (unary version)
 * 
 * Converts a double precision matrix to 16-bit integer in-place with truncation.
 * 
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_d2i16_unary_CPU(double *data, int mb, int nb, int lda);

/**
 * @brief Convert double precision to 16-bit integer (binary version)
 * 
 * Converts a double precision matrix to 16-bit integer with separate buffers.
 * 
 * @param[out] _target Pointer to the target 16-bit integer buffer
 * @param[in] _source Pointer to the source double precision buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_d2i16_binary_CPU(int16_t *_target, double *_source, int mb, int nb, int lda);

// 16-bit integer to Single precision conversions
/**
 * @brief Convert 16-bit integer to single precision (unary version)
 * 
 * Converts a 16-bit integer matrix to single precision in-place.
 * 
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_16i2s_unary_CPU(int16_t *data, int mb, int nb, int lda);

/**
 * @brief Convert 16-bit integer to single precision (binary version)
 * 
 * Converts a 16-bit integer matrix to single precision with separate buffers.
 * 
 * @param[out] _target Pointer to the target single precision buffer
 * @param[in] _source Pointer to the source 16-bit integer buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_16i2s_binary_CPU(float *_target, int16_t *_source, int mb, int nb, int lda);

// 16-bit integer to Double precision conversions
/**
 * @brief Convert 16-bit integer to double precision (unary version)
 * 
 * Converts a 16-bit integer matrix to double precision in-place.
 * 
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_16i2d_unary_CPU(int16_t *data, int mb, int nb, int lda);

/**
 * @brief Convert 16-bit integer to double precision (binary version)
 * 
 * Converts a 16-bit integer matrix to double precision with separate buffers.
 * 
 * @param[out] _target Pointer to the target double precision buffer
 * @param[in] _source Pointer to the source 16-bit integer buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_16i2d_binary_CPU(double *_target, int16_t *_source, int mb, int nb, int lda);

/* ============================================================================
   Individual Conversion Functions - 32-bit Integer
   
   These functions provide optimized implementations for 32-bit integer conversions.
   They handle conversions between floating-point types and 32-bit signed integers
   (int), which are the standard integer type on most systems and provide good
   precision for most numerical computations.
   ============================================================================ */

// Single to 32-bit integer conversions
/**
 * @brief Convert single precision to 32-bit integer (unary version)
 * 
 * Converts a single precision matrix to 32-bit integer in-place with truncation.
 * 
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_s2i_unary_CPU(float *data, int mb, int nb, int lda);

/**
 * @brief Convert single precision to 32-bit integer (binary version)
 * 
 * Converts a single precision matrix to 32-bit integer with separate buffers.
 * 
 * @param[out] _target Pointer to the target 32-bit integer buffer
 * @param[in] _source Pointer to the source single precision buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_s2i_binary_CPU(int *_target, float *_source, int mb, int nb, int lda);

// Double to 32-bit integer conversions
/**
 * @brief Convert double precision to 32-bit integer (unary version)
 * 
 * Converts a double precision matrix to 32-bit integer in-place with truncation.
 * 
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_d2i_unary_CPU(double *data, int mb, int nb, int lda);

/**
 * @brief Convert double precision to 32-bit integer (binary version)
 * 
 * Converts a double precision matrix to 32-bit integer with separate buffers.
 * 
 * @param[out] _target Pointer to the target 32-bit integer buffer
 * @param[in] _source Pointer to the source double precision buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_d2i_binary_CPU(int *_target, double *_source, int mb, int nb, int lda);

// 32-bit integer to Single precision conversions
/**
 * @brief Convert 32-bit integer to single precision (unary version)
 * 
 * Converts a 32-bit integer matrix to single precision in-place.
 * 
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_i2s_unary_CPU(int *data, int mb, int nb, int lda);

/**
 * @brief Convert 32-bit integer to single precision (binary version)
 * 
 * Converts a 32-bit integer matrix to single precision with separate buffers.
 * 
 * @param[out] _target Pointer to the target single precision buffer
 * @param[in] _source Pointer to the source 32-bit integer buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_i2s_binary_CPU(float *_target, int *_source, int mb, int nb, int lda);

// 32-bit integer to Double precision conversions
/**
 * @brief Convert 32-bit integer to double precision (unary version)
 * 
 * Converts a 32-bit integer matrix to double precision in-place.
 * 
 * @param[in,out] data Pointer to the matrix data to convert
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_i2d_unary_CPU(int *data, int mb, int nb, int lda);

/**
 * @brief Convert 32-bit integer to double precision (binary version)
 * 
 * Converts a 32-bit integer matrix to double precision with separate buffers.
 * 
 * @param[out] _target Pointer to the target double precision buffer
 * @param[in] _source Pointer to the source 32-bit integer buffer
 * @param[in] mb Number of rows
 * @param[in] nb Number of columns
 * @param[in] lda Leading dimension
 */
void convert_i2d_binary_CPU(double *_target, int *_source, int mb, int nb, int lda);

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
 * @brief Convert to half precision using bit manipulation
 * 
 * Converts input data to half precision format using efficient bit manipulation
 * techniques. This function is optimized for performance-critical applications
 * where speed is more important than absolute precision.
 * 
 * @param[in] A Input data buffer to convert
 * @param[out] A_use Output buffer for converted half precision data
 * @param[in] mb Block size in rows
 * @param[in] nb Block size in columns
 * @param[in] decision 
 * 
 * @note This function uses bit manipulation for faster conversion
 * @note Precision may be slightly lower than standard conversion methods
 * @note Optimized for large matrices and performance-critical applications
 * @note The function assumes the output buffer is pre-allocated
 * 
 * @see hicma_parsec_params_t for parameter structure details
 */
void hicma_parsec_convert_2h_bit(void *A, float *A_use, int mb, int nb, uint16_t decision);

/**
 * @brief Convert float to half precision
 * 
 * Converts a single precision (float) matrix to half precision format.
 * This function performs the conversion on CPU using optimized algorithms
 * designed for high-performance computing applications.
 * 
 * @param[in] nrows Number of rows in the matrix
 * @param[in] ncols Number of columns in the matrix
 * @param[in] _source Source single precision data buffer
 * @param[in] ld_s Leading dimension of source matrix
 * @param[out] _target Target half precision data buffer
 * @param[in] ld_t Leading dimension of target matrix
 * 
 * @note This function is optimized for CPU execution
 * @note Memory layout (row-major vs column-major) is preserved
 * @note The function assumes the target buffer is pre-allocated
 * @note Useful for mixed precision computations
 * @note Half precision support requires appropriate hardware/compiler support
 * 
 * @see convert_s2h_unary_CPU() for in-place conversion
 * @see convert_s2h_binary_CPU() for binary conversion
 */
void float2half_CPU(int nrows, int ncols, const float *_source, int ld_s, void *_target, int ld_t);

#ifdef __cplusplus
}
#endif

#endif /* HICMA_PARSEC_DATATYPE_CONVERT_H */ 

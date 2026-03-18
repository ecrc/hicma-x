/**
 * @file hicma_parsec_hip_cuda.h
 * @brief HICMA PaRSEC HIP/CUDA compatibility header file
 * 
 * This header file provides compatibility definitions between CUDA and HIP APIs
 * for the HICMA library. It defines macros that map CUDA functions, types,
 * and constants to their HIP equivalents, enabling code to work on both
 * NVIDIA and AMD GPU platforms.
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 * 
 * @version 1.0.0
 */

#ifndef HICMA_PARSEC_HIP_CUDA_H
#define HICMA_PARSEC_HIP_CUDA_H

/* ============================================================================
 * HIP runtime includes
 * ============================================================================ */
// Core HIP runtime API for GPU memory management and kernel execution
#include <hip/hip_runtime.h>

// HIP BLAS library for linear algebra operations
#include <hipblas/hipblas.h>

// ROCm solver library for advanced linear algebra operations
#include <rocsolver/rocsolver.h>

// Half precision floating point support
#include <hip/hip_fp16.h>

// Brain floating point (bfloat16) support
#include <hip/hip_bfloat16.h>

// ROCm BLAS internal type definitions
#include <rocblas/internal/rocblas-types.h>

// Complex number support for HIP
#include <hip/hip_complex.h>

/* ============================================================================
 * PaRSEC includes
 * ============================================================================ */
// Core PaRSEC runtime system - required in each wrapper
#include "parsec.h"

// PaRSEC HIP device module for GPU task scheduling and execution
#include "dplasma/parsec/parsec/mca/device/hip/device_hip.h"

/* ============================================================================
 * CLIMATE_EMULATOR compatibility definitions
 * ============================================================================ */
// Complex number type and function mappings for climate emulator applications
// These mappings enable CUDA-based climate emulator code to work with HIP

// Double complex number type and constructor
#define make_cuDoubleComplex make_hipDoubleComplex 
// Use hipBLAS complex type for BLAS operations to avoid type conversion warnings
//#define cuDoubleComplex hipblasDoubleComplex
// Alternative mapping to HIP complex type (commented out)
#define cuDoubleComplex hipDoubleComplex 

// Complex number component access functions
#define cuCreal hipCreal    // Extract real part of complex number
#define cuCimag hipCimag    // Extract imaginary part of complex number

// Complex BLAS operations for climate emulator
#define cublasZgemm hipblasZgemm    // Complex double precision matrix multiplication
#define cublasZgemv hipblasZgemv    // Complex double precision matrix-vector multiplication
#define cublasZdotu hipblasZdotu    // Complex double precision dot product (unconjugated)
#define cublasDaxpy hipblasDaxpy    // Double precision axpy operation

/* ============================================================================
 * CUDA to HIP type mappings
 * ============================================================================ */
// Core CUDA types mapped to their HIP equivalents
// These mappings enable CUDA code to compile and run on AMD GPUs

#define cudaDeviceProp hipDeviceProp_t        // GPU device properties structure
#define cudaError_t hipError_t                // Error code type for HIP operations
#define cudaEvent_t hipEvent_t                // Event synchronization object
#define cudaStream_t hipStream_t              // Asynchronous execution stream
#define cudaSurfaceObject_t hipSurfaceObject_t // Surface memory object for texture operations

/* ============================================================================
 * CUDA to HIP function mappings
 * ============================================================================ */
// Core CUDA runtime functions mapped to their HIP equivalents
// These mappings provide device management, memory operations, and synchronization

// Device management functions
#define cudaDeviceReset hipDeviceReset                    // Reset GPU device to initial state
#define cudaDeviceSynchronize hipDeviceSynchronize        // Synchronize all device operations
#define cudaGetDevice hipGetDevice                        // Get current active device
#define cudaGetDeviceProperties hipGetDeviceProperties    // Get device properties
#define cudaSetDevice hipSetDevice                        // Set active device

// Event synchronization functions
#define cudaEventCreate hipEventCreate                    // Create event object
#define cudaEventDestroy hipEventDestroy                  // Destroy event object
#define cudaEventElapsedTime hipEventElapsedTime          // Calculate elapsed time between events
#define cudaEventRecord hipEventRecord                    // Record event in stream
#define cudaEventSynchronize hipEventSynchronize          // Wait for event completion

// Memory management functions
#define cudaFree hipFree                                  // Free device memory
#define cudaFreeHost hipHostFree                          // Free host memory (hipFreeHost deprecated)
#define cudaMalloc hipMalloc                              // Allocate device memory
#define cudaMallocHost hipHostMalloc                      // Allocate host memory (hipMallocHost deprecated)
#define cudaMemcpy hipMemcpy                              // Copy memory between host/device
#define cudaMemset hipMemset                              // Initialize device memory

// Memory optimization functions
#define cudaMemAdvise hipMemAdvise                        // Provide memory access hints
#define cudaMemPrefetchAsync hipMemPrefetchAsync          // Prefetch memory asynchronously

// Stream management functions
#define cudaStreamCreate hipStreamCreate                  // Create execution stream
#define cudaStreamDestroy hipStreamDestroy                // Destroy execution stream

// Error handling functions
#define cudaGetErrorName hipGetErrorName                  // Get error name from error code
#define cudaGetErrorString hipGetErrorString              // Get error description from error code
#define cudaGetLastError hipGetLastError                  // Get last error from device

/* ============================================================================
 * CUDA to HIP enum value mappings
 * ============================================================================ */
// CUDA enumeration values mapped to their HIP equivalents
// These mappings ensure compatibility for CUDA code using specific enum values

// Boundary mode constants for texture operations
#define cudaBoundaryModeClamp hipBoundaryModeClamp        // Clamp to edge values
#define cudaBoundaryModeTrap hipBoundaryModeTrap          // Trap on out-of-bounds access
#define cudaBoundaryModeZero hipBoundaryModeZero          // Return zero for out-of-bounds access

// Memory advice constants
#define cudaMemAdviseSetReadMostly hipMemAdviseSetReadMostly // Mark memory as read-mostly

// Memory copy direction constants
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost      // Copy from device to host
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice      // Copy from host to device

// Success status constant
#define cudaSuccess hipSuccess                            // Operation completed successfully

/* ============================================================================
 * cuBLAS to hipBLAS mappings
 * ============================================================================ */
// cuBLAS library functions and constants mapped to hipBLAS equivalents
// These mappings enable CUDA BLAS code to work with AMD GPUs

// Compute type constants for mixed-precision operations
#define CUBLAS_COMPUTE_64F HIPBLAS_R_64F                   // Double precision compute type
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F                   // Single precision compute type
#define CUBLAS_COMPUTE_16F HIPBLAS_R_16F                   // Half precision compute type

// Matrix operation constants
#define CUBLAS_DIAG_NON_UNIT_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#define CUBLAS_DIAG_NON_UNIT_FILL_MODE_LOWER HIPBLAS_FILL_MODE_LOWER
#define CUBLAS_DIAG_NON_UNIT_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_DIAG_NON_UNIT_OP_N HIPBLAS_OP_N
#define CUBLAS_DIAG_NON_UNIT_OP_T HIPBLAS_OP_T
#define CUBLAS_DIAG_NON_UNIT_SIDE_RIGHT HIPBLAS_SIDE_RIGHT
#define CUBLAS_DIAG_NON_UNIT_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS

// cuBLAS handle management functions
#define cublasCreate hipblasCreate                         // Create cuBLAS handle
#define cublasDestroy hipblasDestroy                       // Destroy cuBLAS handle
#define cublasDestroy_v2 hipblasDestroy                    // Destroy cuBLAS handle (v2 API)

// Double precision BLAS operations
#define cublasDgemm hipblasDgemm                           // Double precision matrix multiplication
#define cublasDsyrk hipblasDsyrk                           // Double precision symmetric rank-k update
#define cublasDtrsm hipblasDtrsm                           // Double precision triangular solve

// Single precision BLAS operations
#define cublasSgemm hipblasSgemm                           // Single precision matrix multiplication
#define cublasSsyrk hipblasSsyrk                           // Single precision symmetric rank-k update
#define cublasStrsm hipblasStrsm                           // Single precision triangular solve

// Advanced BLAS operations
#define cublasGemmEx hipblasGemmEx                         // Extended precision matrix multiplication

// cuBLAS type definitions
#define cublasGemmAlgo_t hipblasGemmAlgo_t                 // GEMM algorithm type
#define cublasHandle_t hipblasHandle_t                     // cuBLAS handle type
#define cublasOperation_t hipblasOperation_t               // Matrix operation type
#define cublasStatus_t hipblasStatus_t                     // cuBLAS status type

// Stream management
#define cublasSetStream hipblasSetStream                   // Set stream for cuBLAS operations

// cuBLAS status constants
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH

// Matrix operation constants
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT           // Default GEMM algorithm
#define CUBLAS_OP_T HIPBLAS_OP_T                           // Transpose operation
#define CUBLAS_OP_N HIPBLAS_OP_N                           // No transpose operation
#define CUBLAS_FILL_MODE_LOWER HIPBLAS_FILL_MODE_LOWER     // Lower triangular fill mode
#define CUBLAS_SIDE_RIGHT HIPBLAS_SIDE_RIGHT               // Right side operation

/* ============================================================================
 * CUDA to HIP datatype mappings
 * ============================================================================ */
// CUDA data type constants mapped to hipBLAS equivalents
// These mappings enable mixed-precision operations on AMD GPUs

// Standard precision data types
#define CUDA_R_16BF HIPBLAS_R_16B                         // Brain floating point (16-bit)
#define CUDA_R_16F HIPBLAS_R_16F                          // Half precision floating point
#define CUDA_R_32F HIPBLAS_R_32F                          // Single precision floating point
#define CUDA_R_64F HIPBLAS_R_64F                          // Double precision floating point
#define CUDA_R_32I HIPBLAS_R_32I                          // 32-bit integer
#define CUDA_R_8I HIPBLAS_R_8I                            // 8-bit integer

// Fast compute path mappings for mixed-precision operations
// Note: These mappings may need verification for optimal performance
#define CUBLAS_COMPUTE_32F_FAST_TF32 HIPBLAS_R_32F        // TF32 fast compute path
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_16F         // FP16 fast compute path
#define CUBLAS_COMPUTE_32F_FAST_16BF HIPBLAS_R_16B        // BF16 fast compute path

/* ============================================================================
 * Other CUDA to HIP mappings
 * ============================================================================ */
// Additional CUDA functions and types mapped to HIP equivalents
// These mappings cover advanced memory operations and stream management

// Data type definitions
#define cudaDataType_t hipblasDatatype_t                   // CUDA data type enumeration

// Memory information and management
#define cudaMemGetInfo hipMemGetInfo                       // Get memory usage information
#define cudaMemcpyAsync hipMemcpyAsync                     // Asynchronous memory copy
#define cudaMemcpyKind hipMemcpyKind                       // Memory copy direction type
#define cudaHostRegister hipHostRegister                   // Memory register 
#define cudaHostUnregister hipHostUnregister               // Memory unregister 
#define cudaHostRegisterDefault hipHostRegisterDefault     // Memory register flag 

// Advanced stream operations
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags // Create stream with specific flags
#define cudaStreamNonBlocking hipStreamNonBlocking         // Non-blocking stream flag
#define cudaStreamSynchronize hipStreamSynchronize         // Synchronize stream execution

// Stream type alias
#define cuda_stream hip_stream                             // Stream type alias

/* ============================================================================
 * PaRSEC device module mappings
 * ============================================================================ */
// PaRSEC device module types mapped from CUDA to HIP
// These mappings enable PaRSEC task scheduling to work with AMD GPUs

#define parsec_device_cuda_module_t parsec_device_hip_module_t  // PaRSEC device module type
#define parsec_cuda_exec_stream_t parsec_hip_exec_stream_t     // PaRSEC execution stream type
#define dplasma_cuda_handles_t dplasma_hip_handles_t           // DPLASMA handle type
#define cuda_device hip_device                                 // Device identifier type
#define cuda_index hip_index                                   // Device index type 

/* ============================================================================
 * cuSOLVER to hipBLAS mappings
 * ============================================================================ */
// cuSOLVER library functions mapped to hipBLAS equivalents
// Note: cuSOLVER functionality is provided through hipBLAS in ROCm
// rocsolver_status is an alias of rocblas_status

// cuSOLVER handle and status types
#define cusolverDnHandle_t hipblasHandle_t                 // cuSOLVER handle type
// Alternative status type mapping (commented out)
//#define cusolverStatus_t rocblas_status 
#define cusolverStatus_t hipblasStatus_t                   // cuSOLVER status type

// cuSOLVER status constants
#define CUSOLVER_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS     // Success status
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT         // Non-unit diagonal constant

// cuSOLVER handle management functions
#define cusolverDnSetStream hipblasSetStream               // Set stream for cuSOLVER operations
#define cusolverDnCreate hipblasCreate                     // Create cuSOLVER handle
#define cusolverDnDestroy hipblasDestroy                   // Destroy cuSOLVER handle

#endif /* HICMA_PARSEC_HIP_CUDA_H */

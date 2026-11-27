/**
 * @file hicma_parsec_io.h
 * @brief HICMA PaRSEC I/O operations header file
 * 
 * This header file contains declarations for I/O operations including file reading,
 * writing, and MPI-based data distribution operations for the HICMA library.
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 * 
 * @version 0.2.0
 */

#ifndef HICMA_PARSEC_IO_H
#define HICMA_PARSEC_IO_H

/* ============================================================================
 * HICMA PaRSEC includes
 * ============================================================================ */
#include "hicma_parsec_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * MPI data type creation functions
 * ============================================================================ */

/**
 * @brief Create MPI datatype for 2D data distribution
 * 
 * Creates a custom MPI datatype for efficient 2D data distribution across
 * a process grid.
 * 
 * @param[in] procs Total number of processes
 * @param[in] mpirank Rank of the current process
 * @param[in] M Number of rows in the global matrix
 * @param[in] N Number of columns in the global matrix
 * @param[in] distM Row distribution factor
 * @param[in] distN Column distribution factor
 * @param[in] mb Block size for rows
 * @param[in] nb Block size for columns
 * @param[in] prow Row coordinate in process grid
 * @param[in] pcol Column coordinate in process grid
 * @param[in] oldtype Original MPI datatype
 * @param[out] newtype New MPI datatype for 2D distribution
 */
/**
 * @brief Create MPI datatype for 2D data distribution
 * 
 * Creates a custom MPI datatype for efficient 2D data distribution across
 * a process grid. This function optimizes data transfer patterns for
 * matrix operations in distributed memory systems.
 * 
 * @param[in] procs Total number of processes
 * @param[in] mpirank Rank of the current process
 * @param[in] M Number of rows in the global matrix
 * @param[in] N Number of columns in the global matrix
 * @param[in] distM Row distribution factor
 * @param[in] distN Column distribution factor
 * @param[in] mb Block size for rows
 * @param[in] nb Block size for columns
 * @param[in] prow Row coordinate in process grid
 * @param[in] pcol Column coordinate in process grid
 * @param[in] oldtype Original MPI datatype
 * @param[out] newtype New MPI datatype for 2D distribution
 */
void Creating_MPI_Datatype_2D(int procs, int mpirank,
                int M, int N, int distM, int distN,
                int mb, int nb, int prow, int pcol,
                MPI_Datatype oldtype, MPI_Datatype *newtype);

/* ============================================================================
 * MPI file I/O functions
 * ============================================================================ */

/**
 * @brief Write single precision matrix to file using MPI I/O
 * 
 * Writes a single precision matrix to a file using MPI I/O for efficient
 * parallel file operations. The function uses custom MPI datatypes for
 * optimal data distribution and transfer patterns.
 * 
 * @param[in] filename Output filename for the matrix data
 * @param[in] mpirank Rank of the current MPI process
 * @param[in] matrix Single precision matrix data to write
 * @param[in] size Size of the matrix data in elements
 * @param[in] oldtype Original MPI datatype for the matrix elements
 * @param[in] newtype New MPI datatype optimized for distribution
 */
void MPI_Writing_sfile(const char *filename, int mpirank, float *matrix, MPI_Count size, MPI_Datatype oldtype, MPI_Datatype newtype);

/**
 * @brief Write double precision matrix to file using MPI I/O
 * 
 * Writes a double precision matrix to a file using MPI I/O for efficient
 * parallel file operations. The function uses custom MPI datatypes for
 * optimal data distribution and transfer patterns.
 * 
 * @param[in] filename Output filename for the matrix data
 * @param[in] mpirank Rank of the current MPI process
 * @param[in] matrix Double precision matrix data to write
 * @param[in] size Size of the matrix data in elements
 * @param[in] oldtype Original MPI datatype for the matrix elements
 * @param[in] newtype New MPI datatype optimized for distribution
 */
void MPI_Writing_dfile(const char *filename, int mpirank, double *matrix, MPI_Count size, MPI_Datatype oldtype, MPI_Datatype newtype);

/**
 * @brief Read single precision matrix from file using MPI I/O
 * 
 * Reads a single precision matrix from a file using MPI I/O for efficient
 * parallel file operations. The function uses custom MPI datatypes for
 * optimal data distribution and transfer patterns.
 * 
 * @param[in] filename Input filename containing the matrix data
 * @param[in] mpirank Rank of the current MPI process
 * @param[out] matrix Single precision matrix buffer to read data into
 * @param[in] size Size of the matrix data in elements
 * @param[in] oldtype Original MPI datatype for the matrix elements
 * @param[in] newtype New MPI datatype optimized for distribution
 */
void MPI_Reading_sfile(char *filename, int mpirank, float *matrix, MPI_Count size, MPI_Datatype oldtype, MPI_Datatype newtype);

/**
 * @brief Read double precision matrix from file using MPI I/O
 * 
 * Reads a double precision matrix from a file using MPI I/O for efficient
 * parallel file operations. The function uses custom MPI datatypes for
 * optimal data distribution and transfer patterns.
 * 
 * @param[in] filename Input filename containing the matrix data
 * @param[in] mpirank Rank of the current MPI process
 * @param[out] matrix Double precision matrix buffer to read data into
 * @param[in] size Size of the matrix data in elements
 * @param[in] oldtype Original MPI datatype for the matrix elements
 * @param[in] newtype New MPI datatype optimized for distribution
 */
void MPI_Reading_dfile(char *filename, int mpirank, double *matrix, MPI_Count size, MPI_Datatype oldtype, MPI_Datatype newtype);

/* ============================================================================
 * Standard file I/O functions
 * ============================================================================ */

/**
 * @brief Read matrix from file
 * 
 * Reads a single precision matrix from a file with specified dimensions.
 * The function assumes the file contains raw binary data in column-major format.
 * 
 * @param[in] filename Input filename containing the matrix data
 * @param[in] M Number of rows in the matrix
 * @param[in] N Number of columns in the matrix
 * @param[out] matrix Single precision matrix buffer to read data into
 * @return 0 on success, non-zero on failure
 */
int read_file(char *filename, size_t M, size_t N, float *matrix);

/**
 * @brief Write double precision matrix to binary file
 * 
 * Writes a double precision matrix to a binary file in raw format.
 * The matrix is stored in column-major order for compatibility with
 * numerical computing libraries.
 * 
 * @param[in] filename Output filename for the binary matrix data
 * @param[in] M Number of rows in the matrix
 * @param[in] N Number of columns in the matrix
 * @param[in] matrix Double precision matrix data to write
 */
void writeMatrixToBinaryFile(const char *filename, size_t M, size_t N, double *matrix);

/**
 * @brief Write single precision matrix to binary file
 * 
 * Writes a single precision matrix to a binary file in raw format.
 * The matrix is stored in column-major order for compatibility with
 * numerical computing libraries.
 * 
 * @param[in] filename Output filename for the binary matrix data
 * @param[in] M Number of rows in the matrix
 * @param[in] N Number of columns in the matrix
 * @param[in] matrix Single precision matrix data to write
 */
void writeMatrixToBinaryFileSingle(const char *filename, size_t M, size_t N, float *matrix);

/* ============================================================================
 * Matrix address calculation functions
 * ============================================================================ */

/**
 * @brief Get address for matrix element in column-major format
 * 
 * Calculates the memory address offset for a specific matrix element
 * in a distributed tiled matrix with column-major storage layout.
 * This function is essential for accessing matrix elements across
 * different processes in the distributed computation.
 * 
 * @param[in] dcA PaRSEC tiled matrix descriptor
 * @param[in] m Row index of the element (global coordinates)
 * @param[in] n Column index of the element (global coordinates)
 * @param[in] p Process row coordinate in the process grid
 * @param[in] q Process column coordinate in the process grid
 * @return Address offset for the specified element in the local matrix buffer
 */
size_t parsec_getaddr_cm(parsec_tiled_matrix_t *dcA, int m, int n, int p, int q);

/* ============================================================================
 * Distributed file operations
 * ============================================================================ */

/**
 * @brief Read distributed files and connect them together
 * 
 * Reads matrix data from distributed files and assembles them into a single
 * matrix descriptor for computation.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] path Path to the distributed files
 * @param[in] desc Matrix descriptor to populate
 * @param[in] MB Block size for rows
 * @param[in] NB Block size for columns
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int dist_read(parsec_context_t *parsec,
        char *path,
        parsec_matrix_block_cyclic_t *desc,
        int MB, int NB, hicma_parsec_params_t *params);

#ifdef __cplusplus
}
#endif

#endif /* HICMA_PARSEC_IO_H */

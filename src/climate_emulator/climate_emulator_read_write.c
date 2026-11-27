/**
 * @file climate_emulator_read_write.c
 * @brief HICMA PaRSEC climate emulator I/O functions
 * 
 * This file contains functions for reading and writing climate data matrices
 * to and from files. It supports both MPI-IO and binary file formats,
 * with automatic data distribution across MPI processes.
 * 
 * The I/O functions handle matrix data in block-cyclic distribution format
 * and can work with both symmetric and general matrices.
 * 
 * Key features:
 * - MPI-IO support for parallel file operations
 * - Binary file format for single-process output
 * - Automatic data format conversion (PaRSEC tiled to Lapack)
 * - Support for symmetric and general matrices
 * - Error handling and validation
 */

#include "climate_emulator.h"

/* ============================================================================
 * Matrix reading functions
 * ============================================================================ */

/**
 * @brief Read climate emulator data from file
 * 
 * Reads matrix data from a file using MPI-IO. The data is distributed
 * across MPI processes according to the matrix descriptor's block-cyclic
 * distribution pattern.
 * 
 * This function creates an MPI derived datatype for the matrix distribution,
 * allocates local memory, reads the data, and converts it from Lapack
 * format to the PaRSEC tiled format.
 * 
 * The reading process includes:
 * - Creating MPI derived datatype for 2D block-cyclic distribution
 * - Allocating local memory for matrix data
 * - Reading data using MPI-IO
 * - Converting from Lapack to PaRSEC tiled format
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] filename Input filename to read from
 * @param[in] A Matrix descriptor for the data to be read
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 * @param[in] symm Symmetry flag (1 for symmetric, 0 for general)
 */
void climate_emulator_read(parsec_context_t *parsec,
        char *filename,
        parsec_tiled_matrix_t *A,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *parsec_params,
        int symm)
{
    int rank = parsec_params->rank;
    int nodes = parsec_params->nodes;
    int P = parsec_params->P;
    int Q = parsec_params->Q;
    int MB = A->mb;        /* Block size for rows */
    int NB = A->nb;        /* Block size for columns */
    int M = A->lm;         /* Global number of rows */
    int N = A->ln;         /* Global number of columns */

    if(rank == 0) fprintf(stderr, RED "\nRead matrix from file:\n" RESET);

    /* Create MPI derived datatype for 2D block-cyclic distribution */
    MPI_Datatype darrayA;
    Creating_MPI_Datatype_2D(nodes, rank,
            M, N, MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC,
            MB, NB, P, Q,
            MPI_DOUBLE, &darrayA);

    /* Allocate memory for local matrix data */
    double* DA = (double *) parsec_data_allocate((size_t)A->nb_local_tiles *
            (size_t)A->bsiz *
            (size_t)parsec_datadist_getsizeoftype(A->mtype));

    /* Read data from file using MPI-IO */
    MPI_Reading_dfile(filename, rank, DA, A->llm*A->lln, MPI_DOUBLE, darrayA);
    
    /* Convert from Lapack format to PaRSEC tiled format */
    if( 0 != hicma_parsec_Lapack_to_Tile( parsec, A, DA, P, Q) ) exit(1);  /* message printed by the conversion */
    /* Free allocated memory */
    parsec_data_free(DA);
}

/* ============================================================================
 * Matrix writing functions
 * ============================================================================ */

/**
 * @brief Write climate emulator data to file
 * 
 * Writes matrix data to a file in either MPI-IO or binary format depending
 * on configuration. Supports both symmetric and general matrices.
 * 
 * This function can write data using MPI-IO for parallel I/O operations
 * or binary format for single-process output. It handles the conversion
 * from PaRSEC tiled format to Lapack format before writing.
 * 
 * The writing process includes:
 * - Converting from PaRSEC tiled format to Lapack format
 * - Creating MPI derived datatype for distribution (if using MPI-IO)
 * - Writing data using appropriate I/O method
 * - Memory cleanup after writing
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] filename Output filename to write to
 * @param[in] A Matrix to write
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] params HICMA PaRSEC parameters
 * @param[in] symm Symmetry flag (1 for symmetric, 0 for general)
 */
void climate_emulator_write(parsec_context_t *parsec,
        char *filename,
        parsec_tiled_matrix_t *A,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *parsec_params,
        int symm)
{
    int rank = parsec_params->rank;
    int nodes = parsec_params->nodes;
    int P = ((parsec_matrix_block_cyclic_t *)A)->grid.rows;  /* Number of process rows */
    int Q = ((parsec_matrix_block_cyclic_t *)A)->grid.cols;  /* Number of process columns */
    int MB = A->mb;        /* Block size for rows */
    int NB = A->nb;        /* Block size for columns */
    int M = A->lm;         /* Global number of rows */
    int N = A->ln;         /* Global number of columns */

    if(rank == 0) {
        fprintf(stderr, RED "\nWrite matrix to file:\n" RESET);
        fprintf(stderr, RED "\nP %d Q %d MB %d NB %d M %d N %d:\n" RESET, P, Q, MB, NB, M, N);
    }
  
#if CLIMATE_EMULATOR_MPIIO
    /* Use MPI-IO for parallel file writing */
    
    /* Create MPI derived datatype for 2D block distribution */
    MPI_Datatype darrayA;
    Creating_MPI_Datatype_2D(nodes, rank,
        M, N, MPI_DISTRIBUTE_BLOCK, MPI_DISTRIBUTE_BLOCK,
        MB, NB, P, Q,
        MPI_DOUBLE, &darrayA);

    /* Allocate memory for local matrix data */
    double* DA = (double *) parsec_data_allocate((size_t)A->nb_local_tiles *
            (size_t)A->bsiz *
            (size_t)parsec_datadist_getsizeoftype(A->mtype));

    /* Convert from PaRSEC tiled format to Lapack format */
    if(symm) {
        /* Handle symmetric matrices */
        if( 0 != hicma_parsec_Tile_to_Lapack_sym( parsec, (parsec_tiled_matrix_t *)A, DA, P, nodes/P) ) exit(1);  /* message printed by the conversion */
    } else {
        /* Handle general matrices */
        if( 0 != hicma_parsec_Tile_to_Lapack( parsec, (parsec_tiled_matrix_t *)A, DA, P, nodes/P) ) exit(1);  /* message printed by the conversion */
    }
    /* Write data to file using MPI-IO */
    MPI_Writing_dfile(filename, rank, DA, A->llm*A->lln, MPI_DOUBLE, darrayA);
    
    /* Free allocated memory */
    parsec_data_free(DA);
#else
    /* Use binary file format for single-process output */
    
#if CLIMATE_EMULATOR_DEBUG_INFO
    /* Print matrix in debug mode */
    if( symm ) parsec_print_cm_sym(parsec, A, nodes, P, nodes/P);  /* the matrix being written */
#endif

    /* Allocate memory for the entire matrix */
    double* DA = (double *)malloc((size_t)A->llm*(size_t)A->lln*
                                    (size_t)parsec_datadist_getsizeoftype(A->mtype));

    /* Convert from PaRSEC tiled format to Lapack format */
    if(symm) {
        /* Handle symmetric matrices */
        if( 0 != hicma_parsec_Tile_to_Lapack_sym( parsec, (parsec_tiled_matrix_t *)A, DA, P, nodes/P) ) exit(1);  /* message printed by the conversion */
    } else {
        /* Handle general matrices */
        if( 0 != hicma_parsec_Tile_to_Lapack( parsec, (parsec_tiled_matrix_t *)A, DA, P, nodes/P) ) exit(1);  /* message printed by the conversion */
    }
    /* Write data to binary file */
    writeMatrixToBinaryFile(filename, N, N, DA);

    /* Free allocated memory */
    free(DA);
#endif
}

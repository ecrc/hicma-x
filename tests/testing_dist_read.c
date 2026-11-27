/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

/**
 * @file testing_dist_read.c
 * @brief Distributed matrix reading and redistribution test program
 * 
 * This program tests the functionality of reading distributed genotype data
 * from files and redistributing it across the computational grid. It demonstrates
 * the use of HICMA and PaRSEC for distributed matrix operations.
 * 
 * The program:
 * 1. Initializes HICMA and PaRSEC contexts
 * 2. Reads distributed genotype data from files
 * 3. Redistributes the data to a new matrix layout
 * 4. Cleans up allocated resources
 */

#include "hicma_parsec.h"

/**
 * @brief Main function for testing distributed matrix reading and redistribution
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return 0 on success, non-zero on failure
 */
int main(int argc, char ** argv)
{
    // HICMA and PaRSEC parameter structures
    hicma_parsec_params_t params;
    starsh_params_t params_kernel;
    hicma_parsec_data_t data;
    hicma_parsec_matrix_analysis_t analysis = {0};
    
    // PaRSEC context pointer
    parsec_context_t* parsec = NULL;

    /* Initialize HICMA and PaRSEC contexts */
    parsec = hicma_parsec_init( argc, argv, &params, &params_kernel, &data ); 
    if (parsec == NULL) {
        fprintf(stderr, "Error: Failed to initialize HICMA and PaRSEC contexts\n");
        return 1;
    }

    // Define descriptor for the genotype data matrix
    parsec_matrix_block_cyclic_t genotype_desc;
 
    // Extract parameters for matrix dimensions
    int SNPS = params.nsnp;        // Number of SNPs (Single Nucleotide Polymorphisms)
    int CHK = params.order;        // Order/chunk size for matrix tiling
    
    printf("Initializing test with %d SNPs and chunk size %d\n", SNPS, CHK);
   
    // Initialize the destination matrix A with block-cyclic distribution
    parsec_matrix_block_cyclic_t A;
    parsec_matrix_block_cyclic_init(&A, PARSEC_MATRIX_FLOAT, PARSEC_MATRIX_TILE,
            params.rank,                    // Process rank
            ((params.nsnp<params.NB)? params.nsnp:params.NB),  // Number of rows in local tile
            params.NB,                      // Block size
            params.nsnp,                    // Total number of SNPs
            params.N,                       // Total number of samples
            0, 0,                          // Starting row and column indices
            params.nsnp,                   // Number of rows in global matrix
            params.N,                       // Number of columns in global matrix
            params.P,                       // Number of processes in row
            (params.nodes)/(params.P),      // Number of processes in column
            1, 1,                          // Row and column strides
            0, 0);                         // Row and column offsets

    // Allocate memory for the matrix data
    A.mat = parsec_data_allocate((size_t)A.super.nb_local_tiles *
            (size_t)A.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(A.super.mtype));   
    
    if (A.mat == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for matrix A\n");
        parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&A);
        hicma_parsec_fini(parsec, argc, argv, &params, &params_kernel, &data, &analysis);
        return 1;
    }
    
    // Set a key for the data collection for debugging purposes
    parsec_data_collection_set_key((parsec_data_collection_t*)&A, "A"); 
    
    printf("Reading distributed genotype data from: %s\n", params.mesh_file);
    
    /* Read distributed genotype data from files */
    dist_read(parsec, params.mesh_file, &genotype_desc, SNPS, CHK, &params);
    
    printf("Redistributing data from genotype descriptor to matrix A\n");
    
    /* Redistribute data from genotype descriptor to matrix A */
    parsec_redistribute(parsec, (parsec_tiled_matrix_t *) &genotype_desc, 
                       (parsec_tiled_matrix_t *)&A,  
                       params.nsnp, params.N, 0, 0, 0, 0);

    printf("Cleaning up allocated resources\n");

    /* Cleanup: Free allocated memory and destroy matrices */
    parsec_data_free(A.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&A);

    /* Finalize HICMA and PaRSEC contexts */
    hicma_parsec_fini(parsec, argc, argv, &params, &params_kernel, &data, &analysis);

    printf("Test completed successfully\n");
    return 0;
}


/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

/**
 * @file testing_igemm_summa.c
 * @brief Test program for integer GEMM operations using HiCMA and PaRSEC with SUMMA algorithm
 * 
 * This program benchmarks integer matrix-matrix multiplication (GEMM) operations
 * using the HiCMA library and PaRSEC runtime system. It implements the SUMMA
 * (Scalable Universal Matrix Multiplication Algorithm) for distributed memory
 * parallel computing.
 * 
 * Features:
 * - Integer GEMM operations with HiCMA
 * - PaRSEC-based parallel task scheduling
 * - SUMMA algorithm implementation
 * - GPU acceleration support
 * - Performance benchmarking with multiple runs
 * - Memory management and cleanup
 */

#include "hicma_parsec.h"

/**
 * @brief Check the residual of the solution for symmetric matrix operations
 * 
 * This function computes and checks the residual ||Ax - b|| to verify
 * the correctness of the solution for symmetric matrix operations.
 * 
 * @param parsec PaRSEC context
 * @param loud Verbosity level for output
 * @param A Input symmetric matrix A
 * @param b Right-hand side vector b
 * @param x Solution vector x
 * @return 0 if solution is correct, 1 if suspicious
 */
int check_saxmb2( parsec_context_t *parsec, int loud,
                 parsec_tiled_matrix_t *A,
                 parsec_tiled_matrix_t *b,
                 parsec_tiled_matrix_t *x )
{
    int info_solution = 0;
    double Rnorm = 0.0;  /* Residual norm */
    double Anorm = 0.0;  /* Matrix A norm */
    double Bnorm = 0.0;  /* Vector b norm */
    double Xnorm, result;
    int N = b->m;
    double eps = LAPACKE_slamch_work('e');  /* Machine epsilon */

    /* Compute matrix and vector norms */
    //Anorm = dplasma_dlansy(parsec, PlasmaFrobeniusNorm, A);
    Anorm = dplasma_slansy(parsec, PlasmaFrobeniusNorm, PlasmaLower, A);
    Bnorm = dplasma_slange(parsec, PlasmaFrobeniusNorm, b);
    Xnorm = dplasma_slange(parsec, PlasmaFrobeniusNorm, x);

    /* Compute residual: b - A*x */
    dplasma_ssymm( parsec, dplasmaLeft, PlasmaLower, -1.0, A, x, 1.0, b);

    /* Compute residual norm */
    Rnorm = dplasma_slange(parsec, PlasmaFrobeniusNorm, b);

    /* Calculate relative residual */
    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * N * eps ) ;

    /* Simplified residual check */
    result = Rnorm /  ( Anorm); // * Xnorm + Bnorm ) * N * eps ) ;

    /* Print detailed residual information if verbose */
    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||X||_oo = %e, ||B||_oo= %e, ||A X - B||_oo = %e\n",
                    Anorm, Xnorm, Bnorm, Rnorm );

        //printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n", result);
        printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo)) = " RED "%e" RESET "\n", result);

    }

    /* Check if solution is numerically stable */
    if (  isnan(Xnorm) || isinf(Xnorm) || isnan(result) || isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Solution is suspicious ! \n");
        info_solution = 1;
    }
    else{
        if( loud ) printf(GRN "-- Solution is CORRECT ! \n" RESET);
        info_solution = 0;
    }

    return info_solution;
}

/**
 * @brief Main function for integer GEMM SUMMA testing
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return 0 on success
 */
int main(int argc, char ** argv)
{
    /* HiCMA and PaRSEC parameter structures */
    hicma_parsec_params_t params;
    starsh_params_t params_kernel;
    hicma_parsec_data_t data;
    hicma_parsec_matrix_analysis_t analysis;
    
    /* Initialize HiCMA and PaRSEC */
    parsec_context_t* parsec = hicma_parsec_init( argc, argv, &params, &params_kernel, &data );
    if( NULL == parsec ) {
        fprintf(stderr, "Failed to initialize HiCMA PaRSEC\n");
        return EXIT_FAILURE;
    }

    /* Initialize input matrix A with block-cyclic distribution */
    /* Matrix A: size params.nsnp x params.N, tile size params.NB */
    parsec_matrix_block_cyclic_t A;
    parsec_matrix_block_cyclic_init(&A, PARSEC_MATRIX_FLOAT, PARSEC_MATRIX_TILE,
            params.rank, ((params.nsnp<params.NB)? params.nsnp:params.NB), params.NB, params.nsnp, params.N, 0, 0,
            params.nsnp, params.N, params.P, (params.nodes)/(params.P), 1, 1, 0, 0);

    /* Initialize output matrix AA for storing A^T * A result */
    /* Matrix AA: size params.N x params.N, tile size params.NB */
    parsec_matrix_block_cyclic_t AA;
    parsec_matrix_block_cyclic_init(&AA, PARSEC_MATRIX_FLOAT, PARSEC_MATRIX_TILE,
            params.rank, params.NB, params.NB, params.N, params.N, 0, 0,
            params.N, params.N, params.P, (params.nodes)/(params.P), 1, 1, 0, 0);
 
    /* Allocate memory for output matrix AA */
    parsec_dist_allocate(parsec, (parsec_tiled_matrix_t *)&AA, &params);

    /* Generate test data for input matrix A */
    parsec_genotype_generator(parsec, (parsec_tiled_matrix_t *)&A, &params, 1); 
 
    /* Print timing information for memory allocation and data generation */
    SYNC_TIME_PRINT(params.rank, ("memory allocate and data generation\n"));

    /* Perform GPU warmup if GPUs are available */
    if(params.gpus > 0) {
        SYNC_TIME_START();
        
        /* Warmup with input matrix A */
        hicma_parsec_potrf_L_warmup( parsec, (parsec_tiled_matrix_t *)&A, &params, 0);
        
        /* Warmup with output matrix AA */
        hicma_parsec_potrf_L_warmup( parsec, (parsec_tiled_matrix_t *)&AA, &params, 0);
        
        SYNC_TIME_PRINT(params.rank, ("warmup\n"));
    }

    /* Execute GEMM operation multiple times for benchmarking */
    for(int i = 0; i < params.nruns; i++) {
        /* Synchronize all processes before timing */
        MPI_Barrier( MPI_COMM_WORLD );
        SYNC_TIME_START();
        
        /* Start timing */
        struct timeval tstart;
        gettimeofday(&tstart, NULL);
        params.start_time_gemm = tstart.tv_sec + tstart.tv_usec / 1.0e6;

        /* Execute integer GEMM: AA = A^T * A */
        /* This computes the symmetric matrix AA = A^T * A using SUMMA algorithm */
        hicma_parsec_gemmex(parsec,
                PlasmaTrans,      /* Transpose matrix A */
                PlasmaNoTrans,    /* Don't transpose matrix A */
                1.0,              /* Scaling factor alpha */
                (parsec_tiled_matrix_t *)&A,    /* Input matrix A */
                (parsec_tiled_matrix_t *)&A,    /* Input matrix A (same as A) */
                0.0,              /* Scaling factor beta */
                (parsec_tiled_matrix_t *)&AA,   /* Output matrix AA */
                &params, &data);

        /* Synchronize all processes after computation */
        MPI_Barrier( MPI_COMM_WORLD );
        
        /* End timing */
        struct timeval tend;
        gettimeofday(&tend, NULL);
        
        /* Calculate execution time and performance */
        double igemm_time = tend.tv_sec + tend.tv_usec / 1.0e6 - params.start_time_gemm;
        double flops = (FLOPS_GEMM(params.N, params.N, params.nsnp) * 1e-9)/igemm_time;
        
        /* Print performance results for this run */
        SYNC_TIME_PRINT(params.rank, ("hicma_parsec_gemm nodes %d gpus %d N %d SNP %d lookahead %d\t : %lf Gflop/s\n",
                    params.nodes, params.gpus, params.N, params.nsnp, params.lookahead, flops));
    }

    /* Clean up allocated memory */
    parsec_memory_free_tile(parsec, (parsec_tiled_matrix_t*)&A, &params, 0); 
    parsec_memory_free_tile(parsec, (parsec_tiled_matrix_t*)&AA, &params, 0); 

    /* Finalize HiCMA and PaRSEC */
    hicma_parsec_fini( parsec, argc, argv, &params, &params_kernel, &data, &analysis );

    return 0;
}


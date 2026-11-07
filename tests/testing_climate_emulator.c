/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2023-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "climate_emulator.h"

/**
 * @brief Validates climate emulator configuration parameters
 * 
 * @param params HiCMA parameters
 * @param gb Climate emulator structure
 * @return 0 if validation passes, non-zero otherwise
 */
static int validateClimateEmulatorConfig(hicma_parsec_params_t *params, climate_emulator_struct_t *gb) {
    if (params == NULL || gb == NULL) {
        return 1;
    }
    
    // Check if matrix dimensions are consistent
    if (params->N != gb->L * gb->L) {
        if (0 == params->rank) {
            fprintf(stderr, RED "Fatal error: N (-N) %d should be L (-K) %d * L %d\n" RESET, 
                    params->N, gb->L, gb->L);
        }
        return 1;
    }
    
    return 0;
}

/**
 * @brief Performs forward spherical harmonic transform (SHT) operations
 * 
 * @param parsec PaRSEC context
 * @param gb Climate emulator structure
 * @param params HiCMA parameters
 */
static void performForwardSHT(parsec_context_t *parsec, climate_emulator_struct_t *gb, hicma_parsec_params_t *params) {
    // Forward SHT using pre-computed version
    SYNC_TIME_START();
#if !READ_FROM_FILE
    climate_emulator_geqsht_forward_pre_computed_version(parsec, gb, params); 
#endif
    SYNC_TIME_PRINT(params->rank, ("geqsht_forward_pre_computed_version: %.2lf Gflop/s\n", 
                                   gb->flops_forward/sync_time_elapsed/1.0e9));
    gb->time_forward = sync_time_elapsed;
    
    // Forward SHT reshape operation
    SYNC_TIME_START();
    climate_emulator_geqsht_forward_reshape(parsec, gb, params);
    SYNC_TIME_PRINT(params->rank, ("geqsht_forward_reshape\n"));
    gb->time_forward_reshape = sync_time_elapsed;
}

/**
 * @brief Performs matrix operations including SYRK and norm calculation
 * 
 * @param parsec PaRSEC context
 * @param gb Climate emulator structure
 * @param data HiCMA data structure
 * @param params HiCMA parameters
 */
static void performMatrixOperations(parsec_context_t *parsec, climate_emulator_struct_t *gb, 
                           hicma_parsec_data_t *data, hicma_parsec_params_t *params) {
    // Matrix memory allocation
    SYNC_TIME_START();
    if (params->adaptive_memory) {
        // Allocate dense matrix memory
        hicma_parsec_memory_allocation_dense_static(parsec, dplasmaLower, 
                                                   (parsec_tiled_matrix_t *)&data->dcA, "double");
    }
    SYNC_TIME_PRINT(params->rank, ("Matrix memory allocate\n"));
    
    // Perform SYRK operation: C = alpha * A * A^T + beta * C
    SYNC_TIME_START();
    dplasma_dsyrk(parsec, dplasmaLower, dplasmaNoTrans,
                   1.0, (parsec_tiled_matrix_t *)&gb->desc_A,
                   0.0, (!params->adaptive_memory)? 
                        (parsec_tiled_matrix_t *)&data->dcAd : 
                        (parsec_tiled_matrix_t *)&data->dcA);
    SYNC_TIME_PRINT(params->rank, ("SYRK M %d N %d K %d\n", 
                                   gb->desc_A.super.lm, data->dcA.super.ln, gb->desc_A.super.ln));
    gb->time_syrk = sync_time_elapsed;
    params->time_syrk_app = sync_time_elapsed;
    
    // Calculate matrix norm for numerical stability analysis
    SYNC_TIME_START();
    climate_emulator_matrix_norm_get(parsec, dplasmaLower, 
            (!params->adaptive_memory) ?
            (parsec_tiled_matrix_t *)&data->dcAd :
            (parsec_tiled_matrix_t *)&data->dcA,
            params, "double"); 
    SYNC_TIME_PRINT(params->rank, ("Matrix norm: norm_global= %le\n", params->norm_global));
}

/**
 * @brief Performs Cholesky factorization with analysis
 * 
 * @param parsec PaRSEC context
 * @param data HiCMA data structure
 * @param params HiCMA parameters
 * @param params_kernel StarSH kernel parameters
 * @param analysis Matrix analysis structure
 */
static int performCholeskyFactorization(parsec_context_t *parsec, hicma_parsec_data_t *data,
                                 hicma_parsec_params_t *params, starsh_params_t *params_kernel,
                                 hicma_parsec_matrix_analysis_t *analysis) {
    int failed = 0;

    // Analyze matrix structure before Cholesky factorization
    hicma_parsec_matrix_pre_analysis(parsec, data, params, params_kernel, analysis);
    
    // Perform HiCMA Cholesky factorization with multiple runs if specified
    for (int i = 0; i < params->nruns; i++) {
        hicma_parsec_potrf(parsec, data, params, analysis);
        if (params->info != 0) {
            failed = 1;
            if (0 == params->rank) {
                fprintf(stderr, "-- Factorization failed (info = %d): leading minor not positive definite\n",
                        params->info);
            }
        }
    }
    
    // Analyze matrix after Cholesky factorization
    hicma_parsec_matrix_post_analysis(parsec, data, params, params_kernel, analysis);

    return failed;
}

/**
 * @brief Performs inverse operations if enabled
 * 
 * @param parsec PaRSEC context
 * @param gb Climate emulator structure
 * @param params HiCMA parameters
 */
static void performInverseOperations(parsec_context_t *parsec, climate_emulator_struct_t *gb,
                             hicma_parsec_params_t *params) {
#if CLIMATE_EMULATOR_ENABLE_INVERSE 
    // Compare with MATLAB reference result
    climate_emulator_diff_double(parsec, &gb->desc_flm, &gb->desc_flmERA);
    
    // Inverse spherical harmonic transform (SHT)
    SYNC_TIME_START();
    climate_emulator_geqsht_inverse_pre_computed_version(parsec, gb, params);
    SYNC_TIME_PRINT(params->rank, ("geqsht_inverse_pre_computed_version\n"));
    gb->time_backward = sync_time_elapsed;
    
    // Calculate Mean Square Error (MSE)
    SYNC_TIME_START();
    climate_emulator_mse(parsec, gb, params);
    SYNC_TIME_PRINT(params->rank, ("mse\n"));
    gb->time_mse = sync_time_elapsed;
#endif
}

/**
 * @brief Main function - entry point of the climate emulator testing program
 * 
 * This program tests the HiCMA climate emulator functionality, performing:
 * 1. Data reading and initialization
 * 2. Forward spherical harmonic transform (SHT)
 * 3. Matrix operations (SYRK, norm calculation)
 * 4. Cholesky factorization with analysis
 * 5. Optional inverse operations and error analysis
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return Exit status (0 for success, non-zero for failure)
 */
int main(int argc, char **argv)
{
    // HiCMA and PaRSEC configuration structures
    hicma_parsec_params_t params;
    starsh_params_t params_kernel;
    hicma_parsec_data_t data;
    hicma_parsec_matrix_analysis_t analysis;
    climate_emulator_struct_t gb;
    
    // PaRSEC context
    parsec_context_t *parsec = NULL;
    
    // Initialize climate emulator and PaRSEC context
    parsec = climate_emulator_init(argc, argv, &params, &params_kernel, &data);
    if (parsec == NULL) {
        fprintf(stderr, "Failed to initialize climate emulator\n");
        return EXIT_FAILURE;
    }
    
    // Read climate data and initialize structures
    SYNC_TIME_START();
    climate_emulator_reading_data(parsec, &gb, &params);
    SYNC_TIME_PRINT(params.rank, ("climate_emulator_reading_data: M %d N %d K %d\n", 
                                   gb.desc_A.super.lm, data.dcA.super.ln, gb.desc_A.super.ln));
    gb.time_init = sync_time_elapsed;
    
    // Validate configuration parameters
    if (validateClimateEmulatorConfig(&params, &gb) != 0) {
        climate_emulator_fini(parsec, &gb, &params);
        hicma_parsec_fini(parsec, argc, argv, &params, &params_kernel, &data, &analysis);
        return EXIT_FAILURE;
    }
    
    // Stage 0: Data preparation and reshaping
    // Note: Future implementation will include:
    // - Reading multiple files to source arrays
    // - Data reshaping using MPI_Alltoall
    // - Operations using desc_flmT
    // - Reshaping data back
    // - Preparing data for each time slot
    
    // Perform forward spherical harmonic transform operations
    performForwardSHT(parsec, &gb, &params);
    
    // Perform matrix operations (SYRK, norm calculation)
    performMatrixOperations(parsec, &gb, &data, &params);
    
    // Perform Cholesky factorization with analysis
    int failed = performCholeskyFactorization(parsec, &data, &params, &params_kernel, &analysis);
    
    // TODO: Future implementation for final output
    // Matrix size: L^2 * 1828
    // Random normal distribution generation
    // TRMM of Cholesky and this matrix
    // Final output to file
    
    // Perform inverse operations if enabled
    performInverseOperations(parsec, &gb, &params);
    
    // Cleanup and finalization
    climate_emulator_fini(parsec, &gb, &params);
    hicma_parsec_fini(parsec, argc, argv, &params, &params_kernel, &data, &analysis);
    
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

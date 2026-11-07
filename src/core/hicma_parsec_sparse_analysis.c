/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"
#include "hicma_parsec_sparse_analysis.h"

/* Configuration flags for sparse matrix analysis */

/* Whether to use OpenMP for parallelization when matrix is very large */
#define USE_OPENMP 0
#if USE_OPENMP
#include <omp.h>
#endif

/* If allocate memory in local_gemm dynamically (1) or statically (0) */
#define DYNAMIC_MEMORY_ALLOCATION 1

/* Define the number of integers allocated for each tile index array
 * First allocation: max(tile_n_index/INT_EACH_TIME_PER_TILE, INT_EACH_TIME_PER_TILE)
 * Subsequent allocations: INT_EACH_TIME_PER_TILE */
#define INT_EACH_TIME_PER_TILE 10

/* Print analyzed results for debugging (1) or not (0) */
#define PRINT_ANALYSED_RESULT 0

/* Print redefined distribution for debugging (1) or not (0) */ 
#define PRINT_REDEFINED_DISTRIBUTION 0

/* Set the number of threads used for analysis based on matrix size */
#define NUM_THREADS_ANALYSIS ((NT-k)/10000)

/* Global array storing process ID for each tile (used for workload balancing) */
extern DATATYPE_ANALYSIS *dist_array;

/**
 * @brief Initialize the sparse matrix analysis structure
 * 
 * This function initializes the analysis structure with default values and
 * performs basic validation checks on the matrix dimensions.
 * 
 * @param[in,out] analysis Pointer to the analysis structure to initialize
 * @param[in] NT Number of tiles in each dimension
 * @param[in] rank Process rank (for error reporting)
 * @return 0 on success
 */
int hicma_parsec_sparse_analysis_init( hicma_parsec_matrix_analysis_t *analysis, int NT, int rank)
{
    /* Validate that the number of tiles does not exceed DATATYPE_ANALYSIS limit */
    if( NT > 65535 ) {
        if( 0 == rank ) fprintf(stderr, RED "The number of tiles is %d > 65535; "
                                            "make sure this right DATATYPE_ANALYSIS is used\n" RESET, NT);
    }

    /* Initialize memory tracking */
    analysis->total_memory = 0LLU;

    /* Initialize density metrics to full density (1.0) */
    analysis->initial_density = 1.0;    /* Initial matrix density */
    analysis->density_trsm = 1.0;       /* TRSM operation density */
    analysis->density_gemm = 1.0;       /* GEMM operation density */

    return 0;
}

/**
 * @brief Free memory allocated for TRSM and SYRK analysis structures
 * 
 * This function deallocates all memory associated with the initial and final
 * rank distributions used in TRSM and SYRK operations.
 * 
 * @param[in,out] analysis Pointer to the analysis structure
 * @param[in] NT Number of tiles in each dimension
 */
void hicma_parsec_sparse_analysis_trsm_syrk_free(
        hicma_parsec_matrix_analysis_t *analysis,
        unsigned long int NT) {

    /* Free TRSM initial distribution arrays */
    for(int i = 0; i < NT-1; i++) {
        free( analysis->trsm_initial[i] );
    }
    free( analysis->trsm_initial );
    free( analysis->trsm_num_initial );
    
    /* Free rank distribution arrays */
    free( analysis->initial_rank );
    free( analysis->final_rank );
}

/**
 * @brief Initialize TRSM and SYRK analysis structures with rank information
 * 
 * This function allocates and initializes the data structures needed for
 * TRSM and SYRK analysis based on the input rank array. It processes the
 * rank information to determine initial and final rank distributions.
 * 
 * @param[in,out] analysis Pointer to the analysis structure
 * @param[in] rank_array Array containing rank information for each tile
 * @param[in] NT Number of tiles in each dimension
 */
void hicma_parsec_sparse_analysis_trsm_syrk_init( 
        hicma_parsec_matrix_analysis_t *analysis,
        int *rank_array,
        unsigned long int NT ) {

    /* Allocate memory for TRSM initial distribution */
    analysis->trsm_initial = (DATATYPE_ANALYSIS **)malloc( NT * sizeof(DATATYPE_ANALYSIS *) );
    for(int i = 0; i < NT-1; i++)
        analysis->trsm_initial[i] = (DATATYPE_ANALYSIS *)malloc( (NT-i+1) * sizeof(DATATYPE_ANALYSIS) );
    analysis->trsm_num_initial = (DATATYPE_ANALYSIS *)calloc( NT, sizeof(DATATYPE_ANALYSIS) );

    /* Allocate memory for rank distribution tracking */
    analysis->initial_rank = (DATATYPE_ANALYSIS *)calloc( NT * NT, sizeof(DATATYPE_ANALYSIS) );
    analysis->final_rank = (uint8_t *)calloc( NT * NT, sizeof(uint8_t) );
    
    /* Process rank array to build initial and final rank distributions */
    for(int j = 0; j < NT-1; j++) {
        int num_tmp = 0;
        for(int i = j+1; i < NT; i++) {
            if( rank_array[j*NT+i] > 0 ) {
                /* Store initial rank information */
                analysis->initial_rank[j*NT+i] = (DATATYPE_ANALYSIS)rank_array[j*NT+i];
                analysis->final_rank[j*NT+i] = (uint8_t)1;
                analysis->initial_density += 1.0;
                
                /* Build TRSM initial distribution */
                analysis->trsm_initial[j][num_tmp] = (DATATYPE_ANALYSIS)i;
                num_tmp += 1;
            }
        }
        analysis->trsm_num_initial[j] = (DATATYPE_ANALYSIS)num_tmp;
    }
    
    /* Calculate initial density (normalized by matrix size) */
    analysis->initial_density = analysis->initial_density / NT / NT * 2;
}


/**
 * @brief Allocate memory and analyze TRSM and SYRK operations for sparse matrices
 * 
 * This function performs the core analysis of TRSM and SYRK operations for sparse
 * matrices. It allocates necessary memory structures and analyzes the computational
 * patterns based on the sparsity level.
 * 
 * @param[in] A Pointer to the tiled matrix structure
 * @param[in,out] analysis Pointer to the analysis structure
 * @param[in] rank_array Array containing rank information for each tile
 * @param[in] rank Process rank
 * @param[in] NT Number of tiles in each dimension
 * @param[in] sparse Sparsity level (0=dense, 1=basic sparse, 2=advanced sparse)
 * @return 0 on success
 */
int hicma_parsec_sparse_analysis_trsm_syrk(  parsec_tiled_matrix_t *A,
        hicma_parsec_matrix_analysis_t *analysis,
        int *rank_array, int rank, unsigned long int NT, int sparse )
{
    /* Initialize initial and final rank distributions */
    hicma_parsec_sparse_analysis_trsm_syrk_init( analysis, rank_array, NT );

    /* Early return for dense matrices - only basic rank info is needed for checks */
    if( sparse < 1 )
        return 0;

    /* Allocate memory for TRSM distribution analysis */
    analysis->trsm = (DATATYPE_ANALYSIS **)malloc( NT * sizeof(DATATYPE_ANALYSIS *) );
    for(int i = 0; i < NT-1; i++)
        analysis->trsm[i] = (DATATYPE_ANALYSIS *)malloc( (NT-i+1) * sizeof(DATATYPE_ANALYSIS) );
    analysis->trsm_num = (DATATYPE_ANALYSIS *)calloc( NT, sizeof(DATATYPE_ANALYSIS) );
    analysis->total_trsm_num = (unsigned long int)0;

#if WORKLOAD_BALANCE == 2 
    /* Allocate memory for row-based workload balancing */
    analysis->row = (DATATYPE_ANALYSIS **)malloc( NT * sizeof(DATATYPE_ANALYSIS *) );
    for(int i = 1; i < NT; i++)
        analysis->row[i] = (DATATYPE_ANALYSIS *)malloc( i * sizeof(DATATYPE_ANALYSIS) );
    analysis->row_num = (DATATYPE_ANALYSIS *)calloc( NT, sizeof(DATATYPE_ANALYSIS) );
#endif

    /* Allocate memory for SYRK local operations per tile */
    analysis->syrk_local = (DATATYPE_ANALYSIS **)malloc( NT * sizeof(DATATYPE_ANALYSIS *) );
    for(int i = 1; i < NT; i++)
        analysis->syrk_local[i] = (DATATYPE_ANALYSIS *)malloc( i * sizeof(DATATYPE_ANALYSIS) );
    analysis->syrk_local_num = (DATATYPE_ANALYSIS *)calloc( NT, sizeof(DATATYPE_ANALYSIS) );
    analysis->total_syrk_num = (unsigned long int)0;

    /* Early return if advanced workload balancing is not needed */ 
    if( sparse < 2 )
        return 0;

    /* Main analysis loop: iterate through each panel factorization step */
    for(int k = 0; k < NT-1; k++) {

        /* Analyze TRSM and SYRK operations for current panel */
        DATATYPE_ANALYSIS trsm_num_tmp = 0;
        for(int m = k+1; m < NT; m++) {
            if( (DATATYPE_ANALYSIS)1 == analysis->final_rank[k*NT+m] ) {
                /* Record TRSM operation (triangular solve) */
                analysis->trsm[k][trsm_num_tmp] = (DATATYPE_ANALYSIS)m;
                trsm_num_tmp += (DATATYPE_ANALYSIS)1;

                /* Record SYRK operation (symmetric rank-k update) */
                DATATYPE_ANALYSIS next_local_syrk_index = analysis->syrk_local_num[m];
                analysis->syrk_local[m][next_local_syrk_index] = (DATATYPE_ANALYSIS)k;
                analysis->syrk_local_num[m] += (DATATYPE_ANALYSIS)1;
                analysis->total_syrk_num += (unsigned long int)1;
            }
        }
        analysis->trsm_num[k] = trsm_num_tmp;
        analysis->total_trsm_num += (unsigned long int)trsm_num_tmp;

        /* Analyze fill-in patterns for GEMM operations */
        for(int i = 1; i <= analysis->trsm_num[k]-1; i++) {
            for(int j = 0; j < i; j++) {
                /* Get tile indices for GEMM operations */
                DATATYPE_ANALYSIS m_index = analysis->trsm[k][i];
                DATATYPE_ANALYSIS n_index = analysis->trsm[k][j];

                /* Update fill-in pattern (new non-zero entries created) */
                analysis->final_rank[n_index*NT+m_index] = (uint8_t)1;
            }
        }
    }

#if WORKLOAD_BALANCE == 2 
    for(int m = 1; m < NT; m++) {
        DATATYPE_ANALYSIS row_num_tmp = 0;
        for(int n = 0; n < m; n++) {
            if( (DATATYPE_ANALYSIS)1 == analysis->final_rank[n*NT+m] ) {
                analysis->row[m][row_num_tmp] = (DATATYPE_ANALYSIS)n;
                row_num_tmp += (DATATYPE_ANALYSIS)1;
            }
        }
        analysis->row_num[m] = row_num_tmp;
    }

#if PRINT_ANALYSED_RESULT
    if( 0 == rank ) {
        printf("Row count\n");
        for(int m = 1; m < NT; m++) {
            for(int n = 0; n < analysis->row_num[m]; n++) {
                printf("%2u ", analysis->row[m][n]);
            }
            printf("\n");
        }
        printf("\n");
    }
#endif
#endif

    return 0;
}

/**
 * @brief Analyze GEMM operations and memory requirements for sparse matrices
 * 
 * This function performs detailed analysis of GEMM (General Matrix Multiply)
 * operations required for sparse matrix factorization. It analyzes memory
 * requirements, workload distribution, and computational patterns.
 * 
 * @param[in] A Pointer to the original tiled matrix structure
 * @param[in] Dist Pointer to the distribution matrix structure
 * @param[in,out] analysis Pointer to the analysis structure
 * @param[in] rank_array Array containing rank information for each tile
 * @param[in] rank Process rank
 * @param[in] NT Number of tiles in each dimension
 * @param[in] sparse Sparsity level (0=dense, 1=basic sparse, 2=advanced sparse)
 * @return 0 on success
 */ 
int hicma_parsec_sparse_analysis_gemm( parsec_tiled_matrix_t *A,
        parsec_tiled_matrix_t *Dist,
        hicma_parsec_matrix_analysis_t *analysis,
        int *rank_array, int rank, unsigned long int NT, int sparse )
{
    /* Allocate memory for GEMM local operations per tile */
    analysis->gemm_local_memory = (DATATYPE_ANALYSIS *)calloc( NT * NT, sizeof(DATATYPE_ANALYSIS) );
    analysis->gemm_local = (DATATYPE_ANALYSIS **)calloc( (size_t)NT * NT, sizeof(DATATYPE_ANALYSIS *) );
    /* Initialize arena indicator to 0 (indicates memory is from arena) */
    analysis->gemm_local_memory_arena_indicator = (uint8_t *)calloc( NT * NT, sizeof(uint8_t) );
    
    /* Initialize GEMM local arrays to NULL */
    for(int i = 2; i < NT; i++) {
        for(int j = 1; j < i; j++) {
            analysis->gemm_local[j*NT+i] = NULL; 
        }
    }
    analysis->gemm_local_num = (DATATYPE_ANALYSIS *)calloc( NT * NT, sizeof(DATATYPE_ANALYSIS) );
    analysis->total_gemm_num = (unsigned long long int)0; 

    /* Arrays to track memory and operation counts per thread */
    int k = 0;
    unsigned long long int *dgemm_memory_per_thread = (unsigned long long int *)calloc(hicma_parsec_max(NUM_THREADS_ANALYSIS, 1), sizeof(unsigned long long int));
    unsigned long long int *dgemm_num_per_thread = (unsigned long long int *)calloc(hicma_parsec_max(NUM_THREADS_ANALYSIS, 1), sizeof(unsigned long long int));

    /* Main analysis loop: iterate through each panel factorization step */
    for(k = 0; k < NT-1; k++) {

        /* Handle basic sparse case (sparse < 2) */
        if( sparse < 2 ) {
            /* Analyze TRSM and SYRK operations for current panel */
            DATATYPE_ANALYSIS trsm_num_tmp = 0;
            for(int m = k+1; m < NT; m++) {
                if( (DATATYPE_ANALYSIS)1 == analysis->final_rank[k*NT+m] ) {
                    /* Record TRSM operation (triangular solve) */
                    analysis->trsm[k][trsm_num_tmp] = (DATATYPE_ANALYSIS)m;
                    trsm_num_tmp += (DATATYPE_ANALYSIS)1;

                    /* Record SYRK operation (symmetric rank-k update) */
                    DATATYPE_ANALYSIS next_local_syrk_index = analysis->syrk_local_num[m];
                    analysis->syrk_local[m][next_local_syrk_index] = (DATATYPE_ANALYSIS)k;
                    analysis->syrk_local_num[m] += (DATATYPE_ANALYSIS)1;
                    analysis->total_syrk_num += (unsigned long int)1;
                }
            }
            analysis->trsm_num[k] = trsm_num_tmp;
            analysis->total_trsm_num += (unsigned long int)trsm_num_tmp;
        }

        /* Determine optimal number of threads for parallel processing */
        int nthreads_set = hicma_parsec_max(1, hicma_parsec_min(NUM_THREADS_ANALYSIS, (int)analysis->trsm_num[k]-1)); 

#if USE_OPENMP
        /* Use OpenMP for parallel processing */
#pragma omp parallel num_threads(nthreads_set)
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
#else
        /* Sequential processing */
        {
            int tid = 0;
            int nthreads = 1;
#endif
            /* Calculate work distribution among threads */
            int nb_rows = (analysis->trsm_num[k] - 1) / nthreads;
            int first_row = 1 + tid * nb_rows;
            int last_row = (tid + 1) * nb_rows;
            if( tid == nthreads-1 )
                last_row = hicma_parsec_max( (tid + 1) * nb_rows, analysis->trsm_num[k]-1 ); 

#if DEBUG
            fprintf(stderr, "k %d nb_trsm %d rank %d tid %d nthreads %d nthread_set %d nb_rows %d, first_row %d, last_row %d\n",
                    k, analysis->trsm_num[k], rank, tid, nthreads, nthreads_set, nb_rows, first_row, last_row);
#endif

            /* Analyze fill-ins and GEMM operations per tile */
            for(int i = first_row; i <= last_row; i++) {
                for(int j = 0; j < i; j++) {
                    /* Get tile indices for GEMM operations */
                    DATATYPE_ANALYSIS m_index = analysis->trsm[k][i];
                    DATATYPE_ANALYSIS n_index = analysis->trsm[k][j];

                    /* Update fill-in pattern (new non-zero entries created) */
                    analysis->final_rank[n_index*NT+m_index] = (uint8_t)1;

                    /* Check if current process owns this tile */
                    if( rank == A->super.rank_of(&A->super, m_index, n_index)
                            || rank == Dist->super.rank_of(&Dist->super, m_index, n_index) ) {

                        DATATYPE_ANALYSIS next_local_gemm_index = analysis->gemm_local_num[n_index*NT+m_index];
#if DYNAMIC_MEMORY_ALLOCATION
                        /* Dynamic memory allocation: reallocate when needed */
                        if( analysis->gemm_local_memory[n_index*NT+m_index] <= next_local_gemm_index ) {
                            /* Reallocate memory with additional space */
                            analysis->gemm_local[n_index*NT+m_index] = (DATATYPE_ANALYSIS *)realloc( analysis->gemm_local[n_index*NT+m_index], 
                                    (next_local_gemm_index + INT_EACH_TIME_PER_TILE) * sizeof(DATATYPE_ANALYSIS) );

                            /* Update memory tracking */
                            analysis->gemm_local_memory[n_index*NT+m_index] += (DATATYPE_ANALYSIS)INT_EACH_TIME_PER_TILE;
                            dgemm_memory_per_thread[tid] += INT_EACH_TIME_PER_TILE;
                        }
#else
                        /* Static memory allocation: allocate once */
                        if( analysis->gemm_local_memory[n_index*NT+m_index] == 0 ) {
                            /* Allocate memory for all potential GEMM operations on this tile */
                            analysis->gemm_local[n_index*NT+m_index] = (DATATYPE_ANALYSIS *)malloc( n_index * sizeof(DATATYPE_ANALYSIS) );

                            /* Update memory tracking */
                            analysis->gemm_local_memory[n_index*NT+m_index] = n_index; 
                            dgemm_memory_per_thread[tid] = n_index; 
                        }
#endif

                        /* Record the iteration where this GEMM operation occurs */
                        analysis->gemm_local[n_index*NT+m_index][next_local_gemm_index] = (DATATYPE_ANALYSIS)k;
                    }

                    /* Count total number of GEMM operations */
                    analysis->gemm_local_num[n_index*NT+m_index] += (DATATYPE_ANALYSIS)1;
                    dgemm_num_per_thread[tid] += (unsigned long long int)1;
                } /* end loop j */
            } /* end loop i */
        } /* end openmp */ 
    } /* end loop k */

    /* Aggregate memory and operation counts from all threads */
    k = 0;
    for(int i = 0; i < hicma_parsec_max(NUM_THREADS_ANALYSIS, 1); i++) {
        analysis->total_memory += dgemm_memory_per_thread[i]; 
        analysis->total_gemm_num += dgemm_num_per_thread[i];
    }

    /* Calculate total memory usage for analysis structures:
     * - initial_rank: NT^2 elements
     * - final_rank: NT^2 * sizeof(uint8_t) = NT^2/2 by default DATATYPE_ANALYSIS
     * - trsm_initial: NT^2/2 elements; trsm_num: NT elements
     * - trsm: NT^2/2 elements; trsm_num: NT elements
     * - syrk: NT^2/2 elements; syrk_num: NT elements
     * - gemm_local_num: NT^2 elements; gemm_local_memory: NT^2 elements
     */
    analysis->total_memory += NT * NT * 4 + NT * 3;
    analysis->total_memory = analysis->total_memory * sizeof(DATATYPE_ANALYSIS);

    /* Calculate computational density metrics */
    analysis->density_trsm = (double)analysis->total_trsm_num / NT / NT * 2;
    analysis->density_gemm = (double)analysis->total_gemm_num / NT / NT / NT * 6;

#if PRINT_ANALYSED_RESULT
    /* Print analysed results */
    if( 0 == rank ) {
        printf("************** Analysis ***************:\n");
        printf("\nInitial ranks:\n");
        for(int i = 0; i < NT; i++) {
            for(int j = 0; j < NT; j++) {
                printf("%2u ", analysis->initial_rank[j*NT+i]);
            }
            printf("\n");
        }

        printf("\nFinal ranks:\n");
        for(int i = 0; i < NT; i++) {
            for(int j = 0; j < NT; j++) {
                printf("%2u ", analysis->final_rank[j*NT+i]);
            }       
            printf("\n");
        }       

        printf("\nTRSM:\n");
        for(int i = 0; i < NT; i++) {
            printf("Iteration %3d has %3u trsm: ", i, analysis->trsm_num[i]);
            for(int j = 0; j < analysis->trsm_num[i]; j++) {
                printf("%3u ", analysis->trsm[i][j]);
            }
            printf("\n");
        } 

        printf("\nSYRK:\n");
        for(int i = 0; i < NT; i++) {
            printf("Tile (%3d, %3d) has %3u syrk: ", i, i, analysis->syrk_local_num[i]);
            for(int j = 0; j < analysis->syrk_local_num[i]; j++) {
                printf("%3u ", analysis->syrk_local[i][j]);
            }
            printf("\n");
        }

            printf("\nGEMM:\n");
        }

        for(int i = 2; i < NT; i++) {
            for(int j = 1; j < i; j++) {
                if( rank == A->super.rank_of(&A->super, i, j) ) {
                    printf("Tile (%3d, %3d) has %3u gemm: ", i, j, analysis->gemm_local_num[j*NT+i]);
                for(int k = 0; k < analysis->gemm_local_num[j*NT+i]; k++) {
                    printf("%3u ", analysis->gemm_local[j*NT+i][k]);
                }
                printf("\n");
            }
        }
    }

    if( 0 == rank ) {
        printf("\n************** Analysis ***************:\n");
        fflush(stdout);
    }
#endif

    /* free memory */
    free( dgemm_memory_per_thread );
    free( dgemm_num_per_thread );

    /* Print analysed results */
    if( 0 == rank ) {
        fprintf(stderr, MAG "Analysis_result: mumber_potrf= %lu number_trsm= %lu number_syrk= %lu number_gemm= %llu "
                   "number_total= %llu initial_density= %lf density_trsm= %lf density_gemm= %lf "
                   "memory_used_for_analysis= %lf GB\n" RESET,
                NT, analysis->total_trsm_num, analysis->total_syrk_num, analysis->total_gemm_num,
                NT + analysis->total_trsm_num + analysis->total_syrk_num + analysis->total_gemm_num,
                analysis->initial_density, analysis->density_trsm, analysis->density_gemm, 
                (double)analysis->total_memory / 1.0e9 );
    }

    return 0;
}

/**
 * @brief Free all memory allocated for sparse matrix analysis
 * 
 * This function deallocates all memory structures used in the sparse matrix
 * analysis, including TRSM, SYRK, and GEMM analysis data.
 * 
 * @param[in] A Pointer to the tiled matrix structure
 * @param[in,out] analysis Pointer to the analysis structure
 * @param[in] NT Number of tiles in each dimension
 * @param[in] rank Process rank
 * @param[in] sparse Sparsity level
 * @param[in] check Whether check mode was used
 */
void hicma_parsec_sparse_analysis_free( parsec_tiled_matrix_t *A,
        hicma_parsec_matrix_analysis_t *analysis,
        int NT, int rank, int sparse, int check)
{
    (void)rank;
    /* Free TRSM and SYRK structures if sparse analysis or check was performed */
    if( sparse || check ) {
        hicma_parsec_sparse_analysis_trsm_syrk_free( analysis, NT );
    }

    /* Early return for dense matrices - only basic rank info was allocated */
    if( sparse < 1 )
        return;

    /* Free TRSM analysis structures */
    for(int i = 0; i < NT-1; i++)
        free( analysis->trsm[i] ); 
    free( analysis->trsm ); 
    free( analysis->trsm_num ); 

    /* Free SYRK analysis structures */
    for(int i = 1; i < NT; i++)
        free( analysis->syrk_local[i] ); 
    free( analysis->syrk_local ); 
    free( analysis->syrk_local_num ); 

    /* Free the GEMM lists: they are allocated for the tiles owned by A or by
     * the balanced distribution (Dist), so free every non-NULL entry (the
     * array is calloc'd) rather than guessing the owner here */
    for(size_t idx = 0; idx < (size_t)NT * NT; idx++) {
        free( analysis->gemm_local[idx] );
    }
    free( analysis->gemm_local );
    free( analysis->gemm_local_memory ); 
    free( analysis->gemm_local_num ); 
    free( analysis->gemm_local_memory_arena_indicator );

#if WORKLOAD_BALANCE == 2 
    /* Free row-based workload balancing structures */
    for(int i = 1; i < NT; i++)
        free( analysis->row[i] );
    free( analysis->row );
    free( analysis->row_num );
#endif
}


/**
 * @brief Calculate optimized process distribution for sparse matrix tiles
 * 
 * This function redistributes tiles among processes to balance workload
 * for sparse matrix operations. Different distribution strategies are
 * available based on the WORKLOAD_BALANCE configuration.
 * 
 * @param[in] A Pointer to the tiled matrix structure
 * @param[in,out] dist_array Array to store new process assignments
 * @param[in] analysis Pointer to the analysis structure
 * @param[in] P Number of process rows
 * @param[in] Q Number of process columns
 * @param[in] band_size_dist Size of the band for distribution
 * @param[in] rank Process rank
 * @param[in] NT Number of tiles in each dimension
 * @return 0 on success
 */
int hicma_parsec_sparse_dist_calculate( parsec_tiled_matrix_t *A,
        DATATYPE_ANALYSIS *dist_array, hicma_parsec_matrix_analysis_t *analysis,
        int P, int Q, int band_size_dist, int rank, int NT ) {

    int local_p, local_q, process_id;

/* Workload balancing strategies:
 * 0: no balance (default distribution)
 * 1: diamond distribution for off-band tiles; best performance
 * 2: row-based balance, a new 2dbcdd in row of round-robin
 * 3: column-based balance, a new 2dbcdd in row of round-robin
 * 4: column-based balance, as balanced as possible but with minimal
 *    communication overhead
 */
#if WORKLOAD_BALANCE == 1 
    for( int k = 0; k < NT-band_size_dist; k++) {
        if( analysis->trsm_num[k] < 1 ) continue; 
        local_q = k % Q;
        for( int m = 0; m < analysis->trsm_num[k]; m++) {
            /* Real tile row index */
            int m_index = analysis->trsm[k][m];

            /* Skip if within band_size_dist */
            if( m_index-k < band_size_dist )
                continue;

            /* New process ID */
            local_p = (m_index-k-band_size_dist) % P;
            process_id = local_p * Q + local_q;

            dist_array[k*NT+m_index] = process_id;
        }
    }

#elif WORKLOAD_BALANCE == 2
    for( int m = 1; m < NT; m++) {
        if( analysis->row_num[m] < 1 ) continue;
        local_p = m % P;
        for( int n = 0; n < analysis->row_num[m]; n++) {
            /* Real tile row index */
            int n_index = analysis->row[m][n];

            /* Skip if within band_size_dist */
            if( m-n_index < band_size_dist )
                continue;

            int offset = analysis->row[m][0];
            local_q = (n+offset) % Q;
            process_id = local_p * Q + local_q;
            dist_array[n_index*NT+m] = process_id;
        }
    }

#elif WORKLOAD_BALANCE == 3
    for( int k = 0; k < NT-band_size_dist; k++) {
        if( analysis->trsm_num[k] < 1 ) continue;
        local_q = k % Q;
        for( int m = 0; m < analysis->trsm_num[k]; m++) {
            /* Real tile row index */
            int m_index = analysis->trsm[k][m];

            /* Skip if within band_size_dist */
            if( m_index-k < band_size_dist )
                continue;

            if( analysis->trsm[k][0] == k + 1 )
                local_p = (m-1) % P;
            else
                local_p = m % P;

            /* New process ID */
            process_id = local_p * Q + local_q;

            dist_array[k*NT+m_index] = process_id;
        }
    }

#elif WORKLOAD_BALANCE == 4
    int *count = (int *)malloc( P * sizeof(int) );
    for( int k = 0; k < NT-band_size_dist; k++) {
        /* No trsm in this column */
        if( analysis->trsm_num[k] < 1 ) continue;

        local_q = k % Q;

        /* Reset count */
        memset(count, 0, P*sizeof(int));
        int current = 0;

        /* It should be limit or limit+1 if index < left */
        int limit, left;

        if( analysis->trsm[k][0] == k + 1 ) {
            limit = (analysis->trsm_num[k]-1) / P;
            left = (analysis->trsm_num[k]-1) % P;
        } else {
            limit = analysis->trsm_num[k] / P;
            left = analysis->trsm_num[k] % P;
        }

        for( int m = 0; m < analysis->trsm_num[k]; m++) {
            /* Real tile row index */
            int m_index = analysis->trsm[k][m];

            /* Skip if within band_size_dist */
            if( m_index-k < band_size_dist )
                continue;

            int count_index = m_index % P;

            /* Check whether needs to rest process id */
            if( count_index < left ) { 

                /* Not need to reset */
                if( count[count_index] + 1 <= limit+1 ) {
                    count[count_index]++;
                    local_p = m_index % P;

                } else {
                    /* Check which to rest */ 
                    while( (current < left && count[current] >= limit+1)
                            || (current >= left && count[current] >= limit) )
                        current++;

                    if( current >= P )
                        fprintf(stderr, "Wrong : %d !!!\n", current);

                    count[current]++;
                    local_p = current;
                }

            } else {

                /* Not need to reset */
                if( count[count_index] + 1 <= limit ) {
                    count[count_index]++;
                    local_p = m_index % P;

                } else {
                    /* Check which to rest */
                    while( (current < left && count[current] >= limit+1)
                            || (current >= left && count[current] >= limit) )
                        current++;

                    if( current >= P )
                        fprintf(stderr, "Wrong : %d !!!\n", current);

                    count[current]++;
                    local_p = current;
                }

            }

#if DEBUG
            if( 0 == rank ) {
                fprintf(stderr, "k %d m_index %d left %d limit %d count_index %d count[count_index] %d current %d count[current] %d\n",
                        k, m_index, left, limit, count_index, count[count_index], current, count[current]);
            }
#endif

            /* New process ID */
            process_id = local_p * Q + local_q;
            dist_array[k*NT+m_index] = process_id;
        }
    }

#else
    for( int k = 0; k < NT-band_size_dist; k++) {
        if( analysis->trsm_num[k] < 1 ) continue;
        local_q = k % Q;
        for( int m = 0; m < analysis->trsm_num[k]; m++) {
            /* Real tile row index */
            int m_index = analysis->trsm[k][m];

            /* Skip if within band_size_dist */
            if( m_index-k < band_size_dist )
                continue;

            local_p = m_index % P;

            /* New process ID */
            process_id = local_p * Q + local_q;

            dist_array[k*NT+m_index] = process_id;
        }
    }

#endif


#if PRINT_REDEFINED_DISTRIBUTION 
    sleep(2);
    if( 0 == rank ) {
        fprintf(stderr, "\n\nInitial rank:\n");
        for(int i = 0; i < NT; i++) {
            for(int j = 0; j < NT; j++)
                fprintf(stderr, "%u ", analysis->initial_rank[j*NT+i]);
            fprintf(stderr, "\n");
        }

        fprintf(stderr, "\n\nFinal rank:\n");
        for(int i = 0; i < NT; i++) {
            for(int j = 0; j < NT; j++)
                fprintf(stderr, "%u ", analysis->final_rank[j*NT+i]);
            fprintf(stderr, "\n");
        }

        fprintf(stderr, "\n\nMatrix distribution:\n");
        for(int i = 0; i < NT; i++) {
            for(int j = 0; j < NT; j++) {

                if( i < j ) {
                    fprintf(stderr, "%2d ", -1);
                    continue;
                }

                if( 0 == j )
                    fprintf(stderr, "%2d ", i%P);

                if( i-j < band_size_dist )
                    fprintf(stderr, RED "%2d " RESET, j%(P*Q));
                else if( analysis->final_rank[j*NT+i] == 0 )
                    fprintf(stderr, "%2d ", -2);
                else
                    fprintf(stderr, BLU "%2u " RESET, dist_array[j*NT+i]);
            }
            fprintf(stderr, "\n");
        }
    }

    sleep(2);

#endif

    return 0;
}

/**
 * @brief Custom rank calculation for symmetric two-dimensional block cyclic distribution
 * 
 * This function provides a custom rank calculation for symmetric two-dimensional
 * block cyclic distribution, specifically designed for off-band tiles in sparse
 * matrix computations. It overrides the default rank calculation to optimize
 * workload distribution for sparse matrices.
 * 
 * @param[in] desc PaRSEC data collection descriptor
 * @param[in] ... Variable arguments containing tile coordinates (m, n)
 * 
 * @return Process rank responsible for the specified tile coordinates
 * 
 * @note This function uses variable arguments to match PaRSEC's rank_of interface
 * @note Specifically designed for symmetric block cyclic distribution
 * @note Optimized for off-band tile distribution in sparse matrices
 */
uint32_t hicma_parsec_sparse_balance_off_band_rank_of(parsec_data_collection_t * desc, ...)
{
    int m, n;
    va_list ap;
    parsec_matrix_sym_block_cyclic_t * dc;
    dc = (parsec_matrix_sym_block_cyclic_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->super.i / dc->super.mb;
    n += dc->super.j / dc->super.nb;

    assert( m < dc->super.mt );
    assert( n < dc->super.nt );

    return dist_array[n*dc->super.lmt+m]; 
}

/**
 * @brief Calculate the number of local tasks for each process with new distribution
 * 
 * This function calculates the number of local tasks that each process will handle
 * under the new distribution scheme. It accounts for the sparse matrix structure
 * and the custom distribution strategy to ensure balanced workload across processes.
 * 
 * @param[in] A Input tiled matrix
 * @param[in] Dist Distribution matrix containing process assignments
 * @param[in] analysis Sparse matrix analysis structure
 * @param[in] rank Current process rank
 * @param[in] NT Total number of tiles in each dimension
 * @param[in] sparse Sparse matrix flag (1 if sparse, 0 if dense)
 * @param[out] nb_local_tasks Pointer to store the number of local tasks
 * 
 * @return 0 on success
 * 
 * @note This function performs MPI synchronization to ensure consistent results
 * @note The calculation considers both diagonal and off-diagonal tiles
 * @note Used for workload balancing in sparse matrix computations
 */
int hicma_parsec_sparse_nb_tasks( parsec_tiled_matrix_t *A,
        parsec_tiled_matrix_t *Dist,
        hicma_parsec_matrix_analysis_t *analysis,
        int rank, int NT, int sparse, uint32_t *nb_local_tasks )
{

    MPI_Barrier( MPI_COMM_WORLD );

    /* Init to 0 */
    *nb_local_tasks = 0U;

    /* Diagonal */ 
    for(int n = 0; n < NT; n++) {
        //if( rank == Dist->super.rank_of(&Dist->super, n, n) ) {
        if( rank == A->super.rank_of(&A->super, n, n) ) {
            /* POTRF */
            *nb_local_tasks += 1;

            /* SYRK */
            *nb_local_tasks += analysis->syrk_local_num[n];
        }
    }

    /* Off diagonal */
    for(int n = 0; n < NT-1; n++) {

        /* if no trsm in this panel factorization */
        if( analysis->trsm_num[n] < 1)
            continue;

        /* Iterate the initial rank distribution for TRSM_READ and GEMM_READ */
        for(int mi = 0; mi < analysis->trsm_num_initial[n]; mi++) {
            int m = analysis->trsm_initial[n][mi];
            if( sparse > 1 ) {
                if( rank == A->super.rank_of(&A->super, m, n) ) {
                    /* TRSM READ */
                    *nb_local_tasks += 1;

                    /* GEMM READ */
                    if( m != 1 && n != 0 )
                        *nb_local_tasks += 1;
                }
            } /* if sparse > 1 */
        }

        /* Iterate the final rank distribution */
        for(int mi = 0; mi < analysis->trsm_num[n]; mi++) {
            int m = analysis->trsm[n][mi];
            if( sparse < 2 ) {
                if( rank == A->super.rank_of(&A->super, m, n) ) {
                    /* TRSM */
                    *nb_local_tasks += 1; 

                    /* GEMM */
                    *nb_local_tasks += analysis->gemm_local_num[n*NT+m];
                }
            } else {
                if( rank == Dist->super.rank_of(&Dist->super, m, n) ) {
                    /* TRSM */
                    *nb_local_tasks += 1; 

                    /* GEMM */
                    *nb_local_tasks += analysis->gemm_local_num[n*NT+m];
                }

                if( rank == A->super.rank_of(&A->super, m, n) ) {
                    /* TRSM_WRITE*/
                    if( 0 != analysis->final_rank[n*NT+m] )
                        *nb_local_tasks += 1;
                }
            } /* if sparse < 2 */
        }
    }

    MPI_Barrier( MPI_COMM_WORLD );
#if PRINT_REDEFINED_DISTRIBUTION
    unsigned long int total_tasks = 0UL;
    MPI_Reduce( nb_local_tasks, &total_tasks, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD );
    fprintf(stderr, BLU "Number_ranks: Rank %d has %u tasks; Total_tasks %lu\n" RESET, rank, *nb_local_tasks, total_tasks);
#endif

    return 0;
}


/**
 * @brief Reset and adapt maximum rank parameters for sparse matrix computations
 * 
 * This function resets and adapts the maximum rank parameters (maxrank and compmaxrank)
 * based on the current matrix characteristics and sparse analysis results. It handles
 * automatic band size adjustment and rank optimization for sparse matrices.
 * 
 * @param[in,out] params HICMA PaRSEC parameters structure to update
 * 
 * @return 0 on success
 * 
 * @note This function may modify maxrank and compmaxrank based on sparse analysis
 * @note Handles automatic band size adjustment when imaxrk_auto_band is set
 * @note Updates rank parameters for optimal sparse matrix performance
 * @note Only rank 0 prints verbose information to avoid duplicate output
 */
int hicma_parsec_adaptive_maxrank( hicma_parsec_params_t *params )
{
    int rank = params->rank;
    int verbose = params->verbose;
    int maxrank, compmaxrank;
    int adaptive_maxrank = params->adaptive_maxrank;
    int imaxrk = params->imaxrk;
    int sparse = params->sparse;
    
    /* If auto_band, reset imaxrk */
    if( params->imaxrk_auto_band != 0 )
        imaxrk = params->imaxrk_auto_band;

    /* Reset the maxrank and compmaxrank */
    if( adaptive_maxrank && imaxrk > 0 ) {
        if( 1 == adaptive_maxrank ) {
            maxrank = (int)(imaxrk * 1.5 );
            compmaxrank = (int)(imaxrk * 2);
        } else {
            maxrank = (int)(imaxrk * adaptive_maxrank );
            compmaxrank = (int)(imaxrk * adaptive_maxrank + imaxrk);
        }

        params->maxrank = hicma_parsec_min( maxrank, params->maxrank );
        params->genmaxrank = hicma_parsec_min( maxrank, params->genmaxrank );
        params->compmaxrank = hicma_parsec_min( compmaxrank, params->compmaxrank);

        if( 0 == rank && verbose ) fprintf(stderr, GRN "reset maxrank and compmaxrank: maxrank= %d compmaxrank= %d\n" RESET, params->maxrank, params->compmaxrank);
    }

    return 0;
}

/**
 * @brief Binary search for task position in analysis array
 * 
 * This function performs a binary search to find the position of a value
 * in a sorted array. Used for locating specific tasks in the analysis data.
 * 
 * @param[in] array Sorted array to search in
 * @param[in] value Value to search for
 * @param[in] length Length of the array
 * @param[in] index1 First index for error reporting
 * @param[in] index2 Second index for error reporting
 * @return Index of the value if found, -1 if not found
 */
int binary_search_index( DATATYPE_ANALYSIS *array, DATATYPE_ANALYSIS value, int length, int index1, int index2 ) {
    int first = 0;
    int last = length - 1;
    int middle = (first + last) / 2;
    
    while (first <= last) {
        if (array[middle] < value)
            first = middle + 1;
        else if (array[middle] == value) {
            return middle;
        }
        else
            last = middle - 1;
        middle = (first + last) / 2;
    }
    
    /* Value not found - report error */
    if (first > last) {
        fprintf(stderr, RED "Not found! %d is not present in the list: %d %d.\n" RESET, value, index1, index2);
        return -1;
    }
    return 0;
}

/* ============================================================================
 * Main sparse analysis function
 * ============================================================================ */

/**
 * @brief Perform comprehensive sparse matrix analysis for workload optimization
 * 
 * This function performs a complete analysis of the sparse matrix structure
 * to optimize workload distribution and memory allocation for Cholesky factorization.
 * It includes initialization, TRSM/SYRK analysis, GEMM analysis, and distribution
 * calculations based on the sparsity level and matrix characteristics.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in,out] data HICMA data structure containing matrix descriptors
 * @param[in] params HICMA parameters including sparsity settings
 * @param[in,out] analysis Matrix analysis structure to populate
 * @return 0 on success, non-zero on failure
 * 
 * @note This function is the main entry point for sparse matrix analysis
 * @note Results are used for workload balancing and memory optimization
 */
int hicma_parsec_sparse_analysis( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t *analysis )
{
    int rank = params->rank;
    int nodes = params->nodes;
    int band_p = params->band_p;
    int P = params->P;
    int NT = params->NT;
    int band_size_dist = params->band_size_dist;
    int band_size_dense = params->band_size_dense;
    int uplo = params->uplo;
    int sparse = params->sparse;
    int adaptive_maxrank = params->adaptive_maxrank;
    int imaxrk = params->imaxrk;
    int check = params->check;

    /* Initialize the analysis structure */
    hicma_parsec_sparse_analysis_init( analysis, NT, rank );

    /*
     * Perform TRSM and SYRK analysis if sparse processing or check mode is enabled
     * This allocates memory structures and analyzes computational patterns
     * Check mode is needed in hicma_parsec_matrix_uncompress.jdf
     */
    if( sparse || check )
        hicma_parsec_sparse_analysis_trsm_syrk( (parsec_tiled_matrix_t*)&data->dcA, analysis, params->rank_array, rank, NT, sparse );

    if( sparse ) {
        /* Validate sparse processing parameters */
        assert( band_size_dist > 0 );
        assert( band_size_dense == 1 );

        /* Initialize off-band distribution matrix */
        parsec_matrix_sym_block_cyclic_init(&data->dcDist.off_band, PARSEC_MATRIX_INTEGER,
                rank, 1, 1, NT, NT, 0, 0,
                NT, NT, P, nodes/P, uplo);

        /* Initialize band distribution matrix (must match dcA distribution) */
        parsec_matrix_block_cyclic_init(&data->dcDist.band, PARSEC_MATRIX_INTEGER, PARSEC_MATRIX_TILE,
                rank, 1, 1, band_size_dist, NT, 0, 0,
                band_size_dist, NT, band_p, nodes/band_p,
                1, 1, 0, 0);

        /* Initialize the combined band/off-band structure */
        parsec_matrix_sym_block_cyclic_band_init( &data->dcDist, nodes, rank, band_size_dist );
        parsec_data_collection_set_key((parsec_data_collection_t*)&data->dcDist, "dcDist_off_band");
        parsec_data_collection_set_key(&data->dcDist.band.super.super, "dcDist_band");

        /* Calculate optimized process distribution for workload balancing */
        dist_array = (DATATYPE_ANALYSIS *)calloc(NT * NT, sizeof(DATATYPE_ANALYSIS));
        hicma_parsec_sparse_dist_calculate( (parsec_tiled_matrix_t*)&data->dcA,
                dist_array, analysis, P, nodes/P, band_size_dist, rank, NT );

        /* Set the new distribution function for off-band tiles */
        data->dcDist.off_band.super.super.rank_of = hicma_parsec_sparse_balance_off_band_rank_of;

        /* Perform detailed GEMM analysis for memory and workload optimization */
        hicma_parsec_sparse_analysis_gemm( (parsec_tiled_matrix_t*)&data->dcA,
                (parsec_tiled_matrix_t*)&data->dcDist,
                analysis, params->rank_array, rank, NT, sparse );

        /* Calculate the number of local tasks for each process */
        hicma_parsec_sparse_nb_tasks( (parsec_tiled_matrix_t*)&data->dcA,
                (parsec_tiled_matrix_t*)&data->dcDist, analysis, rank, NT, sparse, &params->nb_local_tasks );

        /* Ensure dcRank has the same affinity as TRSM and GEMM operations */
        data->dcRank.off_band.super.super.rank_of = hicma_parsec_sparse_balance_off_band_rank_of;
    }

    /* Adaptively adjust maximum rank based on analysis results */
    if( sparse && adaptive_maxrank && imaxrk > 0 )
        hicma_parsec_adaptive_maxrank( params );

    return 0;
}

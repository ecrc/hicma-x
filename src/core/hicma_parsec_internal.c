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

#ifdef PARSEC_HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef PARSEC_HAVE_LIMITS_H
#include <limits.h>
#endif
#if defined(PARSEC_HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(PARSEC_HAVE_GETOPT_H) */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

#ifdef GSL
#include <gsl/gsl_errno.h>
#endif

/*******************************
 * Global Variables and Constants *
 *******************************/

/**
 * @brief Global timing variables for performance measurement
 * 
 * These variables track execution time for different phases of the computation:
 * - time_elapsed: Total computation time including all phases (generation, factorization, etc.)
 * - sync_time_elapsed: Time spent in synchronization operations and communication overhead
 * 
 * These are used for performance analysis and benchmarking of the HICMA algorithms.
 * The timing is typically measured using high-resolution timers and accumulated
 * across multiple runs for statistical accuracy.
 */
double time_elapsed = 0.0;
double sync_time_elapsed = 0.0;

/**
 * @brief Array of problem type identifiers for HICMA test matrices
 * 
 * This array maps problem type indices to their string representations.
 * Each string corresponds to a specific kernel or problem type that can be
 * used in HICMA computations. These problem types represent different
 * mathematical kernels commonly used in scientific computing applications.
 * 
 * The array is indexed by the kind_of_problem parameter and used for:
 * - Matrix generation with specific mathematical properties
 * - Benchmarking different kernel types
 * - Validation of algorithms across various problem domains
 */
char* str_problem[NB_PROBLEM]={
"randtlr",                           /* Random TLR (Tile Low-Rank) matrix for general testing */
"ed-2d-sin",                         /* 2D exponential decay with sine function */
"st-2d-sqexp",                       /* 2D space-time squared exponential kernel */
"st-3d-sqexp",                       /* 3D space-time squared exponential kernel */
"st-3d-exp",                         /* 3D space-time exponential kernel */
"ed-3d-sin",                         /* 3D exponential decay with sine function */
"md-3d-virus",                       /* 3D molecular dynamics virus simulation */
"md-3d-cube",                        /* 3D molecular dynamics cube simulation */
"matern-2d",                         /* 2D Matern covariance kernel */
"space-time-matern-2d",              /* 2D space-time Matern kernel */
"space-time-matern-real-2d",         /* 2D space-time Matern real-valued kernel */
"space-time-matern-real-exageo-2d",  /* 2D space-time Matern real exageo kernel */
"matern-2d-real",                    /* 2D Matern real-valued kernel */
"matern-2d-real-exageo",             /* 2D Matern real exageo kernel */
"krr_random_gene",                   /* Kernel Ridge Regression random gene data */
"krr_gene"                           /* Kernel Ridge Regression gene data */
};

/**
 * @brief Global array for distributed analysis data
 * 
 * This array stores analysis data for each tile in the distributed computation.
 * It is used to track process IDs and other tile-specific information during
 * the sparse analysis phase of HICMA computations.
 * 
 * The array is allocated during the sparse analysis initialization and contains
 * metadata about each tile's distribution, including:
 * - Process ID assignments
 * - Tile ownership information
 * - Communication patterns
 * - Memory layout details
 */
DATATYPE_ANALYSIS *dist_array;

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
/**
 * @brief Cleanup function for CUDA handles
 * 
 * This function is called during cleanup to properly destroy CUDA library handles.
 * It releases cuBLAS and cuSOLVER handles and frees the associated memory.
 * This is essential for preventing resource leaks in GPU-accelerated computations.
 * 
 * @param _h Pointer to dplasma_cuda_handles_t structure containing CUDA handles
 * @param _n Unused parameter (for compatibility with callback signature)
 * 
 * @note This function should not be called if DPLASMA GPU support is disabled
 * @note Proper cleanup is critical for avoiding CUDA context leaks
 */
static void destroy_cuda_handles(void *_h, void *_n)
{
    // Note: This function should not be called if DPLASMA GPU support is disabled
    dplasma_cuda_handles_t *handles = (dplasma_cuda_handles_t*)_h;
    (void)_n;  // Suppress unused parameter warning
    
    // Destroy cuBLAS handle - releases GPU memory and context resources
    cublasDestroy_v2(handles->cublas_handle);
    
    // Destroy cuSOLVER handle - releases solver-specific GPU resources
    cusolverDnDestroy(handles->cusolverDn_handle);
    
    // Free the handles structure - releases host memory
    free(handles);
}
#endif

/**
 * @brief Macro for conditional matrix allocation in PaRSEC
 * 
 * This macro generates code to allocate a matrix descriptor if the condition is true.
 * It creates a matrix descriptor of the specified type, initializes it with the given
 * parameters, allocates memory for the matrix data, and sets a collection key.
 * 
 * The macro is used to conditionally allocate matrix descriptors based on runtime
 * conditions, which is useful for memory optimization and avoiding unnecessary
 * allocations when certain matrices are not needed.
 * 
 * @param DC Matrix descriptor variable name (will be declared as TYPE##_t)
 * @param COND Condition for allocation (evaluated at runtime)
 * @param TYPE Matrix type (e.g., parsec_matrix_sym_block_cyclic)
 * @param INIT_PARAMS Initialization parameters for the matrix descriptor
 * 
 * @note The macro calculates memory size based on local tiles, block size, and data type
 * @note A collection key is set using the variable name for debugging and tracking
 */
#define PASTE_CODE_ALLOCATE_MATRIX(DC, COND, TYPE, INIT_PARAMS)      \
    TYPE##_t DC;                                                     \
    if(COND) {                                                          \
        TYPE##_init INIT_PARAMS;                                        \
        DC.mat = parsec_data_allocate((size_t)DC.super.nb_local_tiles * \
                                        (size_t)DC.super.bsiz *      \
                                        (size_t)parsec_datadist_getsizeoftype(DC.super.mtype)); \
        parsec_data_collection_set_key((parsec_data_collection_t*)&DC, #DC);          \
    }

/**
 * @brief Configuration option structure for improved argument management
 * 
 * This structure defines a configuration option with its name, description,
 * data type, and target memory location for storing the parsed value.
 * It is used by the improved argument parsing system to provide a more
 * maintainable and extensible way to handle command-line options.
 * 
 * The structure supports different data types and provides a centralized
 * way to define all available options with their descriptions and targets.
 */
typedef struct {
    const char *name;        /**< Option name (e.g., "-N", "--matrix-size") */
    const char *description; /**< Human-readable description of the option */
    int type;                /**< Data type: 0=int, 1=double, 2=string, 3=flag */
    void *target;            /**< Pointer to variable where value will be stored */
} config_option_t;

/**
 * @brief Helper function to parse argument values
 * 
 * Parses a string argument and converts it to the appropriate data type,
 * storing the result in the target memory location. This function handles
 * type conversion with proper error checking and memory management.
 * 
 * @param arg String argument to parse (must not be NULL)
 * @param type Data type to convert to (0=int, 1=double, 2=string, 3=flag)
 * @param target Pointer to memory location where parsed value will be stored
 * @return 0 on success, -1 on parsing error or invalid type
 * 
 * @note For string types, the function handles memory allocation and deallocation
 * @note For flag types, the value is always set to 1 (true)
 * @note The function validates that the entire string was consumed during parsing
 */
static int parse_arg_value(const char *arg, int type, void *target) {
    char *endptr;
    
    switch(type) {
        case 0: // int - parse as base-10 integer
            *(int*)target = strtol(arg, &endptr, 10);
            return (*endptr == '\0') ? 0 : -1;
        case 1: // double - parse as floating-point number
            *(double*)target = strtod(arg, &endptr);
            return (*endptr == '\0') ? 0 : -1;
        case 2: // string - duplicate the string
            if (*(char**)target) free(*(char**)target);
            *(char**)target = strdup(arg);
            return 0;
        case 3: // flag - set to 1 (true)
            *(int*)target = 1;
            return 0;
        default:
            return -1; // Invalid type
    }
}

/**
 * @brief Helper function to find option by name
 * 
 * Searches through an array of configuration options to find the one
 * with the specified name. This is used by the argument parser to
 * locate the appropriate option structure for a given command-line argument.
 * 
 * @param options Array of configuration options (must be NULL-terminated)
 * @param name Name of the option to find (case-sensitive)
 * @return Pointer to the matching option structure, or NULL if not found
 * 
 * @note The options array must be terminated with a NULL name entry
 * @note String comparison is case-sensitive
 */
static config_option_t* find_option(config_option_t *options, const char *name) {
    for (int i = 0; options[i].name != NULL; i++) {
        if (strcmp(options[i].name, name) == 0) {
            return &options[i];
        }
    }
    return NULL;
}

/**
 * @brief Check if a string represents a valid negative number
 * 
 * This function determines if a string starting with '-' is a negative number
 * rather than an option flag. It checks if the string can be parsed as a valid
 * integer or floating-point number.
 * 
 * @param str String to check
 * @return 1 if the string is a valid negative number, 0 otherwise
 */
static int is_negative_number(const char *str) {
    if (str[0] != '-') {
        return 0;
    }
    
    // Check if it's a valid integer or floating-point number
    char *endptr;
    strtol(str, &endptr, 10);
    if (*endptr == '\0') {
        return 1; // Valid integer
    }
    
    strtod(str, &endptr);
    if (*endptr == '\0') {
        return 1; // Valid floating-point number
    }
    
    return 0; // Not a valid number
}

/**
 * @brief Improved command line argument parsing function
 * 
 * This function provides a more maintainable and extensible way to parse
 * command line arguments for HICMA. It uses a table-driven approach with
 * the config_option_t structure to define all available options.
 * 
 * The function handles:
 * - Long options with -- prefix
 * - Option-value pairs (--option=value or --option value)
 * - Flag options (--option without value)
 * - Help message generation
 * - Error handling and validation
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @param params Pointer to HICMA parameters structure to populate
 * @return 0 on success, -1 on parsing error
 * 
 * @note This function skips PaRSEC/MPI specific arguments
 * @note Help is displayed and program exits if --help is requested
 */
static int parse_arguments_parsing(int argc, char **argv, hicma_parsec_params_t *params) {
    config_option_t options[] = {
        // PaRSEC specific options
        {"cores", "Number of concurrent threads (default: number of physical hyper-threads)", 0, &params->cores},
        {"gpus", "Number of GPUs (default: 0)", 0, &params->gpus},
        {"grid_rows", "Rows (P) in the PxQ process grid (default: NP)", 0, &params->P},
        {"grid_cols", "Columns (Q) in the PxQ process grid (default: NP/P)", 0, &params->Q},
        {"N", "Dimension (N) of the matrices (required)", 0, &params->N},
        {"NB", "Tile size (required)", 0, &params->NB},
        {"check", "Verify the results", 3, &params->check},
        {"verbose", "Extra verbose output: 1=normal, >1=more verbose about decisions", 0, &params->verbose},
        {"help", "Show this help message", 3, NULL},
        
        // HiCMA options
        {"fixedrank", "Fixed rank threshold used in recompression stage of HCORE_GEMM", 0, &params->fixedrk},
        {"fixedacc", "Fixed accuracy threshold used in recompression stage of HCORE_GEMM", 1, &params->fixedacc},
        {"maxrank", "Maxrank limit used in creation of descriptors", 0, &params->maxrank},
        {"genmaxrank", "Maxrank limit used in generation", 0, &params->genmaxrank},
        {"compmaxrank", "Maxrank limit used in allocation of buffers for HiCMA_dpotrf operation", 0, &params->compmaxrank},
        {"adddiag", "Add this number to diagonal elements to make the matrix positive definite", 1, &params->add_diag},
        {"lookahead", "Set lookahead, from -1 to NT-1; default -1, will set to auto_tuned band_size_dense", 0, &params->lookahead},
        {"kind_of_problem", "Problem type", 0, &params->kind_of_problem},
        {"send_full_tile", "Send full tile instead of compressed", 0, &params->send_full_tile},
        {"auto_band", "Auto select the most suitable band size", 0, &params->auto_band},
        {"sparse", "Sparse mode: 0=no sparse, 1=band distribution, 2=diamond-shaped distribution", 0, &params->sparse},
        {"band_dense_dp", "Dense band double precision", 0, &params->band_size_dense_dp},
        {"band_dense_sp", "Dense band single precision", 0, &params->band_size_dense_sp},
        {"band_dense_hp", "Dense band half precision", 0, &params->band_size_dense_hp},
        {"band_dense", "Dense band size, default 1: only diagonal is dense", 0, &params->band_size_dense},
        {"band_low_rank_dp", "Low rank band double precision", 0, &params->band_size_low_rank_dp},
        {"band_dist", "If 0: normal two_dim_block_cyclic; if >0: with band distribution", 0, &params->band_size_dist},
        {"band_p", "Row process grid on band, default 1", 0, &params->band_p},
        {"adaptive_decision", "0: disabled; ~0: adaptive_decision of each tile's format using norm approach", 0, &params->adaptive_decision},
        {"adaptive_memory", "0: memory allocated once; 1: memory reallocated per tile after precision decision", 0, &params->adaptive_memory},
        {"adaptive_maxrank", "In -D 6, adaptively set the maxrank used in Cholesky based on the generation", 0, &params->adaptive_maxrank},
        {"kind_of_cholesky", "Cholesky type", 0, &params->kind_of_cholesky},
        {"mesh_file", "Path to mesh file", 2, &params->mesh_file},
        {"rbf_kernel", "Type of RBF basis function (0:Gaussian, 1:Expon, 2:InvQUAD, 3:InvMQUAD, 4:Maternc1, 5:Maternc2, 6:TPS, 7:CTPS, 8:Wendland)", 0, &params->rbf_kernel},
        {"radius", "Radius of influential nodes", 1, &params->radius},
        {"order", "No, Morton, or Hilbert ordering (0, 1, or 2, respectively)", 0, &params->order},
        {"density", "Density of sphere packing for rbf application. Use -1 for random distribution", 1, &params->density},
        {"tensor_gemm", "Bitmask of GEMM compute modes: 0x1=FP64, 0x2=FP32, 0x4=TF32, 0x8=TF16 A16_B16_C32_OP32, 0x10=TF16 A16_B16_C16_OP16, 0x20=BF16 A16_B16_C32_OP32, 0x40=BF16 A16_B16_C16_OP16, 0x80=TF16 A16_B16_C16_OP32", 0, &params->tensor_gemm},
        {"datatype_convert", "Convert datatype: 0=receiver convert always, 1=sender convert always (only in DENSE_MP_BAND), 2=adaptive", 0, &params->datatype_convert},
        {"band_size_termination", "Band size auto-tuning termination on GPU, default 3.0", 1, &params->band_size_auto_tuning_termination},
        {"left_looking", "Modified left-looking Cholesky (row)", 0, &params->left_looking},
        {"nruns", "The number of runs of Cholesky", 0, &params->nruns},
        
        // Kernel parameters
        {"time_slots", "Time slots in kernels", 0, &params->time_slots},
        {"sigma", "Sigma in kernels", 1, &params->sigma},
        {"beta", "Beta in kernels", 1, &params->beta},
        {"nu", "Nu in kernels", 1, &params->nu},
        {"beta_time", "Beta time in kernels", 1, &params->beta_time},
        {"nu_time", "Nu time in kernels", 1, &params->nu_time},
        {"nonsep_param", "Nonsep param in kernels", 1, &params->nonsep_param},
        {"noise", "Noise in kernels", 1, &params->noise},
        
        // Special cases
        {"HNB", "Inner NB used for recursive algorithms (default: MB)", 0, &params->HNB},
        {"nsnp", "Number of SNPs (number of snps for each individual)", 0, &params->nsnp},
        
        {"numobj", "Number of objects (number of viruses within a population)", 0, &params->numobj},
        {"latitude", "Latitude parameter for climate modeling", 0, &params->latitude},
        {"wavek", "Wave number for electrodynamics problem", 1, &params->wave_k},
        {"pheno_file", "Phenotype file path", 2, &params->pheno_file},
        {"rhs", "Right-hand side vector", 0, &params->RHS},
        
        {NULL, NULL, 0, NULL} // End marker
    };
    
    int help_requested = 0;
    
    // First pass: handle help and special cases
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help_requested = 1;
            break;
        }
    }
    
    if (help_requested) {
        printf("HiCMA PaRSEC Options:\n");
        printf("Usage: %s [options]\n\n", argv[0]);
        printf("Options:\n");
        for (int i = 0; options[i].name != NULL; i++) {
            printf("  --%-20s %s\n", options[i].name, options[i].description);
        }
        
        printf("\nProblem Types (--kind_of_problem):\n");
        for (int i = 0; i < NB_PROBLEM; i++) {
            printf("  %-2d %s\n", i, str_problem[i]);
        }
        
        printf("\nCholesky Variants (--kind_of_cholesky):\n");
        printf("  %-2d DENSE_TLR_MP          - Dense TLR mixed precision (dense band + low-rank off-band)\n", DENSE_TLR_MP);
        printf("  %-2d DENSE_TLR_DP          - Dense TLR double precision (full double precision)\n", DENSE_TLR_DP);
        printf("  %-2d DENSE_MP_BAND         - Dense mixed precision band (mixed precision in band)\n", DENSE_MP_BAND);
        printf("  %-2d DENSE_SP_HP_BAND      - Dense single/half precision band (SP/HP in band)\n", DENSE_SP_HP_BAND);
        printf("  %-2d DENSE_MP_GPU          - Dense mixed precision GPU (GPU-accelerated mixed precision)\n", DENSE_MP_GPU);
        printf("  %-2d SPARSE_TLR_DP_GENERAL - Sparse TLR double precision general (sparse + low-rank)\n", SPARSE_TLR_DP_GENERAL);
        printf("  %-2d SPARSE_TLR_DP_BALANCE - Sparse TLR double precision balanced (workload-balanced sparse)\n", SPARSE_TLR_DP_BALANCE);
        printf("  %-2d DENSE_MP_GPU_FP8      - Dense mixed precision GPU FP8 (FP8 precision on GPU)\n", DENSE_MP_GPU_FP8);
        printf("  %-2d DENSE_MP_GPU_FP8_SP   - Dense mixed precision GPU FP8 single (FP8 + SP on GPU)\n", DENSE_MP_GPU_FP8_SP);
        
        printf("\nSpecial options:\n");
        printf("  --help               Show this help message\n");
        printf("  --check              Enable correctness check\n");
        printf("  --verbose            Set verbosity level\n");
        exit(0);
    }
    
    // Second pass: parse all arguments
    for (int i = 1; i < argc; i++) {
        char *arg = argv[i];
        char *value = NULL;
        
        // Skip if it's a PaRSEC/MPI argument
        // TODO
        if (strncmp(arg, "--parsec", 8) == 0 || 
            strncmp(arg, "--mca", 5) == 0 ||
            strncmp(arg, "--bind-to", 9) == 0 ||
            strncmp(arg, "--map-by", 8) == 0 ||
            strncmp(arg, "--rank-by", 9) == 0 ||
            strncmp(arg, "--report-bindings", 17) == 0 ||
            strncmp(arg, "--display-map", 13) == 0 ||
            strncmp(arg, "--display-allocation", 20) == 0 ||
            strncmp(arg, "--display-devel-map", 19) == 0 ||
            strncmp(arg, "--display-devel-allocation", 26) == 0 ||
            strncmp(arg, "--display-topo", 14) == 0 ||
            strncmp(arg, "--display-cache", 15) == 0 ||
            strncmp(arg, "--display-partition", 19) == 0 ||
            strncmp(arg, "--display-reorder", 17) == 0 ||
            strncmp(arg, "--display-map", 13) == 0 ||
            strncmp(arg, "--display-allocation", 20) == 0 ||
            strncmp(arg, "--display-devel-map", 19) == 0 ||
            strncmp(arg, "--display-devel-allocation", 26) == 0 ||
            strncmp(arg, "--display-topo", 14) == 0 ||
            strncmp(arg, "--display-cache", 15) == 0 ||
            strncmp(arg, "--display-partition", 19) == 0 ||
            strncmp(arg, "--display-reorder", 17) == 0) {
            continue;
        }
        
        // Handle long options (--option=value or --option value)
        if (strncmp(arg, "--", 2) == 0) {
            // Special case: handle the "--" delimiter
            if (strcmp(arg, "--") == 0) {
                // Stop parsing HICMA arguments, let PaRSEC handle the rest
                break;
            }
            
            char *option_name = arg + 2;
            char *equals = strchr(option_name, '=');
            
            if (equals) {
                *equals = '\0';
                value = equals + 1;
            } else if (i + 1 < argc && (argv[i + 1][0] != '-' || is_negative_number(argv[i + 1]))) {
                value = argv[i + 1];
                i++; // Skip next argument
            }
            
            config_option_t *opt = find_option(options, option_name);
            if (opt) {
                if (opt->type == 3) { // Flag
                    parse_arg_value("1", opt->type, opt->target);
                } else if (value) {
                    if (parse_arg_value(value, opt->type, opt->target) != 0) {
                        fprintf(stderr, "Error: Invalid value '%s' for option '--%s'\n", 
                                value, option_name);
                        return -1;
                    }
                } else {
                    fprintf(stderr, "Error: Option '--%s' requires a value\n", option_name);
                    return -1;
                }
            } else {
                fprintf(stderr, "Error: Unknown option '--%s'\n", option_name);
                return -1;
            }
        }
        // Handle short options (-o=value or -o value)
        else if (arg[0] == '-' && arg[1] != '-') {
            fprintf(stderr, "Error: Short options (like -%c) are no longer supported. Use --%s instead.\n", arg[1], arg + 1);
            return -1;
        }
    }
    
    return 0;
}

/**
 * @brief Parse command line arguments and initialize MPI
 * 
 * This function initializes MPI (if available), parses command line arguments,
 * and sets up the HICMA parameters structure with default values and parsed options.
 * It serves as the main entry point for argument processing and system initialization.
 * 
 * The function performs the following operations:
 * 1. Initialize MPI with thread support (if available)
 * 2. Set default parameter values for all HICMA options
 * 3. Parse command line arguments using the improved parser
 * 4. Validate and adjust parameters based on system capabilities
 * 5. Configure GPU settings and process grid
 * 
 * @param _argc Pointer to argument count (modified by MPI_Init_thread)
 * @param _argv Pointer to argument vector (modified by MPI_Init_thread)
 * @param params Pointer to HICMA parameters structure to initialize
 * 
 * @note This function may exit the program if critical parameters are missing
 * @note MPI initialization requires MPI_THREAD_MULTIPLE support
 */
void parse_arguments(int *_argc, char*** _argv, hicma_parsec_params_t *params)
{
    // Default: assume we did not initialize MPI
    params->mpi_initialized_by_hicma = 0;
    // Initialize MPI with thread support if available
    // This enables multiple threads per process for better performance
#ifdef PARSEC_HAVE_MPI
    {
        int already_initialized = 0;
        MPI_Initialized(&already_initialized);
        if (!already_initialized) {
            int provided;
            MPI_Init_thread(_argc, _argv, MPI_THREAD_MULTIPLE, &provided);
            params->mpi_initialized_by_hicma = 1;
        } else {
            params->mpi_initialized_by_hicma = 0;
        }
    }
    // Get MPI communicator information for process grid setup
    MPI_Comm_size(MPI_COMM_WORLD, &params->nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &params->rank);
#else
    // Single process execution - no MPI available
    params->nodes = 1;
    params->rank = 0;
    params->mpi_initialized_by_hicma = 0;
#endif

    // Local variables for argument processing
    extern char **environ;
    int rc;
    int argc = *_argc;
    char **argv = *_argv;
    char *value;

    /* ===========================================
     * Set default parameter values
     * =========================================== */
    
    // Hardware configuration - these control resource utilization
    params->cores = -1;         // Number of cores per node (-1 = auto-detect from system)
    params->gpus  = 0;          // Number of GPUs to use (0 = CPU only, >0 enables GPU acceleration)
    
    // Band size configuration for different precision types
    // Band size controls the width of the dense diagonal band in the matrix
    params->band_size_dense = INT_MAX;      // Band size for dense computation (INT_MAX = full matrix)
    params->band_size_dist = -1;            // Band size for distributed computation (defaults to band_size_dense)
    params->band_p = 1;                     // Row process grid size for band computation
    params->band_size_dense_dp = -1;        // Double precision band size (defaults to band_size_dense)
    params->band_size_dense_sp = -1;        // Single precision band size (defaults to band_size_dense)
    params->band_size_dense_hp = -1;        // Half precision band size (defaults to band_size_dense)
    params->band_size_low_rank_dp = -1;     // Low-rank double precision band size (defaults to NT)
    
    // Adaptive computation settings - control dynamic behavior
    params->adaptive_decision = 0;          // Disable adaptive tile format decision (0=disabled, >0=enabled)
    // TODO: Need to test the overhead and restructure memory allocation strategy
    params->adaptive_memory = 0;            // Enable adaptive memory allocation per tile (1=enabled, 0=disabled)
    params->lookahead = -1;                 // Lookahead depth (auto-tuned based on band_size_dense)
    
    // Problem configuration - define the computational problem
    params->kind_of_problem = 2;            // Default: statistics-2d-sqexp problem (see str_problem array)
    params->send_full_tile  = 0;            // Disable full tile sending (0=compressed, 1=full tiles)
    params->HNB = 300;                      // Subtile size for recursive algorithms (inner block size)
    params->auto_band = 0;                  // Disable automatic band size tuning (0=manual, 1=auto)
    params->reorder_gemm = 0;               // Disable GEMM reordering (0=disabled, 1=enabled)
    params->kind_of_cholesky = DENSE_MP_GPU; // Default Cholesky type: Dense Mixed Precision
    
    // Matrix dimensions (must be set by user via command line)
    params->P = 0;                          // Row process grid (auto-set to number of nodes if 0)
    params->N = 0;                          // Matrix size N (must be specified, typically square matrices)
    params->NB = 0;                         // Tile size (must be specified, affects memory usage and performance)
    params->M = 0;                          // Matrix size M (must be specified, defaults to N if not set)
    params->K = 0;                          // Matrix size K (must be specified, defaults to N if not set)
    params->RHS = 1;                        // Number of right-hand side vectors (for linear systems)
    
    // Algorithm parameters - control numerical behavior and accuracy
    params->verbose = 0;                    // Disable verbose output (0=quiet, >0=increasing verbosity)
    params->add_diag = 0.0;                 // Diagonal regularization term (adds to diagonal for numerical stability)
    params->wave_k = 50;                    // Wave number for synthetic 2D applications (electrodynamics)
    params->fixedacc = 1.0e-8;              // Fixed accuracy threshold (yields ~1.0e-9 accuracy in practice)
    params->maxrank = 0;                    // Maximum rank (auto-set to tile_size/2 if 0)
    params->genmaxrank = 0;                 // Maximum rank for generation (defaults to IPARAM_MAX_RANK)
    params->compmaxrank = 0;                // Maximum rank for computation (defaults to IPARAM_MAX_RANK)
    params->density = 0.0;                  // Matrix density (0.0 = dense, >0 = sparse with given density)
    params->sparse = 0;                     // Sparse computation flag (0 = dense, 1 = band, 2 = diamond)
    params->adaptive_maxrank = 0;           // Disable adaptive maximum rank (0=disabled, >0=enabled)
    
    // GPU and tensor computation settings - control GPU acceleration
    params->tensor_gemm = MASK_FP64 | MASK_FP32 | MASK_TF16_A16_B16_C32_OP32; // Supported tensor GEMM types
    params->datatype_convert = 0;           // Disable datatype conversion on sender side (0=receiver, 1=sender, 2=adaptive)
    params->fixedrk = 0;                    // Disable fixed rank computation (0=adaptive, >0=fixed rank)
    params->band_size_auto_tuning_termination = FLUCTUATION; // Auto-tuning termination criterion
    
    // Validation and execution settings - control testing and execution
    params->check = 0;                      // Disable correctness checking (0=disabled, 1=enabled)
    params->left_looking = 0;               // Disable left-looking Cholesky (0=right-looking, 1=left-looking)
    params->nruns = 1;                      // Number of execution runs (for performance measurement) 

    /* ===========================================
     * Kernel-specific parameters
     * =========================================== */
    
    // 2D Matern space-time kernel parameters - control kernel function behavior
    params->time_slots = 1;                 // Number of time slots (for space-time kernels)
    params->sigma = 1.0;                    // Standard deviation parameter (controls kernel width)
    params->beta = 0.03;                    // Spatial correlation parameter (controls spatial decay)
    params->nu = 1.0;                       // Smoothness parameter (controls kernel smoothness)
    params->beta_time = 1.0;                // Temporal correlation parameter (controls temporal decay)
    params->nu_time = 0.5;                  // Temporal smoothness parameter (controls temporal smoothness)
    params->nonsep_param = 0.5;             // Non-separability parameter (controls space-time coupling)
    
    // Object and location parameters - for specific problem types
    params->numobj = 0;                     // Number of objects (IPARAM_NUMOBJ, for molecular dynamics)
    params->latitude = 0;                   // Latitude coordinate (for climate modeling applications)
    
    // Noise and regularization - control numerical stability
    params->noise = 0.1;                    // Noise level in kernel computation (adds numerical noise)
    
    // File I/O - external data sources
    params->mesh_file = NULL;               // Mesh file path (initialized to NULL, used for mesh-based problems)

    /* Parse arguments using improved parser */
    if (parse_arguments_parsing(argc, argv, params) != 0) {
        if (params->rank == 0) {
            fprintf(stderr, "Error parsing arguments\n");
        }
        exit(1);
    }

    // Set verbose flag for rank 0 only (avoid duplicate output in parallel execution)
    int verbose = params->rank ? 0 : params->verbose;

    /* GPU Configuration - setup GPU acceleration if requested */
    if(params->gpus < 0) params->gpus = 0;  // Ensure non-negative GPU count

    // Configure PaRSEC GPU environment variables
    rc = asprintf(&value, "%d", params->gpus);
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    parsec_setenv_mca_param( "device_cuda_enabled", value, &environ );
#endif

#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    parsec_setenv_mca_param( "device_hip_enabled", value, &environ );
#endif
    free(value);

    // Enable GPU debugging output if verbose mode is enabled
    if(params->gpus > 0 && params->verbose > 1) {
        parsec_setenv_mca_param( "device_show_capabilities", "1", &environ );
        parsec_setenv_mca_param( "device_show_statistics", "1", &environ );
    }

    /* Process Grid Validation and Setup */
    // Ensure process grid dimensions are valid and consistent with number of nodes
    if( 0 == params->P
            || params->Q * params->P != params->nodes ) {
        params->Q = hicma_parsec_process_grid_calculation( params->nodes );
        params->P = params->nodes / params->Q;
    }
    if( 0 == params->rank ) {
        fprintf(stderr, BLU "Process grid set to %d X %d\n" RESET, params->P, params->Q);
    }

    /* Matrix Dimension Validation and Default Assignment */
    // Matrix size N is required for most computations
    if( params->N <= 0 ) {
        if( 0 == params->rank )
            fprintf(stderr, "Matrix size (N) is not set; set to 2048!\n");
        params->N = 2048;  // Set minimal default for testing
    }

    // Tile size NB is required for tiled algorithms
    if( params->NB <= 0 ) {
        if( 0 == params->rank )
            fprintf(stderr, "Tile size (NB) is not set; set to 2048!\n");
        params->NB = 2048;  // Set minimal default for testing
    }

    // Matrix size M defaults to N if not specified
    if( params->M <= 0 ) {
        if( 0 == params->rank && params->verbose )
            fprintf(stderr, "Matrix size (M) is not set; Automatically set to N!\n");
        params->M = params->N;
    }

    // Matrix size K defaults to N if not specified
    if( params->K <= 0 ) {
        if( 0 == params->rank && params->verbose )
            fprintf(stderr, "Matrix size (K) is not set; Automatically set to N!\n");
        params->K = params->N;
    }

    /* Diagonal Regularization Setup */
    // Set diagonal regularization to matrix size if not specified (for numerical stability)
    if( params->add_diag < 0.0 )
        params->add_diag = (double)params->N;

    /* Rank Parameter Validation and Default Assignment */
    // Set maximum rank to half of tile size if not specified (common heuristic)
    if( params->maxrank <= 0 ) {
        params->maxrank = params->NB / 2;
        if( 0 == params->rank )
            fprintf(stderr, YEL "Max rank has not been specified. Forced to half of the size of a tile %d\n" RESET,
                    params->maxrank);
    }

    // Set generation maxrank to maxrank by default (for matrix generation phase)
    if( params->genmaxrank <= 0 )
        params->genmaxrank = params->maxrank;

    // Set computation maxrank to maxrank by default (for factorization phase)
    if( params->compmaxrank <= 0 )
        params->compmaxrank = params->maxrank;

    /* Band Size Configuration for Different Precision Types */
    // Set double precision band size based on Cholesky type
    if( params->band_size_dense_dp < 0 ) { 
        if( DENSE_SP_HP_BAND == params->kind_of_cholesky
                || DENSE_MP_GPU_FP8_SP == params->kind_of_cholesky )
            params->band_size_dense_dp = 0;  // No double precision band for these types
        else
            params->band_size_dense_dp = params->band_size_dense;  // Use general band size
    }

    // Set single precision band size (defaults to general band size)
    if( params->band_size_dense_sp < 0 ) {
        params->band_size_dense_sp = params->band_size_dense;
        if( 0 == params->rank ) {
            fprintf( stderr, MAG "Set band_size_dense_sp = band_size_dense\n" RESET );
        }
    }

    // Set half precision band size (defaults to general band size)
    if( params->band_size_dense_hp < 0 ) {
        params->band_size_dense_hp = params->band_size_dense;
        if( 0 == params->rank ) {
            fprintf( stderr, MAG "Set band_size_dense_hp = band_size_dense\n" RESET );
        }
    }

    // Set low-rank double precision band size (defaults to number of tiles)
    if( params->band_size_low_rank_dp < 1 )
        params->band_size_low_rank_dp = ceil( (double)params->N / params->NB );
    
    /* Adaptive Decision Validation */
    // Ensure adaptive decision is only enabled for compatible Cholesky types
    if( params->adaptive_decision != 0
            && !(params->kind_of_cholesky == DENSE_TLR_MP
                || params->kind_of_cholesky == DENSE_MP_GPU
                || params->kind_of_cholesky == DENSE_MP_GPU_FP8
                || params->kind_of_cholesky == DENSE_MP_GPU_FP8_SP
                || params->kind_of_cholesky == DENSE_SP_HP_BAND)
        ) {
        if( 0 == params->rank ) {
            fprintf( stderr, RED "Wrong Cholesky version is selected!\n" RESET );
        }
        exit(1);
    }

    /* Kernel Parameter Default Adjustment */
    // Adjust default kernel parameters for certain problem types (0-7)
    if( params->kind_of_problem <= 7
            && params->sigma == 1.0  
            && params->beta == 0.03
            && params->nu == 1.0 ) {
        params->sigma = 1.0;    // Standard deviation
        params->beta = 0.1;     // Spatial correlation (increased from 0.03)
        params->nu = 0.5;       // Smoothness parameter (decreased from 1.0)
    }

    (void)rc;
}

/**
 * @brief Print parameter summary at the beginning of execution
 * 
 * This function displays a summary of the key parameters that will be used
 * for the HICMA computation. It only prints on rank 0 to avoid duplicate
 * output in parallel execution.
 * 
 * @param params Pointer to HICMA parameters structure
 * 
 * @note Only rank 0 prints to avoid duplicate output in parallel execution
 * @note Output includes hardware configuration, matrix dimensions, and algorithm settings
 */
static void print_arguments(hicma_parsec_params_t *params)
{
    // Only print on rank 0 to avoid duplicate output in parallel execution
    int verbose = params->rank ? 0 : params->verbose;

	if( verbose ) {
		fprintf(stderr, "#+++++ cores detected       : %d\n", params->cores);

		fprintf(stderr, "#+++++ nodes x cores + gpu  : %d x %d + %d (%d+%d)\n"
				"#+++++ P x Q                : %d x %d (%d/%d)\n",
				params->nodes,
				params->cores,
				params->gpus,
				params->nodes * params->cores,
				params->nodes * params->gpus,
				params->P, params->Q,
				params->Q * params->P, params->nodes);

#if GENOMICS
		fprintf(stderr, "#+++++ M x N                : %d x %d\n",
				params->N, params->nsnp);

		fprintf(stderr, "#+++++ M x RHS              : %d x %d\n",
				params->N, params->RHS);
#else
        fprintf(stderr, "#+++++ M x N                : %d x %d\n",
                params->N, params->N);
#endif

		fprintf(stderr, "#+++++ MB x NB              : %d x %d\n",
				params->NB, params->NB);
		fprintf(stderr, "#+++++ HMB x HNB            : %d x %d\n", params->HNB, params->HNB);
	}
}

/**
 * @brief Set up PaRSEC runtime environment
 * 
 * This function initializes the PaRSEC runtime system, which provides the
 * task-based parallel execution environment for HICMA computations. It handles
 * argument processing, core detection, GPU setup, and performance monitoring.
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @param params Pointer to HICMA parameters structure
 * @return Pointer to initialized PaRSEC context, or NULL on failure
 * 
 * @note This function may exit the program if initialization fails
 * @note GPU support is automatically detected and configured if available
 */
parsec_context_t* setup_parsec(int argc, char **argv, hicma_parsec_params_t * params)
{
	// Set verbose flag for rank 0 only (avoid duplicate output in parallel execution)
	int verbose = params->verbose;
	if(params->rank > 0) verbose = 0;
   
	/* Start performance timing */
	TIME_START();

	/* Process command line arguments for PaRSEC */
	// Extract PaRSEC-specific arguments (after "--" delimiter)
	int parsec_argc, idx;
	char** parsec_argv = (char**)calloc(argc, sizeof(char*));
	parsec_argv[0] = argv[0];  /* the application name */
	for( idx = parsec_argc = 1;
			(idx < argc) && (0 != strcmp(argv[idx], "--")); idx++);
	if( idx != argc ) {
		for( parsec_argc = 1, idx++; idx < argc;
				parsec_argv[parsec_argc] = argv[idx], parsec_argc++, idx++);
	}

	// Initialize PaRSEC runtime with specified number of cores
	parsec_context_t* ctx = parsec_init(params->cores,
			&parsec_argc, &parsec_argv);

	free(parsec_argv);
	if( NULL == ctx ) {
		/* Failed to correctly initialize PaRSEC runtime.
		 * In a production scenario, this should be reported upstream,
		 * but in this case we bail out immediately.
		 */
		exit(-1);
	}

	/* Core Detection and Performance Monitoring Setup */
	// If the number of cores was not specified, detect from PaRSEC context
	if(params->cores <= 0)
	{
		int p, nb_total_comp_threads = 0;
		// Count total computational threads across all virtual processes
		for(p = 0; p < ctx->nb_vp; p++) {
			nb_total_comp_threads += ctx->virtual_processes[p]->nb_cores;
		}
		params->cores = nb_total_comp_threads;

        /* Performance Monitoring Arrays - track operations per core */
        // Band operations (operations within the dense diagonal band)
        params->op_band = (unsigned long *)calloc(params->cores, sizeof(unsigned long));
        // Off-band operations (operations outside the dense diagonal band)
        params->op_offband = (unsigned long *)calloc(params->cores, sizeof(unsigned long));
        // Path operations (operations along the critical path)
        params->op_path = (unsigned long *)calloc(params->cores, sizeof(unsigned long));
        // Off-path operations (operations not on the critical path)
        params->op_offpath = (unsigned long *)calloc(params->cores, sizeof(unsigned long));

        /* Timing Arrays - track execution time per core */
        // Gather time in JDF (Job Data Flow) execution
        params->gather_time = (double *)calloc(params->cores, sizeof(double));
        params->gather_time_tmp = (double *)calloc(params->cores, sizeof(double));
    }

    // Print parameter summary for verification
    print_arguments(params);

    /* Parameter Validation */
    // Ensure critical parameters are set (matrix size and tile size)
    if( 0 == params->N || 0 == params->NB )
        exit(1);

    /* GPU Detection and Configuration */
    int nb_gpus_check = 0;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    int *dev_index;
    // Detect available GPU devices
    hicma_parsec_find_cuda_devices( &dev_index, &nb_gpus_check);
    free( dev_index );

    /* GPU Memory Information and Setup */
    if( nb_gpus_check > 0 ) {
        size_t free, total;
        // Query GPU memory information
        cudaMemGetInfo( &free, &total );
        params->gpu_total_memory = (double)total/1.0e9;  // Convert to GB
        params->gpu_free_memory = (double)free/1.0e9;    // Convert to GB
        VERBOSE_PRINT(params->rank, params->verbose, (BLU "GPU MEMORY: FREE %lf GB ; TOTAL %lf GB\n" RESET, params->gpu_free_memory, params->gpu_total_memory));

        /* CUDA Library Initialization */
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        /* cublasInit() is deprecated in newer cuBLAS versions, 
           cuBLAS is now initialized automatically when creating handles */
        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
        // Register CUDA handles for proper cleanup
        parsec_info_register(&parsec_per_stream_infos, "DPLASMA::CUDA::HANDLES",
                             destroy_cuda_handles, NULL,
                             dplasma_create_cuda_handles, NULL,
                             NULL);
#endif

        // Check GPU architecture compatibility
        hicma_parsec_check_gpu_arch( params );
    }

#endif
    params->gpus = hicma_parsec_min(nb_gpus_check, params->gpus);
    if(params->verbose > 1) {
        fprintf(stderr, RED "Rank %d: Update nb_gpus from input %d to real %d\n" RESET, params->rank, params->gpus, params->gpus);
    }

    params->gpu_cols = hicma_parsec_process_grid_calculation( params->gpus );
    params->gpu_rows = params->gpus / params->gpu_cols;

    /* Set default lookahead */
    // TODO: this is for dense
#if !GENOMICS
    if(-1 == params->lookahead && params->band_size_dense >= params->NT && 0 == params->auto_band) {
        params->lookahead = params->gpu_rows * params->P;
        if( params->rank == 0 ) {
            fprintf(stderr, RED "Set lookahead to %d = %d X %d\n" RESET, params->lookahead, params->gpu_rows, params->P);
        }
    }
#endif

    /* Update CPU performance per node */
    params->cpu_perf *= params->cores;

    /* Calculate the max band_size that could fit in GPU memory */
    hicma_parsec_find_band_size_dense_gpu_memory_max( params );

    /* Check if exceed GPU memory */
    //if( params->band_size_dense_gpu_memory_max < params->band_size_dense && params->gpus > 0 ) {
    //    if( 0 == params->rank )
    //        fprintf(stderr, RED "The current band_size_dense= %d is too big to fit in GPU memory: limit %d\n" RESET, params->band_size_dense, params->band_size_dense_gpu_memory_max);
    //    //exit(0);
    //}

    /* If mixed-precision of DP/SP/HP, disable HP on CPU */
    if( DENSE_MP_BAND == params->kind_of_cholesky ) {
        if( 0 == params->gpus ) {
            params->band_size_dense_sp = params->NT;
            params->band_size_dense_hp = params->NT;
            if( params->rank == 0 && params->verbose) {
                printf(YEL "%d: mixed DP/SP/HP: disable HP on CPU at the beginning\n" RESET, __LINE__);
                fflush(stdout);
            }
        }
    }

    if( verbose ) TIME_PRINT(params->rank, ("PaRSEC initialized\n"));
    return ctx;
}

/**
 * @brief Cleanup PaRSEC context and finalize MPI
 * 
 * This function performs cleanup operations for the PaRSEC runtime system,
 * including CUDA handle cleanup (if available) and MPI finalization.
 * 
 * @param parsec Pointer to PaRSEC context to finalize
 * @param params Pointer to HICMA parameters (unused, for compatibility)
 * 
 * @note This function should be called before program termination
 * @note CUDA handles are automatically cleaned up by the CUDA runtime
 */
void cleanup_parsec(parsec_context_t* parsec, hicma_parsec_params_t *params) 
{
    // Cleanup CUDA handles if CUDA support is available
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    parsec_info_id_t CuHI = parsec_info_lookup(&parsec_per_stream_infos, "DPLASMA::CUDA::HANDLES", NULL);
    parsec_info_unregister(&parsec_per_stream_infos, CuHI, NULL);
    /* Note: cublasShutdown() is deprecated in newer cuBLAS versions.
       cuBLAS is now cleaned up automatically by the CUDA runtime. */
#endif

    // Finalize PaRSEC runtime system - releases all PaRSEC resources
    parsec_fini(&parsec);

    // Finalize MPI only if we initialized it
#ifdef PARSEC_HAVE_MPI
    if (params != NULL && params->mpi_initialized_by_hicma) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            // Ensure all processes are synchronized before finalizing MPI
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
        }
    }
#endif

    // Suppress unused parameter warning
    (void)params;
}


/**
 * @brief Initialize HICMA parameters after argument parsing
 * 
 * This function performs post-parsing initialization of HICMA parameters,
 * including tile count calculations, executable path extraction, and
 * memory allocation for decision arrays and performance tracking.
 * 
 * @param params Pointer to HICMA parameters structure to initialize
 * @param argv Command line arguments (used to extract executable path)
 * @return 0 on success, non-zero on error
 */
int hicma_parsec_params_init(hicma_parsec_params_t *params, char **argv)
{
    /* ===========================================
     * Calculate tile dimensions
     * =========================================== */
    
    // Calculate number of tiles in each dimension (round up if not evenly divisible)
    params->MT = (params->M % params->NB == 0) ? (params->M/params->NB) : (params->M/params->NB + 1);
    params->NT = (params->N % params->NB == 0) ? (params->N/params->NB) : (params->N/params->NB + 1);
    params->KT = (params->K % params->NB == 0) ? (params->K/params->NB) : (params->K/params->NB + 1);    

    /* ===========================================
     * Extract executable path
     * =========================================== */
    
    // Copy full executable path
    // Find the executable name within the path and truncate to get directory
    sprintf(params->exe_file_path, "%s", argv[0]); 

    char *slash = strrchr(params->exe_file_path, '/');
    if (!slash) {
        params->exe_file_path[0] = '\0';          // no '/', nothing before it
    }
    *(slash + 1) = '\0';
    
    /* ===========================================
     * Initialize decision counters
     * =========================================== */
    
    // Counters for different precision decisions
    params->nb_dense_dp = 0.0;        // Number of double precision dense tiles
    params->nb_dense_sp = 0.0;        // Number of single precision dense tiles
    params->nb_dense_hp = 0.0;        // Number of half precision dense tiles
    params->nb_dense_fp8 = 0.0;       // Number of FP8 precision dense tiles
    params->nb_low_rank_dp = 0.0;     // Number of double precision low-rank tiles
    params->nb_low_rank_sp = 0.0;     // Number of single precision low-rank tiles

    /* ===========================================
     * Initialize performance tracking
     * =========================================== */
    
    // CPU performance metrics (per core)
    params->cpu_perf = 0.0;                    // CPU performance baseline
    
    // GPU performance metrics (per GPU)
    params->gpu_perf_nb_nb_nb = 0.0;           // GPU GEMM performance: NB × NB × NB
    params->gpu_perf_nb_nb_r = 0.0;            // GPU GEMM performance: NB × NB × rank
    params->gpu_perf_nb_r_r = 0.0;             // GPU GEMM performance: NB × rank × rank

    // GPU memory information
    params->gpu_free_memory = 0.0;             // Available GPU memory per GPU
    params->gpu_total_memory = 0.0;            // Total GPU memory per GPU

    // GPU band size limitations
    params->band_size_dense_gpu_memory_max = params->NT;    // Memory-limited band size
    params->band_size_dense_gpu_balance_max = params->NT;   // Load balance-limited band size
    params->band_size_dense_gpu_time_max = params->NT;      // Time-limited band size

    /* ===========================================
     * Allocate decision arrays
     * =========================================== */
    
    // Precision decision arrays for different matrices
    params->decisions = (uint16_t *)calloc(params->MT * params->NT, sizeof(uint16_t));      // Main matrix decisions

    // Data type conversion decisions
    params->decisions_send = (uint16_t *)calloc(params->NT * params->NT, sizeof(uint16_t));

    // GPU GEMM type decisions
    params->decisions_gemm_gpu = (uint16_t *)calloc(params->NT * params->NT, sizeof(uint16_t));

    /* ===========================================
     * Allocate norm tracking arrays
     * =========================================== */
    
    // Norm tracking for each tile
    params->norm_tile = (double *)calloc( params->MT * params->NT, sizeof(double) );        // Main matrix norms

    /* ===========================================
     * Initialize problem type strings
     * =========================================== */
    
    // Copy problem type strings to parameter structure
    for(int i = 0; i < NB_PROBLEM; i++) {
        params->str_problem[i] = str_problem[i]; 
    }

    /* ===========================================
     * Initialize timing variables
     * =========================================== */
    
    // Performance timing for different phases
    params->time_starsh = 0.0;              // STARSH kernel generation time
    params->time_hicma = 0.0;               // HICMA computation time
    params->time_opt_band = 0.0;            // Band size optimization time
    params->time_regenerate = 0.0;          // Matrix regeneration time
    params->time_reorder = 0.0;             // Matrix reordering time
    params->time_analysis = 0.0;            // Sparse analysis time
    params->time_decision_kernel = 0.0;     // Kernel decision time
    params->time_decision_sender = 0.0;     // Sender decision time

    /* ===========================================
     * Initialize norm tracking
     * =========================================== */
    
    // Global norm tracking
    params->band_size_norm = 1;             // Band size for norm computation
    params->norm_global = 0.0;              // Global matrix norm
    params->norm_global_diff = 0.0;         // Global norm difference

    /* ===========================================
     * Initialize log-likelihood calculation
     * =========================================== */
    
    // Note: log_det_dp is not initialized here (commented out)
    params->log_det_mp = 0.0;               // Log determinant for mixed precision

    /* ===========================================
     * Initialize rank statistics
     * =========================================== */
    
    // Allocate and initialize rank tracking arrays
    params->rank_array = (int *)malloc(params->NT * params->NT * sizeof(int));
    params->imaxrk = 0;                     // Maximum rank encountered
    params->iminrk = 0;                     // Minimum rank encountered
    params->iavgrk = 0.0;                   // Average rank
    params->imaxrk_auto_band = 0;           // Maximum rank during auto-band tuning
    params->iminrk_auto_band = 0;
    params->iavgrk_auto_band = 0.0;
    params->fmaxrk = 0;
    params->fminrk = 0;
    params->favgrk = 0.0;

    /* flops */
    if( params->cores > 0 ) {
        params->op_band = (unsigned long *)calloc(params->cores, sizeof(unsigned long));
        params->op_offband = (unsigned long *)calloc(params->cores, sizeof(unsigned long));
        params->op_path = (unsigned long *)calloc(params->cores, sizeof(unsigned long));
        params->op_offpath = (unsigned long *)calloc(params->cores, sizeof(unsigned long));
    }
    PASTE_CODE_FLOPS(FLOPS_DPOTRF, ((double)params->N), params->flops);

    /* check results accuracy */
    double result_accuracy = 0.0;

    /* Gather time in JDF */
    if( params->cores > 0 ) {
        params->gather_time = (double *)calloc(params->cores, sizeof(double));
        params->gather_time_tmp = (double *)calloc(params->cores, sizeof(double));
    }
    params->critical_path_time = 0.0;
    params->potrf_time = 0.0;
    params->trsm_time = 0.0;
    params->syrk_time = 0.0;
    params->potrf_time_temp = 0.0;
    params->trsm_time_temp = 0.0;
    params->syrk_time_temp = 0.0;
    params->wrap_potrf = NULL;
    params->wrap_trsm = NULL;
    params->wrap_syrk = NULL;
    params->wrap_gemm = NULL;
    params->wrap_potrf_complete = NULL;
    params->wrap_trsm_complete = NULL;
    params->wrap_syrk_complete = NULL;
    params->wrap_gemm_complete = NULL;

    /* Others */
    params->uplo = PlasmaLower;            /* Only support lower right now */ 
    params->band_size_dist_provided = 1;   /* band_size_dist is provided */
    params->info = 0;
    if(params->gpus > 0) {
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        cudaMallocHost((void**)&params->info_gpu, params->NT * sizeof(int));
#elif defined(PARSEC_HAVE_DEV_HIP_SUPPORT) 
        hipHostMalloc((void **)&params->info_gpu, params->NT * sizeof(int), hipHostMallocDefault);
#else
        fprintf(stderr, RED"Devices is not supported!!!");
#endif
    } else {
        params->info_gpu = (int *)calloc( params->NT, sizeof(int) );
    }

#if PREDICTION 
    params->NP= (params->N * 0.8);
    while(params->NP%params->NB!=0) params->NP++;
#endif    

    return 0;
}


/**
 * @brief Print initial parameter configuration
 * 
 * This function displays the complete parameter configuration at the beginning
 * of execution. It provides a comprehensive overview of all HICMA settings
 * including matrix dimensions, hardware configuration, algorithm parameters,
 * and kernel-specific settings.
 * 
 * @param params Pointer to HICMA parameters structure
 * 
 * @note Only rank 0 prints to avoid duplicate output in parallel execution
 * @note Output includes all major configuration parameters for debugging and verification
 */
void hicma_parsec_params_print_initial( hicma_parsec_params_t *params )
{
    // Only print on rank 0 to avoid duplicate output in parallel execution
    if( 0 == params->rank ) {
        printf("\nM=%d N=%d NB=%d NB=%d HNB=%d HNB=%d\n", params->N, params->N, params->NB, params->NB, params->HNB, params->HNB);
        printf("nodes=%d P=%d Q=%d cores=%d nb_gpus= %d gpu_type= %d verbose= %d\n", params->nodes, params->P, params->Q, params->cores, params->gpus, params->gpu_type, params->verbose);
        printf("kind_of_problem=%d %s\n", params->kind_of_problem, params->str_problem[params->kind_of_problem]);
        printf("fixedacc=%.1e add_diag=%g fixed_rk=%d wave_k=%g\n", params->fixedacc, params->add_diag, params->fixedrk, params->wave_k);
        printf("send_full_tile=%d lookahead= %d adaptive_decision= %d adaptive_memory= %d\n", params->send_full_tile, params->lookahead, params->adaptive_decision, params->adaptive_memory);
        printf("band_size_dist= %d band_size_dense_dp:%d band_size_dense_sp:%d band_size_dense_hp: %d band_size_dense: %d band_size_low_rank_dp:%d NT= %d band_p= %d\n", params->band_size_dist, params->band_size_dense_dp, params->band_size_dense_sp, params->band_size_dense_hp, params->band_size_dense, params->band_size_low_rank_dp, params->NT, params->band_p);
        printf("band_size_auto_tuning_termination= %lf band_size_dense_gpu_memory_max= %d exe_file_path= %s\n", params->band_size_auto_tuning_termination, params->band_size_dense_gpu_memory_max, params->exe_file_path);
        printf("max_rank=%d gen=%d comp=%d\n", params->maxrank, params->genmaxrank, params->compmaxrank);
        printf("auto_band=%d sparse=%d kind_of_cholesky=%d tensor_gemm=%d datatype_convert=%d left_looking=%d\n", params->auto_band, params->sparse, params->kind_of_cholesky, params->tensor_gemm, params->datatype_convert, params->left_looking);
        if( 6 == params->kind_of_problem) printf("mesh_file=%s, rbf_kernel=%d, numobj=%d, order=%d, radius=%f, density=%f\n", params->mesh_file, params->rbf_kernel, params->numobj, params->order, params->radius, params->density);
        printf("time_slots=%d sigma=%lf beta=%lf nu=%lf beta_time=%lf nu_time=%lf nonsep_param=%lf noise= %lf\n", params->time_slots, params->sigma, params->beta, params->nu, params->beta_time, params->nu_time, params->nonsep_param, params->noise);
        printf("numobj/L= %d gpu_rows= %d gpu_cols= %d\n\n", params->numobj, params->gpu_rows, params->gpu_cols);
        fflush(stdout);
    }
}

/**
 * @brief Print final execution results and performance statistics
 * 
 * This function displays comprehensive execution results including performance
 * metrics, memory usage, timing information, and analysis results. The output
 * is formatted for easy parsing and analysis of HICMA execution results.
 * 
 * @param argc Number of command line arguments (unused)
 * @param argv Command line arguments (unused)
 * @param params Pointer to HICMA parameters structure
 * @param analysis Pointer to matrix analysis results structure
 * 
 * @note Only rank 0 prints to avoid duplicate output in parallel execution
 * @note Output format is designed for automated result analysis and benchmarking
 */
void hicma_parsec_params_print_final( int argc, char **argv,
        hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t *analysis )
{
    // Only print on rank 0 to avoid duplicate output in parallel execution
    if( 0 == params->rank ) {
        printf("\nR-LO ");
        printf("%d %d %d %d   ", params->N, params->N, params->NB, params->NB);
        printf("%d %d   ", params->HNB, params->HNB);
        printf("%d %d %d %d %d %d   ", params->nodes, params->P, params->Q, params->cores, params->NT, params->adaptive_decision);
        printf("%d %s %.1e %g %d %g   ", params->kind_of_problem, params->str_problem[params->kind_of_problem], params->fixedacc, params->add_diag, params->fixedrk, params->wave_k);
        printf("%d %d %d %d %d %d  ", params->send_full_tile, params->band_size_dense, params->band_size_dist, params->lookahead, params->reorder_gemm, params->auto_band);
        printf("%d %d %d   ", params->maxrank, params->genmaxrank, params->compmaxrank);
        printf("%lf %lu %lu %llu   ",  params->norm_global, analysis->total_trsm_num, analysis->total_syrk_num, analysis->total_gemm_num);
        printf("%d %lf %lf %lf  ", params->sparse, analysis->initial_density, analysis->density_trsm, analysis->density_gemm);
        printf("%lf %lld   ", (double)analysis->total_memory / 1.0e9, params->total_critical_path_trsm_message);
        printf("%g %d %d   ", params->iavgrk/analysis->initial_density, params->iminrk, params->imaxrk);
        printf("%g %d %d   ", params->iavgrk_auto_band/analysis->initial_density, params->iminrk_auto_band, params->imaxrk_auto_band);
        printf("%g %d %d   ", params->favgrk/analysis->density_trsm, params->fminrk, params->fmaxrk);
        printf("%g %d %d   ", params->iavgrk, params->iminrk, params->imaxrk);
        printf("%g %d %d   ", params->iavgrk_auto_band, params->iminrk_auto_band, params->imaxrk_auto_band);
        printf("%g %d %d   ", params->favgrk, params->fminrk, params->fmaxrk);
        printf("%lf %lf   ", params->memory_per_node, params->memory_per_node_maxrank);
        printf("%lf %lf %lf %lf    ", params->critical_path_time, params->potrf_time, params->trsm_time, params->syrk_time); 
        printf("%lf %lf %lf %lf %lf %lf    ", params->time_starsh, params->time_hicma, params->time_analysis, params->time_opt_band, params->time_regenerate, params->time_reorder);
        printf("%le %le %le %le %le ", (double)params->total_flops, (double)params->total_band, (double)params->total_offband, (double)params->total_path, (double)params->total_offpath);
        printf("%lf %lf %d  ", (double)params->total_band/params->total_offband, (double)params->total_path/params->total_offpath, params->kind_of_cholesky);
        printf("%d %d %d %d %d %d %lf    ", params->gpus, params->band_size_dense_dp, params->band_size_dense_sp, params->band_size_low_rank_dp, params->tensor_gemm, params->datatype_convert, params->band_size_auto_tuning_termination ); 
        printf("%d %d %d    ", params->band_size_dense_gpu_memory_max, params->band_size_dense_gpu_balance_max, params->band_size_dense_gpu_time_max ); 
        printf("%.0lf %.0lf %.0lf %.0lf %.0lf %.0lf   ", params->nb_dense_dp, params->nb_dense_sp, params->nb_dense_hp, params->nb_dense_fp8, params->nb_low_rank_dp, params->nb_low_rank_sp ); 
        printf("%d %d %d %d    ", params->kind_of_cholesky, params->adaptive_decision, params->adaptive_memory, params->adaptive_maxrank ); 
        printf("%d %lf %lf %lf %lf %lf %lf %lf    ", params->time_slots, params->sigma, params->beta, params->nu, params->beta_time, params->nu_time, params->nonsep_param, params->noise);
        printf("%e %d %d %e %e %e %e ", params->result_accuracy, params->left_looking, params->gpu_type, params->norm_global_diff, params->fixedacc * params->norm_global, params->log_det_dp, params->log_det_mp);
        printf("%lf %lf %lf %d  ", params->time_decision_kernel, params->time_decision_sender, params->time_syrk_app, params->numobj);
        printf("%d %d %d %g %g  ", params->order, params->nsnp, params->rbf_kernel, params->radius, params->density);
#ifdef GITHASH
        printf("%s ", xstr(GITHASH));
#else
        printf("GITHASH:N/A    ");
#endif
        printf("\"");
        for(int i = 0; i < argc; i++){
            printf("%s ",  argv[i]);
        }
        printf("\"");
        printf("\n");
        fflush(stdout);
        fflush(stderr);
    }
}


/**
 * @brief Initialize STARSH kernel parameters for matrix generation
 * 
 * This function initializes the STARSH (Structured Tensor Approximation for 
 * Rapid Scientific Computing) kernel parameters based on the HICMA configuration.
 * It sets up the appropriate kernel function and parameters for the specified
 * problem type, which will be used for generating the test matrices.
 * 
 * @param params_kernel Pointer to STARSH parameters structure to initialize
 * @param params Pointer to HICMA parameters structure
 * @return 0 on success, non-zero on error
 * 
 * @note This function configures different kernel types based on kind_of_problem
 * @note Kernel parameters are extracted from HICMA parameters and validated
 */
int hicma_parsec_kernel_init( starsh_params_t *params_kernel, hicma_parsec_params_t *params )
{
    /* Extract kernel parameters from HICMA configuration */
    int time_slots = params->time_slots;
    double sigma = params->sigma; 
    double beta = params->beta;
    double nu = params->nu;
    double beta_time = params->beta_time;
    double nu_time = params->nu_time;
    double nonsep_param = params->nonsep_param;
    double noise = params->noise;
    double aux_param = 0;
    int ndim;

    int N = params->N;
    int NB = params->NB;
    int kind_of_problem = params->kind_of_problem;
    double add_diag = params->add_diag;
    double wave_k = params->wave_k;
    char *mesh_file = params->mesh_file;
    char *pheno_file = params->pheno_file;
    int rbf_kernel = params->rbf_kernel;
    int numobj = params->numobj;
    int nsnp = params->nsnp;
    int order = params->order;
    double radius = params->radius;
    double density = params->density;
    location* locations;
    double *xycoord;
    double *points;
    int j =0, k=0;

    /* Placement template for particles */
    enum STARSH_PARTICLES_PLACEMENT place = STARSH_PARTICLES_UNIFORM;
    

    /* Set seed */
    srand(0);

    /* Allocate memory */ 
    params_kernel->index = malloc( N * sizeof(STARSH_int) );
    
    /* Index */
    for(STARSH_int i = 0; i < N; ++i)
        params_kernel->index[i] = i;
    

     /* Choose the right problem */
    switch( kind_of_problem ) {
        case 0:
            /* Synthetic matrix with equal ranks */
            params_kernel->kernel = starsh_randtlr_block_kernel;
            double decay = 0.5;
            params->info = starsh_randtlr_generate((STARSH_randtlr **)&params_kernel->data, N, NB, decay,
                    add_diag);
            break;
        case 1:
            /* Semi-synthetic matrix with non-equal ranks
             * electrodynamics application from STARS-H
             */
            ndim = 2;
            params_kernel->kernel = starsh_eddata_block_sin_kernel_2d;
            params->info = starsh_eddata_generate((STARSH_eddata **)&params_kernel->data, N, ndim, wave_k,
                    add_diag, place);
            break;

        case 2:
            /* statistics-2d-sqexp */
            ndim = 2;
            params_kernel->kernel = starsh_ssdata_block_sqrexp_kernel_2d;
            params->info = starsh_ssdata_generate((STARSH_ssdata **)&params_kernel->data, N, ndim,
                    beta, nu, noise,
                    place, sigma);
            break;

        case 3:
            /* statistics-3d-sqexp */
            ndim = 3;
            params_kernel->kernel = starsh_ssdata_block_sqrexp_kernel_3d;
            params->info = starsh_ssdata_generate((STARSH_ssdata **)&params_kernel->data, N, ndim,
                    beta, nu, noise,
                    place, sigma);
            break;

        case 4:
            /* statistics-3d-exp */
            ndim = 3;
            params_kernel->kernel = starsh_ssdata_block_exp_kernel_3d;
            params->info = starsh_ssdata_generate((STARSH_ssdata **)&params_kernel->data, N, ndim,
                    beta, nu, noise,
                    place, sigma);
            break;

        case 5:
            /* electrodynamics-3d-sin */
            ndim = 3;
            params_kernel->kernel = starsh_eddata_block_sin_kernel_3d;
            params->info = starsh_eddata_generate((STARSH_eddata **)&params_kernel->data, N, ndim, wave_k,
                    add_diag, place);
            break;

        case 6:
            /* rbf for covid virus */
            ndim = 3;
            params_kernel->kernel = starsh_generate_3d_virus;
            params->info = starsh_generate_3d_rbf_mesh_coordinates_virus((STARSH_mddata **)&params_kernel->data,
                    mesh_file, N, ndim, rbf_kernel, numobj, 1, add_diag,
                    radius, density, order);
            break;

        case 7:
            /* 3D cube */
            ndim = 3;
            params_kernel->kernel = starsh_generate_3d_cube;
            params->info = starsh_generate_3d_rbf_mesh_coordinates_cube((STARSH_mddata **)&params_kernel->data,
                    N, ndim, rbf_kernel, 1, add_diag, radius, order);
            break;

#ifdef GSL
        case 8:
            /* statistics-2d-space-matern */
            place = STARSH_PARTICLES_OBSOLETE1;
            ndim = 2;
            params_kernel->kernel = starsh_ssdata_block_matern_kernel_2d_simd;
            params->info = starsh_ssdata_generate((STARSH_ssdata **)&params_kernel->data, N, ndim,
                    beta, nu, noise,
                    place, sigma);

            /* stop gsl error handler */
            gsl_set_error_handler_off();

            break;

        case 9:
            /* statistics-2d-spacetime-matern */
            place = STARSH_PARTICLES_OBSOLETE5;
            ndim = 2;
            params_kernel->kernel = starsh_ssdata_block_space_time_kernel_2d_simd;
            //Matrix size: N but number of locs is N/time_slots
            params->info = starsh_ssdata_generate_space_time((STARSH_ssdata **)&params_kernel->data, N, ndim, 
                    beta, nu, noise, place, sigma, beta_time, nu_time, nonsep_param, aux_param, time_slots);

            /* stop gsl error handler */
            gsl_set_error_handler_off();

            break;

        case 10:
            // statistics-2d-spacetime-matern-real
            place = STARSH_PARTICLES_OBSOLETE5;
            ndim = 2;
            N = countlines(mesh_file);
            locations = (location *)readLocsFile3d(mesh_file, N);
            //Allocate memory
            xycoord = (double *) malloc( 3 * N * sizeof(double));

            for (int i = 0; i < N; i++){
                xycoord[i] = locations->x[j];
                xycoord[N+i] = locations->y[j];
                xycoord[2*N+i] = locations->z[j];
                k++;
                j++;
            }

            params_kernel->kernel = starsh_ssdata_block_space_time_kernel_2d_simd;
            params->info = starsh_ssdata_generate_space_time_real((STARSH_ssdata **)&params_kernel->data, N, ndim, xycoord,
                    beta, nu, noise, place, sigma, beta_time, nu_time, nonsep_param, aux_param, time_slots);

            // stop gsl error handler
            gsl_set_error_handler_off();

            break;

        case 11:
            // statistics-2d-spacetime-matern-real-exageo
            place = STARSH_PARTICLES_OBSOLETE5;
            points = params->points;
            ndim = 2;

            params_kernel->kernel = starsh_ssdata_block_space_time_kernel_2d_simd;
            params->info = starsh_ssdata_generate_space_time_real_exageo((STARSH_ssdata **)&params_kernel->data, N, ndim, points,
                    beta, nu, noise, place, sigma, beta_time, nu_time, nonsep_param, aux_param, time_slots);

            // stop gsl error handler
            gsl_set_error_handler_off();

            break;

        case 12:
            /* statistics-2d-space-matern-real */
            ndim = 2;
            N = countlines(mesh_file);
            locations = (location *)readLocsFile(mesh_file, N);

            //Allocate memory
            xycoord = (double *) malloc( ndim * N * sizeof(double));

            for (int i = 0; i < N; i++){
                xycoord[i] = locations->x[j];
                xycoord[N+i] = locations->y[j];
                //fprintf(stderr, " %f, %f, %f",xycoord[i] , xycoord[N+i] );
                k++;
                j++;
            }
            params_kernel->kernel = starsh_ssdata_block_matern_kernel_2d_simd;
            params->info = starsh_ssdata_generate_real((STARSH_ssdata **)&params_kernel->data, N, ndim, xycoord,
                    beta, nu, noise,
                    place, sigma);

            /* stop gsl error handler */
            gsl_set_error_handler_off();

            break;

        case 13:
            /* statistics-2d-space-matern-real-exageo */
            ndim = 2;
            points = params->points;

            params_kernel->kernel = starsh_ssdata_block_matern_kernel_2d_simd;
            params->info = starsh_ssdata_generate_real_exageo((STARSH_ssdata **)&params_kernel->data, N, ndim, points,
                    beta, nu, noise,
                    place, sigma);

            /* stop gsl error handler */
            gsl_set_error_handler_off();

            break;
#endif

        case 14:
            /* gene random */
            //ndim = nsnp;
            //params_kernel->kernel = starsh_generate_gene_matrix;
            //params->info = starsh_generate_random_krr_gene((STARSH_genedata **)&params_kernel->data,
            //       N, nsnp, rbf_kernel, 1, add_diag, radius, order);
            //printf("N:%d, nsnap:%d, rbf_kernel:%d, add_diag:%f, radius:%f", N, nsnp, rbf_kernel, add_diag, radius);
            if(params->rank == 0) printf("\nGenerate random Genotype data in inside parsec!!\n");
            break;

        case 15:
            /* gene file */
            //ndim = nsnp;
            //params_kernel->kernel = starsh_generate_gene_matrix;
            //params->info = starsh_generate_krr_gene((STARSH_genedata **)&params_kernel->data, mesh_file,
            //        N, nsnp, rbf_kernel, 1, add_diag, radius, order);
            if(params->rank == 0) printf("\nRead Genotype data in inside parsec using method in the paper!!\n");
            break;

        default:
            fprintf(stderr, "Wrong value of \"kind_of_problem\" parameter\n");
            return -1;
    }

    if( params->info != 0 )
    {
        fprintf(stderr, "Problem was NOT generated (wrong parameters)\n");
        exit(1);
    }

    return 0;
}


/**
 * @brief Validate and adjust HICMA parameters based on configuration
 * 
 * This function performs comprehensive parameter validation and adjustment
 * based on the selected computation mode, hardware capabilities, and
 * problem characteristics. It ensures parameter consistency and sets
 * appropriate defaults for different computation scenarios.
 * 
 * @param params Pointer to HICMA parameters structure to validate
 * @return 0 on success, non-zero on error
 */
int hicma_parsec_params_check( hicma_parsec_params_t *params )
{
    /* ===========================================
     * Basic parameter validation
     * =========================================== */

    if(GENOMICS || DEBUG_INFO || params->check) {
        params->adaptive_memory = 0;
    }

    // Get the right kind_of_cholesky
    if(params->auto_band) {
        params->kind_of_cholesky = DENSE_TLR_MP;
        if( params->rank == 0 ) {
            fprintf(stderr, RED "Auto_band enabled, so kind_of_cholesky = DENSE_TLR_MP\n" RESET);
        }
    }

    // Validate lookahead parameter (must be >= -1, where -1 means auto-tune)
    assert(params->lookahead >= -1);

    /* ===========================================
     * Compile-time configuration adjustments
     * =========================================== */

#if GENOMICS
    // Gene-specific configuration: disable double precision dense computation
    params->band_size_dense_dp = 0;
#endif

#if defined(ENABLE_SANITIZER)
    // Sanitizer mode: notify user about debugging mode
    if( params->rank == 0 ) {
        fprintf(stderr, RED "ENABLE_SANITIZER enabled\n" RESET);
    }
#endif

#if GENERATE_RANDOM_DATA
    // Random data generation: disable correctness checking
    if( params->rank == 0 && params->check ) {
        fprintf(stderr, RED "Disable check if GENERATE_RANDOM_DATA\n" RESET);
    }
    params->check = 0;
#endif

    /* ===========================================
     * Performance warnings and recommendations
     * =========================================== */
    
    // Warn about tile size importance for performance
    if( params->rank == 0 ) {
        fprintf(stderr, RED "Tile size is very important to performance, especially for Low-rank. Make sure to tune the tile size for performance!!!\n" RESET);
    }

    /* ===========================================
     * Sparse computation configuration
     * =========================================== */
    
    // Configure Cholesky type based on sparsity level
    if( 1 == params->sparse ) {
        // General sparse TLR double precision
        params->kind_of_cholesky = SPARSE_TLR_DP_GENERAL;
        if( params->rank == 0 ) {
            fprintf(stderr, RED "params->kind_of_cholesky is set to SPARSE_TLR_DP_GENERAL\n" RESET);
        }
    } else if( params->sparse > 1 ) {
        // Balanced sparse TLR double precision
        params->kind_of_cholesky = SPARSE_TLR_DP_BALANCE;
        if( params->rank == 0 ) {
            fprintf(stderr, RED "params->kind_of_cholesky is set to SPARSE_TLR_DP_BALANCE\n" RESET);
        }
    } 
   
    /* ===========================================
     * Hardware capability validation
     * =========================================== */
    
    // Check half-precision support on CPU
    if( HAVE_HP && !HAVE_HP_CPU && params->gpus < 1 ) {
        if( params->rank == 0 ) {
            fprintf(stderr, RED "Half-precision only for accuracy purpose! And it's could be disable by setting HAVE_HP to 0\n" RESET);
        }
    }

    /* If ADAPTIVE_CONVERT */
    if( ADAPTIVE_CONVERT == params->datatype_convert ) {
        //params->kind_of_cholesky = DENSE_MP_GPU;
        params->band_size_dense = params->NT;
        if( params->rank == 0 ) {
            fprintf(stderr, RED "ADAPTIVE_CONVERT is selected, so the right one (-w 5 / DENSE_MP_GPU) needs to choose !\n" RESET);
        }
    }

    /* Disable datatype_convert */
    if( params->kind_of_cholesky != DENSE_MP_BAND && params->datatype_convert == SENDER_CONVERT ) { 
        if( params->rank == 0 ) {
            fprintf(stderr, YEL "Warning: datatype_convert is only supported in DENSE_MP_BAND\n" RESET);
        }
        exit(1);
    }

    /* If maxrank > NB / 2, it's better to enable band_size_dense auto tuning */ 
    if( params->maxrank > params->NB / 2 ) {
        if( params->rank == 0 )
            fprintf(stderr, RED "maxrank= %d is bigger than NB/2, it's better to enable band_size auto tuning by -E 1. Also, maxrank could be reset after band_size_dense auto-tuning by -A 1\n" RESET, params->maxrank);
    }

    /* If sparse, disabled auto_band tuning and set band_size_dense to 1 */ 
    if( params->sparse ) {
        params->auto_band = 0;
        params->band_size_dense_dp = 1;
        params->band_size_dense_sp = 1;
        params->band_size_dense = 1;
        params->band_size_low_rank_dp = params->NT;
        if( params->rank == 0 && params->verbose) {
            printf(YEL "%d: Disable auto-tuning band_size_dense and set band_size_dense = 1 at the beginning\n" RESET, __LINE__);
            fflush(stdout);
        }
    }

    /* If mixed-precision of DP/SP/HP */
    if( DENSE_MP_BAND == params->kind_of_cholesky ) {
        params->auto_band = 0;
        params->band_size_dense = params->NT;
        params->band_size_low_rank_dp = params->NT;
        if( params->rank == 0 && params->verbose) {
            printf(YEL "%d: mixed DP/SP/HP: disable auto-tuning band_size_dense and set band_size_dense = NT at the beginning\n" RESET, __LINE__);
            fflush(stdout);
        }
    }

    /* If mixed-precision of SP/HP */ 
    if( DENSE_SP_HP_BAND == params->kind_of_cholesky || DENSE_MP_GPU_FP8_SP == params->kind_of_cholesky ) {
        params->auto_band = 0;
        params->band_size_dense_dp = 0;
        params->band_size_dense = params->NT;
        params->band_size_low_rank_dp = params->NT;
        if( params->rank == 0 && params->verbose) {
            printf(YEL "%d: mixed SP/HP: disable auto-tuning band_size_dense and set band_size_dense_dp = 0, band_size_dense = params->band_size_low_rank_dp = NT at the beginning\n" RESET, __LINE__);
            fflush(stdout);
        }
        if( params->band_size_dense_sp > params->band_size_dense )
            params->band_size_dense_sp = params->band_size_dense;
    }

    /* If auto select band_size_dense, band_size_dense set to 1 at the beginning */
    if( params->auto_band ) {
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        if( params->gpus > 0 ) {
            if( params->rank == 0 && params->verbose ) {
                printf(YEL "%d: on GPU auto_band set to 1\n" RESET, __LINE__);
                fflush(stdout);
            }

            params->auto_band = 1;

            /* Update the termination factor if on GPU */
            hicma_parsec_performance_testing_cpu_gpu( params );
        }
#endif

        if( params->rank == 0 && params->verbose) {
            printf(YEL "%d: Auto-tuning band_size_dense, set band_size_dense_dp = band_size_dense_sp = band_size_dense = %d and band_size_low_rank_dp = NT at the beginning\n" RESET, __LINE__, params->auto_band);
            fflush(stdout);
        }

        params->band_size_dense_dp = params->auto_band;
        params->band_size_dense_sp = params->auto_band;
        params->band_size_dense = params->auto_band;
        params->band_size_low_rank_dp = params->NT;
    }

    /* If allocate memory for band continually, set band_size_dist = band_size_dense */
#if BAND_MEMORY_CONTIGUOUS
    if( params->rank == 0) {
        printf(YEL "%d: Allocate memory on band contiguous on each process, and set band_size_dist = band_size_dense\n" RESET, __LINE__);
        fflush(stdout);
    }
    params->band_size_dist_provided = 0;
    params->band_size_dist = params->band_size_dense;
#endif

    if( 0 == params->rank && params->band_size_dense != params->band_size_dist )
        printf(YEL "%d: Band distribution is different from band dense\n" RESET, __LINE__);

    /* Re-order GEMM is not support now */
    assert( 0 == params->reorder_gemm );
    if( params->reorder_gemm ) {
        if( params->rank == 0 && params->verbose ) {
            printf(YEL "%d: Re-order gemm, set band_size_dense = 1 at the beginning\n" RESET, __LINE__);
            fflush(stdout);
        }
        params->band_size_dense = 1;
    }

    /* reset params of band_size */
    if( params->band_size_dense_dp >= params->NT ) {
        if( params->rank == 0 && params->verbose ) {
            printf(YEL "%d: Set band_size_dense_dp, band_size_dense_sp, band_size_dense, and band_size_low_rank_dp to NT\n" RESET, __LINE__);
            fflush(stdout);
        }
        params->band_size_dense_dp = params->NT;
        params->band_size_dense_sp = params->NT;
        params->band_size_dense = params->NT;
        params->band_size_low_rank_dp = params->NT;
    } else if( params->band_size_dense_sp >= params->NT ) {
        if( params->rank == 0 && params->verbose ) {
            printf(YEL "%d: Set band_size_dense_sp, band_size_dense, and band_size_low_rank_dp to NT\n" RESET, __LINE__);
            fflush(stdout);
        }
        params->band_size_dense_sp = params->NT;
        params->band_size_dense = params->NT;
        params->band_size_low_rank_dp = params->NT;
    } else if( params->band_size_dense >= params->NT ) {
        if( params->rank == 0 && params->verbose ) {
            printf(YEL "%d: Set band_size_dense, and band_size_low_rank_dp to NT\n" RESET, __LINE__);
            fflush(stdout);
        }
        params->band_size_dense = params->NT;
        params->band_size_low_rank_dp = params->NT;
    } else if( params->band_size_low_rank_dp >= params->NT ) {
        if( params->rank == 0 && params->verbose ) {
            printf(YEL "%d: Set band_size_low_rank_dp to NT\n" RESET, __LINE__);
            fflush(stdout);
        }
        params->band_size_low_rank_dp = params->NT;
    }

    /* Check band_size_* */
    assert( params->band_size_dense >= 1 );
    assert( params->band_size_dense_dp <= params->band_size_dense_sp );
    assert( params->band_size_dense_sp <= params->band_size_dense );
    assert( params->band_size_dense <= params->band_size_low_rank_dp ); 

    /* Warning of gather flops */
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    if( params->rank == 0 && params->verbose && params->gpus > 0 ) {
        printf(YEL "%d: flops in not gathered on GPU\n" RESET, __LINE__);
        fflush(stdout);
    }
#endif

#if !BAND_MEMORY_CONTIGUOUS
    /* Make sure if band_size_dense is bigger enough, set 2DBCDD instead of band distribution */
    if( (params->band_size_dense > params->P || params->band_size_dist > params->P) && params->band_size_dist < 1 ) {
        int tmp_band_size_dist = params->band_size_dist;
        if( params->sparse ) {
            params->band_size_dist = 1;
        } else {
            params->band_size_dist = 0;
        }
        VERBOSE_PRINT(params->rank, params->verbose,
                (RED "band_size_dense= %d or band_size_dist= %d is bigger than P= %d so set distribution to 2DBCDD instead of band distribution by -d %d\n" RESET, 
                 params->band_size_dense, tmp_band_size_dist, params->P, params->band_size_dist));
    }
#endif

    /* Check if set the distribution seperately
     * if auto band tuning, then set band_size_dist = band_size_dense,
     * because of process of regenerating matrix */
    if( -1 == params->band_size_dist ) {
        params->band_size_dist_provided = 0;
        params->band_size_dist = params->band_size_dense;
    }

    /* ed-3d-sin and md-3d-cube are not tested so disable them */
    /*if( 5 == params->kind_of_problem || 7 == params->kind_of_problem ) {
        if( 0 == params->rank ) {
            fprintf(stderr, RED "-D 5 (ed-3d-sin) and -D 7 (md-3d-cube) are not tested so they are disabled\n" RESET);
        }
        exit(0);
    }*/

    /* left_looking */
    if( params->left_looking < 0 ) {
        if( 0 == params->rank ) {
            fprintf(stderr, RED "left_looking should be non-negative\n" RESET);
        }
        exit(0);
    }

    /* Make sure tensor_gemm is correct */
    // TODO Add BF16 support is needed
    if( (params->tensor_gemm & MASK_BF16_A16_B16_C32_OP32) && (params->auto_band || params->band_size_dense < params->NT) ) {
        if( params->rank == 0 ) {
            fprintf(stderr, RED "tensor_gemm BF16 only supports MP+dense (no TLR)\n" RESET);
        }
        exit(0);
    }

#if defined(PARSEC_HAVE_DEV_RECURSIVE_SUPPORT)
    // TODO https://github.com/ICLDisco/parsec/issues/541
    params->HNB = params->NB;
    if( params->rank == 0 ) {
        fprintf(stderr, RED "Disable recursive now !!!\n" RESET);
    }
#endif

#if !HAVE_FP8
    params->band_size_dense_hp = params->band_size_dense; 
#endif

    return 0; 
}


/**
 * @brief Initialize data descriptors and GPU workspace
 * 
 * This function initializes all data descriptors required for HICMA computation,
 * including matrix descriptors for different precision types, band structures,
 * and GPU workspace allocation. It sets up the distributed data layout
 * and allocates memory based on the computation mode.
 * 
 * @param data Pointer to HICMA data structure to initialize
 * @param params Pointer to HICMA parameters structure
 * @return 0 on success, non-zero on error
 */
int hicma_parsec_data_init( hicma_parsec_data_t *data, hicma_parsec_params_t *params )
{
    /* ===========================================
     * Extract commonly used parameters
     * =========================================== */
    
    int rank = params->rank; 
    int N = params->N;
    int NB = params->NB;
    int NT = params->NT;
    int RHS = params->RHS;
    int SNP = params->nsnp;
    int P = params->P;
    int nodes = params->nodes;
    int uplo = params->uplo;
    int band_size_dist = params->band_size_dist;
    int band_p = params->band_p;
    int auto_band = params->auto_band;
    int band_size_dense = params->band_size_dense;
    int sparse = params->sparse;
    int verbose = params->verbose;

    /* ===========================================
     * Initialize dense matrix descriptor (dcAd)
     * =========================================== */
    
    // Initialize symmetric block-cyclic matrix descriptor for dense computation
#if GENOMICS
    // Gene computation: use single precision
    parsec_matrix_sym_block_cyclic_init(&data->dcAd, PARSEC_MATRIX_FLOAT,
            rank, NB, NB, N, N, 0, 0,
            N, N, P, nodes/P, uplo);
#else
    // General computation: use double precision
    parsec_matrix_sym_block_cyclic_init(&data->dcAd, PARSEC_MATRIX_DOUBLE,
            rank, NB, NB, N, N, 0, 0,
            N, N, P, nodes/P, uplo);
#endif

    /* ===========================================
     * Initialize matrix copy for prediction/checking
     * =========================================== */
    
#if PREDICTION || CHECKSOLVE && GENOMICS
    // Initialize block-cyclic matrix copy for gene prediction/checking
    parsec_matrix_block_cyclic_init(&data->dcAcpy, PARSEC_MATRIX_FLOAT, PARSEC_MATRIX_TILE,
            rank, NB, NB, N, N, 0, 0,
            N, N, P, nodes/P, 1, 1, 0, 0);
#endif

    /* ===========================================
     * Conditional memory allocation for dense matrices
     * =========================================== */
    
    // Allocate memory for dense computation when band size covers entire matrix
    if( band_size_dense >= NT && 0 == auto_band && !params->adaptive_memory ) {
        
        // Allocate memory for dense matrix descriptor
#if !GENOMICS
        size_t allocate_size = (size_t)data->dcAd.super.nb_local_tiles *
                (size_t)data->dcAd.super.bsiz *
                (size_t)parsec_datadist_getsizeoftype(data->dcAd.super.mtype);
        data->dcAd.mat = parsec_data_allocate(allocate_size);
        // Register memory in advance
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        data->dcAd.super.super.register_memory = NULL;
        data->dcAd.super.super.unregister_memory = NULL;
        cudaHostRegister(data->dcAd.mat, allocate_size, cudaHostRegisterDefault); 
#endif
        
#endif

        // Allocate memory for matrix copy if needed for prediction/checking
#if PREDICTION || CHECKSOLVE
        data->dcAcpy.mat = parsec_data_allocate((size_t)data->dcAcpy.super.nb_local_tiles *
                (size_t)data->dcAcpy.super.bsiz *
                (size_t)parsec_datadist_getsizeoftype(data->dcAcpy.super.mtype));
        parsec_data_collection_set_key(&data->dcAcpy.super.super, "dcAcpy");
#endif

        // Notify user about dense memory allocation strategy
        if( params->rank == 0 ) {
            printf(YEL "Matrix is dense (band_size_dense >= NT) and auto_band is disabled, so memory is allocated continuously using dcAd instead\n" RESET);
            fflush(stdout);
        }
    }

    parsec_data_collection_set_key(&data->dcAd.super.super, "dcAd");

    /* dcA data descriptor */
    parsec_matrix_sym_block_cyclic_init(&data->dcA.off_band, PARSEC_MATRIX_DOUBLE,
            rank, NB, NB, N, N, 0, 0,
            N, N, P, nodes/P, uplo);

    /* Init band */
    parsec_matrix_block_cyclic_init(&data->dcA.band, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
            rank, NB, NB, NB*band_size_dist, N, 0, 0,
            NB*band_size_dist, N, band_p, nodes/band_p,
            1, 1, 0, 0);

#if BAND_MEMORY_CONTIGUOUS
    /* Allocate memory on band */
    if( band_size_dense < NT || auto_band != 0 || params->adaptive_memory ) {
        data->dcA.band.mat = parsec_data_allocate((size_t)data->dcA.band.super.nb_local_tiles *
                (size_t)data->dcA.band.super.bsiz *
                (size_t)parsec_datadist_getsizeoftype(data->dcA.band.super.mtype));
    }
#endif

    /* Init two_dim_block_cyclic_band_t structure */
    if( (auto_band || band_size_dense > 1) && 0 == sparse && band_size_dist )
        hicma_parsec_parsec_matrix_sym_block_cyclic_band_init( &data->dcA, nodes, rank, band_size_dist );
    else
        parsec_matrix_sym_block_cyclic_band_init( &data->dcA, nodes, rank, band_size_dist );
    parsec_data_collection_set_key((parsec_data_collection_t*)&data->dcA, "dcA_off_band");
    parsec_data_collection_set_key(&data->dcA.band.super.super, "dcA_band");

    /* dcAr data descriptor */
    parsec_matrix_sym_block_cyclic_init(&data->dcAr.off_band, PARSEC_MATRIX_INTEGER,
            rank, 1, 1, NT, NT, 0, 0,
            NT, NT, P, nodes/P, uplo);

    /* Init band */
    parsec_matrix_block_cyclic_init(&data->dcAr.band, PARSEC_MATRIX_INTEGER, PARSEC_MATRIX_TILE,
            rank, 1, 1, band_size_dist, NT, 0, 0,
            band_size_dist, NT, band_p, nodes/band_p,
            1, 1, 0, 0);

    /* Allocate memory on band */
    data->dcAr.band.mat = parsec_data_allocate((size_t)data->dcAr.band.super.nb_local_tiles *
            (size_t)data->dcAr.band.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(data->dcAr.band.super.mtype));

    /* Allocate memory off band */
    data->dcAr.off_band.mat = parsec_data_allocate((size_t)data->dcAr.off_band.super.nb_local_tiles *
            (size_t)data->dcAr.off_band.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(data->dcAr.off_band.super.mtype));

    /* Init two_dim_block_cyclic_band_t structure */
    if( (auto_band || band_size_dense > 1) && 0 == sparse && band_size_dist )
        hicma_parsec_parsec_matrix_sym_block_cyclic_band_init( &data->dcAr, nodes, rank, band_size_dist );
    else
        parsec_matrix_sym_block_cyclic_band_init( &data->dcAr, nodes, rank, band_size_dist );
    parsec_data_collection_set_key((parsec_data_collection_t*)&data->dcAr, "dcAr_off_band");
    parsec_data_collection_set_key(&data->dcAr.band.super.super, "dcAr_band");

    /* Initialize dcAr to negtive, used to gather rank using MPI_Gather / MPI_Gatherv */
    for(int i = 0; i < data->dcAr.band.super.nb_local_tiles; i++)
        ((int *)data->dcAr.band.mat)[i] = -1;
    for(int i = 0; i < data->dcAr.off_band.super.nb_local_tiles; i++)
        ((int *)data->dcAr.off_band.mat)[i] = -1;

    /* Gather rank info during Cholesky if PRINT_RANK*/
    /* dcRank data descriptor : init_rank, min_rank, max_rank, final_rank */
    parsec_matrix_sym_block_cyclic_init(&data->dcRank.off_band, PARSEC_MATRIX_INTEGER,
                                  rank, 1, RANK_MAP_BUFF, NT, RANK_MAP_BUFF*NT, 0, 0,
                                  NT, RANK_MAP_BUFF*NT, P, nodes/P, uplo);

    /* Init band */
    parsec_matrix_block_cyclic_init(&data->dcRank.band, PARSEC_MATRIX_INTEGER, PARSEC_MATRIX_TILE,
                              rank, 1, RANK_MAP_BUFF, band_size_dist, RANK_MAP_BUFF*NT, 0, 0,
                              band_size_dist, RANK_MAP_BUFF*NT, band_p, nodes/band_p,
                              1, 1, 0, 0);

    /* Init two_dim_block_cyclic_band_t structure */
    if( (auto_band || band_size_dense > 1) && 0 == sparse && band_size_dist )
        hicma_parsec_parsec_matrix_sym_block_cyclic_band_init( &data->dcRank, nodes, rank, band_size_dist );
    else
        parsec_matrix_sym_block_cyclic_band_init( &data->dcRank, nodes, rank, band_size_dist );
    parsec_data_collection_set_key((parsec_data_collection_t*)&data->dcRank, "dcRank_super");
    parsec_data_collection_set_key(&data->dcRank.band.super.super, "dcRank_band");

    if( 0 == rank && verbose ) { printf("%d: Dense diagonal, U, V, rank Matrices are allocated\n", __LINE__); fflush(stdout); }

    /* Allocate memory for GPU workspace */
#if (defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)) && GPU_BUFFER_ONCE 
    gpu_temporay_buffer_init( data, NB, NB, params->compmaxrank, params->kind_of_cholesky );
#endif

    /* dcFake to control process rank working on*/
    parsec_matrix_block_cyclic_init(&data->dcFake, PARSEC_MATRIX_INTEGER, PARSEC_MATRIX_TILE,
            rank, 1, 1, 1, nodes, 0, 0,
            1, nodes, 1, nodes, 1, 1, 0, 0);
    data->dcFake.mat = parsec_data_allocate((size_t)data->dcFake.super.nb_local_tiles *
            (size_t)data->dcFake.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(data->dcFake.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&data->dcFake, "dcFake");

#if GENOMICS
    /* desc for pheno, X, and prediction*/

    parsec_matrix_block_cyclic_init(&data->dcB, PARSEC_MATRIX_FLOAT, PARSEC_MATRIX_TILE,
            rank, NB, RHS, N, RHS, 0, 0,
            N, RHS, P, nodes/P, 1, 1, 0, 0);
    data->dcB.mat = parsec_data_allocate((size_t)data->dcB.super.nb_local_tiles *
            (size_t)data->dcB.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(data->dcB.super.mtype));   
    parsec_data_collection_set_key((parsec_data_collection_t*)&data->dcB, "dcB");

    parsec_matrix_block_cyclic_init(&data->dcX, PARSEC_MATRIX_FLOAT, PARSEC_MATRIX_TILE,
            rank, NB, RHS, N, RHS, 0, 0,
            N, RHS, P, nodes/P, 1, 1, 0, 0);
    data->dcX.mat = parsec_data_allocate((size_t)data->dcX.super.nb_local_tiles *
            (size_t)data->dcX.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(data->dcX.super.mtype));   
    parsec_data_collection_set_key((parsec_data_collection_t*)&data->dcX, "dcX");

    parsec_matrix_block_cyclic_init(&data->dcP, PARSEC_MATRIX_FLOAT, PARSEC_MATRIX_TILE,
            rank, NB, RHS, N, RHS, 0, 0,
            N, RHS, P, nodes/P, 1, 1, 0, 0);
    data->dcP.mat = parsec_data_allocate((size_t)data->dcP.super.nb_local_tiles *
            (size_t)data->dcP.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(data->dcP.super.mtype));   
    parsec_data_collection_set_key((parsec_data_collection_t*)&data->dcP, "dcP");

#endif

    return 0;
}

/**
 * @brief Automatic tuning of dense band size for optimal performance
 * 
 * This function automatically determines the optimal dense band size for the
 * Cholesky factorization algorithm. It performs a search over different band
 * sizes to find the configuration that provides the best performance for the
 * given problem size and system configuration.
 * 
 * The auto-tuning process:
 * - Temporarily disables GPU acceleration for accurate CPU-only benchmarking
 * - Tests different band sizes using binary search or linear search
 * - Selects the band size that minimizes execution time
 * - Re-enables GPU acceleration after tuning is complete
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param data Pointer to HICMA data structures containing matrix descriptors
 * @param params Pointer to HICMA parameters (modified during tuning)
 * @param params_kernel Pointer to STARSH kernel parameters
 * @return 0 on success, non-zero on error
 * 
 * @note This function modifies params->band_size_dense if a better size is found
 * @note GPU acceleration is temporarily disabled during the tuning process
 * @note Memory allocation may be adjusted based on the new band size
 */
int hicma_parsec_band_size_dense_auto_tuning( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel )
{
    /* If disabled, just return */
    if( 0 == params->auto_band ) return 0;

    int rank = params->rank;
    int nodes = params->nodes;
    int N = params->N;
    int NT = params->NT;
    int NB = params->NB;
    int *rank_array = params->rank_array;
    int band_p = params->band_p;
    int band_size_dense = params->band_size_dense;
    int uplo = params->uplo;
    int nb_gpus = params->gpus;

    /* Set number of GPUs to 0 to disable GPU in band_size_auto_tuning */
    // TODO
    params->gpus = 0;

    /* Find the best band_size_dense */
    int band_size_dense_opt;
    if( params->gpus > 0 ) { 
        band_size_dense_opt = parsec_band_size_dense_auto_tuning_binary_search(parsec,
                (parsec_tiled_matrix_t*)&data->dcAr,
                (parsec_tiled_matrix_t*)&data->dcFake,
                rank_array, params );
    } else {
        band_size_dense_opt = parsec_band_size_dense_auto_tuning(parsec,
                (parsec_tiled_matrix_t*)&data->dcAr,
                (parsec_tiled_matrix_t*)&data->dcFake,
                rank_array, params );
    }

    /* Set back */
    params->gpus = nb_gpus;

    /* If band_size_dense changed */
    if( band_size_dense_opt > params->band_size_dense ) {
        /* Free band memory for A */
#if BAND_MEMORY_CONTIGUOUS
        parsec_data_free(data->dcA.band.mat);
#else
        parsec_band_free_memory(parsec, (parsec_tiled_matrix_t *)&data->dcA, params, FREE_BAND_MEMORY);
#endif

        /* Set band_size_dense */
        params->band_size_dense = band_size_dense_opt;

#if BAND_MEMORY_CONTIGUOUS
        params->band_size_dist = params->band_size_dense; 
#else

        /* Make sure if band_size_dense is bigger enough, set 2DBCDD instead of band distribution */
        if( params->band_size_dense > params->P || params->band_size_dist > params->P ) {
            VERBOSE_PRINT(params->rank, params->verbose,
                    ("band_size_dense= %d is bigger than P= %d so that it's better to set distribution to 2DBCDD instead of band distribution by -d 0\n" RESET,
                     params->band_size_dense, params->P));
            params->band_size_dist = 0;
        } else if ( 0 == params->band_size_dist_provided ) {
            /* If not set distribution, re-init to the same as band_size_dense */
            params->band_size_dist = params->band_size_dense;
        }
#endif

        parsec_tiled_matrix_destroy( &data->dcA.band.super );

        /* Re-init band */
        parsec_matrix_block_cyclic_init(&data->dcA.band, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                rank, NB, NB, NB*params->band_size_dist, N, 0, 0,
                NB*params->band_size_dist, N, band_p, nodes/band_p,
                1, 1, 0, 0);

        /* Set band size */
        data->dcA.band_size = (unsigned int)params->band_size_dist;
        data->dcAr.band_size = (unsigned int)params->band_size_dist;

#if BAND_MEMORY_CONTIGUOUS
        assert( params->band_size_dense == params->band_size_dist );

        /* Re-allocate memory on band */
        data->dcA.band.mat = parsec_data_allocate((size_t)data->dcA.band.super.nb_local_tiles *
                (size_t)data->dcA.band.super.bsiz *
                (size_t)parsec_datadist_getsizeoftype(data->dcA.band.super.mtype));
#endif

        /* band generation */
        VERBOSE_PRINT(params->rank, params->verbose, ("Re-generating matrix of dense tiles:\n"));
        parsec_band_regenerate(parsec, uplo, (parsec_tiled_matrix_t *)&data->dcA,
                params, params_kernel, params->band_size_dense);

        /* Update decision if TLR + mixed-precision */
        /* Update decsions */
        VERBOSE_PRINT(params->rank, params->verbose, ("Update decisions after auto-tuning band_size_dense\n"));
        for(int i = 0; i < NT; i++) {
            for(int j = 0; j <= i; j++) {
                if( i-j < params->band_size_dense )
                    params->decisions[j*NT+i] = DENSE_DP;
                else
                    params->decisions[j*NT+i] = LOW_RANK_DP;
            }
        }

        if( params->verbose > 9 ) {
            print_decisions( params );
        }
    }

    return 0;
}


/**
 * @brief Reorder GEMM operations for improved cache locality
 * 
 * This function implements matrix reordering to optimize the sequence of
 * General Matrix Multiply (GEMM) operations during Cholesky factorization.
 * The reordering aims to improve cache locality and reduce memory access
 * overhead by reorganizing the computation order.
 * 
 * The function initializes data structures for the reordered computation:
 * - Creates a reorder descriptor for tracking the new computation order
 * - Sets up band-specific data structures for dense and distributed regions
 * - Configures the matrix layout for optimal memory access patterns
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param data Pointer to HICMA data structures (modified with reorder info)
 * @param params Pointer to HICMA parameters containing reorder settings
 * @return 0 on success, non-zero on error
 * 
 * @note This feature is disabled by default and controlled by params->reorder_gemm
 * @note The reordering affects both dense and sparse computation regions
 * @note Memory layout is optimized for the specific problem size and system
 */
int hicma_parsec_reorder_gemm( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    int rank = params->rank;
    int nodes = params->nodes;
    int NT = params->NT;
    int band_p = params->band_p;
    int band_size_dense = params->band_size_dense;
    int band_size_dist = params->band_size_dist;
    int uplo = params->uplo;
    int P = params->P;
    int auto_band = params->auto_band;
    int sparse = params->sparse;
    int reorder_gemm = params->reorder_gemm;

    /* Just return */
    if( 0 == reorder_gemm ) return 0;

    /* Re-order GEMM */
    /* dcReorder data descriptor */
    parsec_matrix_sym_block_cyclic_init(&data->dcReorder.off_band, PARSEC_MATRIX_INTEGER,
            rank, 1, NT, NT, NT*NT, 0, 0,
            NT, NT*NT, P, nodes/P, uplo);

    /* Init band */
    parsec_matrix_block_cyclic_init(&data->dcReorder.band, PARSEC_MATRIX_INTEGER, PARSEC_MATRIX_TILE,
                              rank, 1, NT, band_size_dense, NT*NT, 0, 0,
                              band_size_dist, NT*NT, band_p, nodes/band_p,
                              1, 1, 0, 0);

    /* Init two_dim_block_cyclic_band_t structure */
    if( (auto_band || band_size_dense > 1) && 0 == sparse ) 
        hicma_parsec_parsec_matrix_sym_block_cyclic_band_init( &data->dcReorder, nodes, rank, band_size_dense );
    else
        parsec_matrix_sym_block_cyclic_band_init( &data->dcReorder, nodes, rank, band_size_dense );
    parsec_data_collection_set_key((parsec_data_collection_t*)&data->dcReorder, "dcReorder_super");
    parsec_data_collection_set_key(&data->dcReorder.band.super.super, "dcReorder_band");

/* Disabled */
#if 0
    parsec_reorder_gemm(parsec, (parsec_tiled_matrix_t*)&data->.dcAr,
                                (parsec_tiled_matrix_t*)&data->dcReorder,
                                Ar_copy, disp, nb_elem_r, band_size_dense,
                                reorder_gemm);
#endif

    return 0;
}

/**
 * @brief Calculate memory requirements for Cholesky factorization
 * 
 * This function computes the memory requirements for the Cholesky factorization
 * algorithm based on the matrix dimensions, rank parameters, and system
 * configuration. It calculates both average and maximum memory usage scenarios.
 * 
 * Memory calculation considers:
 * - Dense band region: band_size_dense * NT * NB * NB elements
 * - Low-rank region: (NT - band_size_dense) * (NT - band_size_dense + 1) * NB * rank
 * - Average rank scenario: uses iavgrk for typical memory usage
 * - Maximum rank scenario: uses maxrank for worst-case memory usage
 * 
 * The function validates that memory requirements do not exceed system limits
 * and provides early warning if memory usage approaches the threshold.
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param params Pointer to HICMA parameters (memory fields are updated)
 * @return 0 on success, non-zero on error (exits if memory limit exceeded)
 * 
 * @note Memory is calculated per node for distributed systems
 * @note Results are stored in params->memory_per_node and params->memory_per_node_maxrank
 * @note Function exits with error if memory usage exceeds 70% of THRESHOLD_MEMORY_PER_NODE
 */
int hicma_parsec_memory_calculation( parsec_context_t *parsec,
        hicma_parsec_params_t *params ) 
{
    int rank = params->rank;
    int nodes = params->nodes;
    int NT = params->NT;
    int NB = params->NB;
    int band_size_dense = params->band_size_dense;
    int maxrank =  params->maxrank;
    int iavgrk = params->iavgrk;

    long long int size_allocate, size_allocate_maxrank;

    /* Calculate memory needed before factorization
     * memory_per_node: based on actual rank 
     * memory_per_node_max: based on maxrank
     */
    size_allocate = (long long int)(NT - band_size_dense) * (NT - band_size_dense + 1) * NB * iavgrk + (long long int)band_size_dense * NT * NB * NB;
    size_allocate_maxrank = (long long int)(NT - band_size_dense) * (NT - band_size_dense + 1) * NB * maxrank + (long long int)band_size_dense * NT * NB * NB;

    params->memory_per_node = size_allocate / (double)1024 / 1024 / 1024 * 8 / nodes; 
    params->memory_per_node_maxrank = size_allocate_maxrank / (double)1024 / 1024 / 1024 * 8 / nodes; 

    if( params->memory_per_node > THRESHOLD_MEMORY_PER_NODE * 0.7 ) {
        if( 0 == rank )
            fprintf(stderr, "memory_for_matrix_allocation_per_node : 0.7 * %d < %lf Gbytes\n", THRESHOLD_MEMORY_PER_NODE, params->memory_per_node);
        exit(1);
    }

    return 0;
}



/**
 * @brief Calculate computational statistics for Cholesky factorization
 * 
 * This function calculates the number of floating-point operations (FLOPs)
 * and critical path timing for the Cholesky factorization algorithm.
 * It analyzes the computational complexity and performance characteristics
 * of the algorithm based on the matrix dimensions and algorithm parameters.
 * 
 * @param params Pointer to HICMA parameters structure
 * @return 0 on success, non-zero on error
 * 
 * @note This function calculates both total FLOPs and critical path FLOPs
 * @note Results are used for performance analysis and optimization
 */
int hicma_parsec_cholesky_stat( hicma_parsec_params_t *params )
{
    int cores = params->cores;
    int rank = params->rank;
    unsigned long *op_band = params->op_band;
    unsigned long *op_offband = params->op_offband;
    unsigned long *op_path = params->op_path;
    unsigned long *op_offpath = params->op_offpath;

    /* Sum ops in a process */
    for( int i = 1; i < cores; i++) {
        op_band[0] += op_band[i];
        op_offband[0] += op_offband[i];
        op_path[0] += op_path[i];
        op_offpath[0] += op_offpath[i];
    }

    /* Reduce to process 0 */
    MPI_Reduce(&op_band[0], &params->total_band, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&op_offband[0], &params->total_offband, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&op_path[0], &params->total_path, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&op_offpath[0], &params->total_offpath, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if( 0 == rank ) {
        assert(params->total_path + params->total_offpath == params->total_band + params->total_offband);
    }

    /* Only process 0 is the right one */
    params->total_flops = params->total_path + params->total_offpath;

    /* Time for critical path */
    MPI_Allreduce(MPI_IN_PLACE, &params->potrf_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &params->trsm_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &params->syrk_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    params->critical_path_time = params->potrf_time + params->trsm_time + params->syrk_time;

    /* Message size for the trsm in the critical path */ 
    MPI_Reduce(&params->critical_path_trsm_message, &params->total_critical_path_trsm_message, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    return 0;
}


/**
 * @brief Free all allocated memory and cleanup data structures
 * 
 * This function performs comprehensive cleanup of all allocated memory,
 * including matrix descriptors, analysis data, kernel parameters, and
 * PaRSEC data collections. It ensures proper memory deallocation
 * and prevents memory leaks.
 * 
 * @param parsec Pointer to PaRSEC context
 * @param data Pointer to HICMA data structure to cleanup
 * @param params Pointer to HICMA parameters structure
 * @param params_kernel Pointer to STARSH kernel parameters
 * @param analysis Pointer to matrix analysis structure
 */
void hicma_parsec_free_memory( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_matrix_analysis_t *analysis )
{
    /* ===========================================
     * Free analysis data structures
     * =========================================== */
    
    // Free sparse analysis data
    hicma_parsec_sparse_analysis_free( (parsec_tiled_matrix_t *)&data->dcA, analysis, params->NT, params->rank, params->sparse, params->check );

    /* ===========================================
     * Flush Cholesky memory
     * =========================================== */
    
    // Flush memory in Cholesky computation of data->dcA or data->dcAd
    // This API is needed separately when calling Cholesky multiple times
    hicma_parsec_memory_flush_choleksy( parsec, data, params );

    /* ===========================================
     * Destroy main matrix descriptors
     * =========================================== */
    
    // Destroy dcA matrix descriptor and its components
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&data->dcA );
    parsec_tiled_matrix_destroy( &data->dcA.band.super );
    parsec_tiled_matrix_destroy( &data->dcA.off_band.super );

    /* ===========================================
     * Destroy rank matrix descriptor (dcAr)
     * =========================================== */
    
    // Free rank matrix data and destroy descriptor
    parsec_data_free(data->dcAr.band.mat);
    parsec_data_free(data->dcAr.off_band.mat);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&data->dcAr );
    parsec_tiled_matrix_destroy( &data->dcAr.band.super );
    parsec_tiled_matrix_destroy( &data->dcAr.off_band.super );

    /* ===========================================
     * Destroy reordering matrix descriptor (dcReorder)
     * =========================================== */
    
    // Free reordering matrix if GEMM reordering is enabled
    if( params->reorder_gemm ) { 
        parsec_band_free_memory(parsec, (parsec_tiled_matrix_t *)&data->dcReorder, params, FREE_BAND_MEMORY);
        parsec_band_free_memory(parsec, (parsec_tiled_matrix_t *)&data->dcReorder, params, FREE_OFFBAND_MEMORY);
        parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&data->dcReorder );
        parsec_tiled_matrix_destroy( &data->dcReorder.band.super );
        parsec_tiled_matrix_destroy( &data->dcReorder.off_band.super );
    }

    /* dcDist */
    if( params->sparse ) {
        parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&data->dcDist );
        parsec_tiled_matrix_destroy( &data->dcDist.band.super );
        parsec_tiled_matrix_destroy( &data->dcDist.off_band.super );
    }

    /* dcRank */
#if PRINT_RANK
    if( 0 == params->sparse ) parsec_band_free_memory(parsec, (parsec_tiled_matrix_t *)&data->dcRank, params, FREE_OFFBAND_MEMORY);
#endif
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&data->dcRank );
    parsec_tiled_matrix_destroy( &data->dcRank.band.super );
    parsec_tiled_matrix_destroy( &data->dcRank.off_band.super );

    /* dcFake */
    parsec_data_free(data->dcFake.mat);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&data->dcFake);

#if GENOMICS
    /* phenotype, X, prediction */
    parsec_data_free(data->dcB.mat);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&data->dcB);

    parsec_data_free(data->dcX.mat);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&data->dcX);

    parsec_data_free(data->dcP.mat);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&data->dcP);

#endif

    if( params->check ) {
        /* dcA1 */
        parsec_data_free(data->dcA1.mat);
        parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&data->dcA1);

        /* dcA0 */
        parsec_data_free(data->dcA0.mat);
        parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&data->dcA0);
    }

    /* Free params_kernel */
    free( params_kernel->index );
    switch( params->kind_of_problem ) {
        case 0:
            starsh_randtlr_free(params_kernel->data);
            break;
        case 1: case 5:
            starsh_eddata_free(params_kernel->data);
            break;
	    case 2: case 3: case 4: case 9: case 10: case 11:
            starsh_ssdata_free(params_kernel->data);
            break;
        case 6: case 7:
            starsh_mddata_free(params_kernel->data);
            break;
        default:
            break;
    }

    free( params->rank_array );
    free( params->op_band );
    free( params->op_offband );
    free( params->op_path );
    free( params->op_offpath );
    free( params->gather_time );
    free( params->gather_time_tmp );
    free( params->decisions );
    free( params->decisions_send);
    free( params->decisions_gemm_gpu);
    free( params->norm_tile );

    /* Others */
    if (params->gpus > 0) {
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        cudaFreeHost( params->info_gpu );
#elif defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        hipHostFree( params->info_gpu );
#else
        free( params->info_gpu );
#endif
    } else {
        free( params->info_gpu );
    }

    /* GPU workspace */
#if (defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)) && GPU_BUFFER_ONCE 
    gpu_temporay_buffer_fini( data, params->kind_of_cholesky );
#endif
}

/* Flash memory in Cholesky of data->dcA, this API is needed when calling Cholesky multiple times */
void hicma_parsec_memory_flush_choleksy( parsec_context_t *parsec, 
        hicma_parsec_data_t *data, 
        hicma_parsec_params_t *params ) {
    /* If matrix is dense */
    if( params->band_size_dense >= params->NT && params->auto_band == 0 && !params->adaptive_memory ) {
#if GENOMICS
        parsec_memory_free_tile(parsec, (parsec_tiled_matrix_t*)&data->dcAd, params, 1);
#else
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        cudaHostUnregister(data->dcAd.mat);
#endif
        parsec_data_free(data->dcAd.mat);
#endif
#if PREDICTION || CHECKSOLVE
        parsec_data_free(data->dcAcpy.mat);
        parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&data->dcAcpy);
#endif
        return;
    }

#if BAND_MEMORY_CONTIGUOUS
    parsec_data_free(data->dcA.band.mat);
#else
    parsec_band_free_memory(parsec, (parsec_tiled_matrix_t *)&data->dcA, params, FREE_BAND_MEMORY);
#endif
    parsec_band_free_memory(parsec, (parsec_tiled_matrix_t *)&data->dcA, params, FREE_OFFBAND_MEMORY);
}

static uint32_t always_local_rank_of(parsec_data_collection_t * desc, ...)
{
    return desc->myrank;
}

static uint32_t always_local_rank_of_key(parsec_data_collection_t * desc, parsec_data_key_t key)
{
    (void)key;
    return desc->myrank;
}

/**
 * @brief Warm up Cholesky factorization on all available devices
 * 
 * This function performs a warm-up run of the Cholesky factorization (POTRF)
 * on all available computational devices (CPU, GPU, etc.) to initialize
 * device contexts, load kernels, and establish performance baselines.
 * This is important for accurate performance measurements.
 * 
 * @param rank Process rank (unused, for compatibility)
 * @param uplo Upper or lower triangular matrix specification
 * @param random_seed Random seed for matrix generation
 * @param parsec PaRSEC context for task execution
 * 
 * @note This function creates a small test matrix for warm-up
 * @note GPU devices are warmed up after CPU to ensure proper initialization
 */
void hicma_parsec_warmup_potrf(int rank, dplasma_enum_t uplo, int random_seed, parsec_context_t *parsec)
{
    int MB = 64;
    int NB = 64;
    int MT = 4;
    int NT = 4;
    int N = NB*NT;
    int M = MB*MT;
    int did;
    int info;

    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
        parsec_matrix_sym_block_cyclic, (&dcA, PARSEC_MATRIX_COMPLEX_DOUBLE,
                                   rank, MB, NB, M, N, 0, 0,
                                   M, N, 1, 1, uplo));
    dcA.super.super.rank_of = always_local_rank_of;
    dcA.super.super.rank_of_key = always_local_rank_of_key;

    /* Do the CPU warmup first */
    dplasma_dplgsy(parsec, (double)(N), uplo, &dcA.super, random_seed);
    parsec_taskpool_t *dpotrf = dplasma_dpotrf_New(uplo, &dcA.super, &info );
    dpotrf->devices_index_mask = 1<<0; /* Only CPU ! */
    parsec_context_add_taskpool(parsec, dpotrf);
    parsec_context_start(parsec);
    parsec_context_wait(parsec);
    dplasma_dpotrf_Destruct(dpotrf);

    /* Now do the other devices, skipping RECURSIVE */
    /* We know that there is a GPU-enabled version of this operation, so warm it up if some device is enabled */
    for(did = 2; did < (int)parsec_nb_devices; did++) {
        if(PARSEC_MATRIX_LOWER == uplo) {
            for(int i = 0; i < MT; i++) {
                for(int j = 0; j <= i; j++) {
                    parsec_data_t *dta = dcA.super.super.data_of(&dcA.super.super, i, j);
                    parsec_advise_data_on_device( dta, did, PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE );
                }
            }
        } else {
            for(int i = 0; i < MT; i++) {
                for(int j = i; j < NT; j++) {
                    parsec_data_t *dta = dcA.super.super.data_of(&dcA.super.super, i, j);
                    parsec_advise_data_on_device( dta, did, PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE );
                }
            }
        }
        dplasma_dplgsy( parsec, (double)(N), uplo,
                        (parsec_tiled_matrix_t *)&dcA, random_seed);
        dplasma_dpotrf( parsec, uplo, &dcA.super );
        parsec_devices_release_memory();
    }

    parsec_data_free(dcA.mat); dcA.mat = NULL;
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&dcA );
}

/**
 * @brief Calculate memory address offset for a specific tile in a tiled matrix
 * 
 * This function calculates the memory address offset for accessing a specific
 * tile (m,n) in a distributed tiled matrix, taking into account the process
 * grid dimensions (p,q) and the matrix layout.
 * 
 * @param dcA Pointer to the tiled matrix descriptor
 * @param m Row index of the tile
 * @param n Column index of the tile
 * @param p Number of process rows in the grid
 * @param q Number of process columns in the grid
 * @return Memory offset in bytes for the specified tile
 * 
 * @note This function is used for direct memory access to matrix tiles
 * @note The offset calculation accounts for the distributed layout
 */
size_t parsec_getaddr_cm(parsec_tiled_matrix_t *dcA, int m, int n, int p, int q)
{
    size_t mm = m + dcA->i / dcA->mb;
    size_t nn = n + dcA->j / dcA->nb;
    size_t eltsize =  (size_t)parsec_datadist_getsizeoftype(dcA->mtype);
    size_t offset = 0;

    //assert( dcA->>myrank == dcA->>get_rankof( dcA-> mm, nn) );
    mm = mm / p;
    nn = nn / q;

    offset = (size_t)(dcA->llm * dcA->nb) * nn + (size_t)(dcA->mb) * mm;
    return  (offset*eltsize);
}

/**
 * @brief Calculate floating-point operation counts for different linear algebra kernels
 * 
 * This function calculates the number of floating-point operations (FLOPs)
 * for various linear algebra kernels based on their input dimensions.
 * The operation counts are used for performance analysis and theoretical
 * performance calculations.
 * 
 * @param op Operation type ('q' for GEQRF, 'c' for POTRF, etc.)
 * @param a First dimension parameter
 * @param b Second dimension parameter (unused for some operations)
 * @param c Third dimension parameter (unused for some operations)
 * @param d Fourth dimension parameter (unused for some operations)
 * @return Number of floating-point operations
 * 
 * @note Operation counts are based on standard LAPACK/BLAS complexity analysis
 * @note Results are used for performance modeling and optimization
 */
unsigned long int hicma_parsec_op_counts(char op, unsigned long int a, unsigned long int b, unsigned long int c, unsigned long int d)
{
  unsigned long int res = 0;
  if(op == 'q') {//geqrf  if m >= n
    unsigned long int m = a;
    unsigned long int n = b;
    res = 2*m*n*n - (unsigned long int)(2*n*n*n/3.0f) + 2*m*n + (unsigned long int)(17*n/3.0f);
  } 
  else if(op == 'c') {//potrf  
    unsigned long int n = a;
    res = n*n*n/3 - n*n/2.0 + n/6 ;
  }
  else if(op == 't') {//trsm  
    unsigned long int m = a;
    unsigned long int n = b;
    int side = c; //1:left 2:right
    if(side == 1)
        res = n*m*m;
    else if(side == 2)
        res = m*n*n;
    else
        fprintf(stderr, "%s %d: invalid side:%d\n", __FILE__, __LINE__, side);
  }
  else if(op == 'm') {//gemm  
    unsigned long int m = a;
    unsigned long int n = b;
    unsigned long int k = c;
    res = m*n*k*2;
  }
  else if(op == 's') {//svd  
    unsigned long int n = a;
    res = 22*n*n*n;
  }
  else if(op == 'o') {//ormqr  
    unsigned long int m = a;
    unsigned long int n = b;
    unsigned long int k = c;
    int side = d; //1:left 2:right
    if(side == 1)
        res = 4*n*m*k-2*n*k*k+3*n*k;
    else if(side == 2)
        res = 4*n*m*k-2*m*k*k+2*m*k+n*k-k*k/2+k/2;
    else
        fprintf(stderr, "%s %d: invalid side:%d\n", __FILE__, __LINE__, side);
  }
  else if (op == 'r') {//trmm
    unsigned long int m = a;
    unsigned long int n = b;
    int side = c; //1:left 2:right
    if(side == 1) //left
        res = m*m*n;
    else if(side == 2) //right
        res = m*n*n;
    else
        fprintf(stderr, "%s %d: invalid side:%d\n", __FILE__, __LINE__, side);
  }
  return res;
}



/**
 * @brief Print symmetric matrix tile in human-readable format
 * 
 * This function extracts and prints a specific tile from a symmetric distributed
 * matrix in a formatted, human-readable way. It is primarily used for debugging
 * and verification purposes to inspect matrix contents during computation.
 * 
 * The function:
 * - Allocates temporary memory to store the tile data
 * - Converts the tile from PaRSEC format to LAPACK format
 * - Prints the matrix in MATLAB-style format with proper indexing
 * - Only prints the upper triangular part for symmetric matrices
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param dcA Pointer to the distributed symmetric matrix descriptor
 * @param node Node identifier for the tile location
 * @param p Row index of the tile in the tile grid
 * @param q Column index of the tile in the tile grid
 * 
 * @note This function allocates temporary memory that is freed before return
 * @note Output format is compatible with MATLAB matrix notation
 * @note Only the upper triangular part is printed for symmetric matrices
 */
void parsec_print_cm_sym(parsec_context_t* parsec, parsec_tiled_matrix_t *dcA,  int node, int p, int q){
    
    float *tempaa=(float *)malloc((size_t)dcA->llm*(size_t)dcA->lln*
                                        (size_t)parsec_datadist_getsizeoftype(dcA->mtype));
    hicma_parsec_Tile_to_Lapack_sym_single( parsec, (parsec_tiled_matrix_t *)dcA, tempaa, p, q);

    
    printf("\nM:%d, Node(%d): \n", dcA->llm, node);
    printf("[");
    for(int i=0;i<dcA->llm;i++){
       for(int j=0;j<=i;j++){
       //printf("[%d,%d]:(%f), ", i, j, tempaa[i*dcA->.super.llm+j]);
          //printf("%f ", tempaa[i*dcA->llm+j]);
          printf("%f ", tempaa[j*dcA->lm+i]);
          //printf("%f, ", A[j*(dcA.super.llm)+i]);
       }
       printf(";\n");
    }
    printf("]");
    free(tempaa);
}

/**
 * @brief Print general matrix tile in human-readable format
 * 
 * This function extracts and prints a specific tile from a general (non-symmetric)
 * distributed matrix in a formatted, human-readable way. It is primarily used
 * for debugging and verification purposes to inspect matrix contents during computation.
 * 
 * The function:
 * - Allocates temporary memory to store the tile data
 * - Converts the tile from PaRSEC format to LAPACK format
 * - Prints the entire matrix in MATLAB-style format with proper indexing
 * - Shows both dimensions (M x N) in the output header
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param dcA Pointer to the distributed general matrix descriptor
 * @param node Node identifier for the tile location
 * @param p Row index of the tile in the tile grid
 * @param q Column index of the tile in the tile grid
 * 
 * @note This function allocates temporary memory that is freed before return
 * @note Output format is compatible with MATLAB matrix notation
 * @note The entire matrix tile is printed (not just triangular part)
 */
void parsec_print_cm(parsec_context_t* parsec, parsec_tiled_matrix_t * dcA, int node, int p, int q){

    float *tempaa=(float *)malloc((size_t)dcA->llm*(size_t)dcA->lln*
                                        (size_t)parsec_datadist_getsizeoftype(dcA->mtype));
    hicma_parsec_Tile_to_Lapack_single( parsec, (parsec_tiled_matrix_t *)dcA, tempaa, p, q);

      
    printf("\nM:%d, N:%d, Node(%d): \n", dcA->llm, dcA->lln, node);
    printf("[");
    for(int i=0;i<dcA->llm;i++){
       for(int j=0;j<dcA->lln;j++){
       //printf("[%d,%d]:(%f), ", i, j, tempaa[i*dcA->.super.llm+j]);
          printf("%f ", tempaa[j*dcA->lm+i]);
       }
       printf(";\n");
    }
    printf("]");
    free(tempaa);
}

/**
 * @brief Print int8 matrix tile in human-readable format
 * 
 * This function extracts and prints a specific tile from a distributed matrix
 * with int8 data type in a formatted, human-readable way. It is primarily used
 * for debugging and verification purposes to inspect matrix contents during
 * computation, particularly for integer-based algorithms.
 * 
 * The function:
 * - Allocates temporary memory to store the tile data as int8_t
 * - Converts the tile from PaRSEC format to LAPACK format
 * - Prints the matrix in MATLAB-style format with integer values
 * - Shows both dimensions (M x N) in the output header
 * 
 * @param parsec PaRSEC context for distributed computation
 * @param dcA Pointer to the distributed matrix descriptor with int8 data
 * @param node Node identifier for the tile location
 * @param p Row index of the tile in the tile grid
 * @param q Column index of the tile in the tile grid
 * 
 * @note This function allocates temporary memory that is freed before return
 * @note Output format shows integer values instead of floating-point
 * @note The entire matrix tile is printed with proper integer formatting
 */
void parsec_print_cm_int8(parsec_context_t* parsec, parsec_tiled_matrix_t * dcA, int node, int p, int q){

    int8_t *tempaa=(int8_t *)malloc((size_t)dcA->llm*(size_t)dcA->lln*
                                        (size_t)parsec_datadist_getsizeoftype(dcA->mtype));
    hicma_parsec_Tile_to_Lapack_int8( parsec, (parsec_tiled_matrix_t *)dcA, tempaa, p, q);

      
    printf("\nM:%d, N:%d, Node(%d): \n", dcA->llm, dcA->lln, node);
    printf("[");
    for(int i=0;i<dcA->llm;i++){
       for(int j=0;j<dcA->lln;j++){
       //printf("[%d,%d]:(%f), ", i, j, tempaa[i*dcA->.super.llm+j]);
          printf("%d ", tempaa[j*dcA->lm+i]);
       }
       printf(";\n");
    }
    printf("]");
    free(tempaa);
}

/**
 * @brief CPU implementation of Hamming distance binary conversion
 * 
 * Converts input matrix to binary representation for Hamming distance computation.
 * For each element in the input matrix:
 * - If element equals target value: output = 0
 * - If element differs from target value: output = 1
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param aout Output matrix (int8_t) - binary representation
 * @param ain Input matrix (int8_t) - original values
 * @param lda Leading dimension of input matrix
 * @param value Target value for binary conversion
 */
void hicma_parsec_hamming_subtract_ones_CPU(int nrows, int ncols, int8_t *aout, int8_t *ain, int lda, int value)
{
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            // Convert to binary: 0 if matches target, 1 if different
            if (ain[j * lda + i] == value) {
                aout[j * lda + i] = 0;  // Element matches target value
            } else {
                aout[j * lda + i] = 1;  // Element differs from target value
            }
        }
    }
}
/**
 * @brief CPU implementation of Hamming distance identity matrix generation
 * 
 * Creates a binary identity matrix for Hamming distance computation.
 * For each element in the input matrix:
 * - If element equals target value: output = 1
 * - If element differs from target value: output = 0
 * 
 * This is the inverse operation of hamming_subtract_ones_CPU.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param aout Output matrix (uint8_t) - binary identity matrix
 * @param ain Input matrix (int8_t) - original values
 * @param lda Leading dimension of input matrix
 * @param value Target value for identity matching
 */
void hicma_parsec_hamming_get_id_matrix_CPU(int nrows, int ncols, uint8_t *aout, int8_t *ain, int lda, int value)
{
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            // Create identity matrix: 1 if matches target, 0 if different
            if (ain[j * lda + i] == value) {
                aout[j * lda + i] = 1;  // Element matches target value
            } else {
                aout[j * lda + i] = 0;  // Element differs from target value
            }
        }
    }
}

/**
 * @brief CPU implementation of integer memory copy
 * 
 * Performs a simple memory copy operation for integer arrays.
 * This is a utility function for copying integer data between
 * source and destination arrays.
 * 
 * @param nrows Number of rows in the arrays
 * @param ncols Number of columns in the arrays
 * @param _src Source array (int*)
 * @param _dest Destination array (int*)
 */
void hicma_parsec_memcpy_int32_CPU( int nrows, int ncols, int *_src, int *_dest) {
    memcpy(_dest, _src, nrows*ncols*sizeof(int));
}

/**
 * @brief CPU implementation of Hamming distance matrix merging
 * 
 * Merges two Hamming distance matrices by adding corresponding elements.
 * This operation is used to combine partial Hamming distance computations
 * or to accumulate distances from multiple binary conversions.
 * 
 * Note: The operation performs aout[j*lda+i] += ain[i*lda+j], which
 * adds the transpose of the input matrix to the output matrix.
 * 
 * @param nrows Number of rows in the matrices
 * @param ncols Number of columns in the matrices
 * @param aout Output matrix (int) - accumulates the merged result
 * @param ain Input matrix (int) - values to add (transposed)
 * @param lda Leading dimension of both matrices
 */
void hicma_parsec_hamming_merge_CPU(int nrows, int ncols, int *aout, int *ain, int lda) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            // Add transposed input matrix to output matrix
            // This accumulates Hamming distances from different binary conversions
            aout[j * lda + i] += ain[i * lda + j];
        }
    }
}



#if 0 
extern void dgesdd_qc_(char *jobz, lapack_int *m, 
        lapack_int *n, double* a, lapack_int *lda,
        double* s, double* u, lapack_int *ldu,
        double* vt, lapack_int *ldvt, double* work,
        lapack_int *lwork, lapack_int* iwork, lapack_int *info );

lapack_int LAPACKE_dgesdd_work( int matrix_layout, char jobz, lapack_int m,
        lapack_int n, double* a, lapack_int lda,
        double* s, double* u, lapack_int ldu,
        double* vt, lapack_int ldvt, double* work,
        lapack_int lwork, lapack_int* iwork ) {
        lapack_int info = 0;
        /* Call LAPACK function and adjust info */
        dgesdd_qc_( &jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work,
                        &lwork, iwork, &info );
        if( info < 0 ) {
                info = info - 1;
        }
        return info;
}

extern void dorgqr_qc_(lapack_int *m, lapack_int *n,
        lapack_int *k, double* a, lapack_int *lda,
        const double* tau, double* work,
        lapack_int *lwork, lapack_int *info );

lapack_int LAPACKE_dorgqr_work( int matrix_layout, lapack_int m, lapack_int n,
        lapack_int k, double* a, lapack_int lda,
        const double* tau, double* work,
        lapack_int lwork )
{
    lapack_int info = 0;
    /* Call LAPACK function and adjust info */
    dorgqr_qc_( &m, &n, &k, a, &lda, tau, work, &lwork, &info );
    if( info < 0 ) {
        info = info - 1;
    }
    return info;
}


extern void dgeqrf_qc_(lapack_int *m, lapack_int *n,
        double* a, lapack_int *lda, double* tau,
        double* work, lapack_int *lwork, lapack_int *info );

lapack_int LAPACKE_dgeqrf_work( int matrix_layout, lapack_int m, lapack_int n,
        double* a, lapack_int lda, double* tau,
        double* work, lapack_int lwork )
{
    lapack_int info = 0;
    printf("self dgeqrf\n");
    /* Call LAPACK function and adjust info */
    dgeqrf_qc_( &m, &n, a, &lda, tau, work, &lwork, &info );
    if( info < 0 ) {
        info = info - 1;
    }
    return info;
}

#endif


#if 0

void DLASCL(char* type, lapack_int* kl, lapack_int* ku, double* cfrom,
               double* cto, lapack_int* m, lapack_int* n, double* a,
               lapack_int* lda, lapack_int* info) {
    lapack_int i, j;
    double scale_factor;

    // Initialize info to 0 (success)
    *info = 0;

    // Check for invalid inputs
    if (*cfrom == 0.0 || isnan(*cfrom) || isnan(*cto) || isinf(*cfrom) || isinf(*cto)) {
        *info = -4;  // Indicating an illegal value for parameter CFROM
        return;
    }

    // Compute scaling factor
    scale_factor = (*cto) / (*cfrom);

    // Scale the matrix A
    for (j = 0; j < *n; j++) {
        for (i = 0; i < *m; i++) {
            a[i + j * (*lda)] *= scale_factor;
        }
    }
}

void dlascl_(char* type, lapack_int* kl, lapack_int* ku, double* cfrom,
               double* cto, lapack_int* m, lapack_int* n, double* a,
               lapack_int* lda, lapack_int* info) {
    //printf("%s %d %d %lf %lf %d %d %d\n", type, *kl, *ku, *cfrom, *cto, *m, *n, *lda);
    //LAPACK_dlascl( type, kl, ku, cfrom, cto, m, n, a, lda, info);
    //return;

    lapack_int i, j;
    double scale_factor;

    // Initialize info to 0 (success)
    *info = 0;

    // Check for invalid inputs
    if (*cfrom == 0.0 || isnan(*cfrom) || isnan(*cto) || isinf(*cfrom) || isinf(*cto)) {
        *info = -4;  // Indicating an illegal value for parameter CFROM
    	printf("%s %d %d %lf %lf %d %d %d\n", type, *kl, *ku, *cfrom, *cto, *m, *n, *lda);
        return;
    }

    // Compute scaling factor
    scale_factor = (*cto) / (*cfrom);

    // Scale the matrix A
    for (i = 0; i < *m; i++) {
        for (j = 0; j < *n; j++) {
            //a[j + i * (*lda)] *= scale_factor;
            a[i + j * (*lda)] *= scale_factor;
        }
    }
}

#endif

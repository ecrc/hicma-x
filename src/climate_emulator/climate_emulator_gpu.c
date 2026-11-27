/**
 * @file climate_emulator_gpu.c
 * @brief HICMA PaRSEC climate emulator GPU acceleration implementation
 * 
 * This file contains GPU-accelerated versions of the climate emulator functions,
 * including forward and inverse spherical harmonic transforms using CUDA/HIP.
 * 
 * The GPU implementation provides significant performance improvements for
 * large-scale climate data processing through parallel matrix operations
 * and optimized memory management.
 */

#include "climate_emulator.h"

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

/* ============================================================================
 * GPU workspace management functions
 * ============================================================================ */

/**
 * @brief Look up GPU workspace for a specific device and stream
 * 
 * Finds the appropriate GPU workspace for a given CUDA device and execution
 * stream. This function traverses the device and stream arrays to locate
 * the correct workspace for GPU operations.
 * 
 * @param[in] cuda_device CUDA device module
 * @param[in] cuda_stream CUDA execution stream
 * @param[in] ws GPU workspace structure
 * @return Pointer to the appropriate stream workspace
 */
climate_emulator_stream_workspace_t *climate_emulator_lookup_gpu_workspace( parsec_device_cuda_module_t *cuda_device,
                                                      parsec_cuda_exec_stream_t *cuda_stream,
                                                      climate_emulator_workspace_t *ws ) {
    int i, j;

    /* Search for the specified CUDA device */
    for(i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( NULL == device ) continue;
        if( device->type != PARSEC_DEV_CUDA && device->type != PARSEC_DEV_HIP ) continue;
        parsec_device_cuda_module_t *cuda_device_compare = (parsec_device_cuda_module_t*)device;

        if(cuda_device->cuda_index == cuda_device_compare->cuda_index)
            break;
    }

    /* Search for the specified execution stream */
    /* Note: streams 0 and 1 are reserved for h2d and d2h operations */
    for(j = 2; j < cuda_device->super.max_exec_streams; j++) {
        if( cuda_stream == (parsec_cuda_exec_stream_t *)cuda_device->super.exec_stream[j] )
            break;
    }

    return &ws->gpu_workspace[i].stream_workspace[j];
}

/**
 * @brief Allocate memory for GPU workspace
 * 
 * Allocates memory for the GPU workspace structure, including arrays for
 * multiple devices and streams. This function sets up the memory hierarchy
 * needed for GPU operations.
 * 
 * @param[in,out] ws Pointer to workspace pointer (will be allocated)
 */
static void climate_emulator_workspace_memory_allocate( climate_emulator_workspace_t **ws ) {
    *ws = (climate_emulator_workspace_t *)malloc( sizeof(climate_emulator_workspace_t) );
    (*ws)->gpu_workspace = (climate_emulator_gpu_workspace_t *)malloc( parsec_nb_devices * sizeof(climate_emulator_gpu_workspace_t) );

    for( int i = 0; i < parsec_nb_devices; i++ ) {
        (*ws)->gpu_workspace[i].stream_workspace = (climate_emulator_stream_workspace_t *)malloc( PARSEC_GPU_MAX_STREAMS * sizeof(climate_emulator_stream_workspace_t) );
        (*ws)->gpu_workspace[i].cuda_device = (parsec_device_cuda_module_t *)malloc( sizeof(parsec_device_cuda_module_t) );
    }
}

/* ============================================================================
 * GPU initialization and finalization
 * ============================================================================ */

/**
 * @brief Initialize GPU for climate emulator
 * 
 * Sets up GPU resources including cuBLAS handles, memory allocation,
 * and stream management for GPU-accelerated climate computations.
 * 
 * This function initializes all available GPU devices and creates
 * execution streams with associated cuBLAS handles and memory buffers.
 * 
 * @param[in] parsec PaRSEC context
 * @param[in] gb Climate emulator structure
 * @param[in] params HICMA PaRSEC parameters
 */
void climate_emulator_gpu_init(parsec_context_t *parsec, climate_emulator_struct_t *gb, hicma_parsec_params_t *params) {
    /* Allocate memory for GPU workspace */
    climate_emulator_workspace_memory_allocate( &gb->ws );

    /* Traverse all available GPU devices */
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( NULL == device ) continue;
        if( device->type != PARSEC_DEV_CUDA && device->type != PARSEC_DEV_HIP ) continue;

        /* Set the current CUDA device */
        parsec_device_cuda_module_t *cuda_device = (parsec_device_cuda_module_t*)device;
        cudaSetDevice(cuda_device->cuda_index);
        gb->ws->gpu_workspace[i].cuda_device = cuda_device;

        /* Initialize all execution streams for this device */
        for(int j = 0; j < cuda_device->super.max_exec_streams; j++) {
            /* Skip streams 0 and 1 (reserved for h2d and d2h operations) */
            if( j <= 1 ) continue;

            parsec_cuda_exec_stream_t* cuda_stream = (parsec_cuda_exec_stream_t*)cuda_device->super.exec_stream[j];

            cublasStatus_t status;
            cudaError_t cudaStatus;
            cublasHandle_t handle;

            /* Create cuBLAS handle for this stream */
            status = cublasCreate(&handle);
            cublasSetStream( handle, cuda_stream->cuda_stream );
            assert(CUBLAS_STATUS_SUCCESS == status);
            gb->ws->gpu_workspace[i].stream_workspace[j].handle = handle;

            /* Allocate GPU memory buffers for this stream */
            /* Gmtheta_r buffer: f_data_M * Ep_N elements */
            gb->ws->gpu_workspace[i].stream_workspace[j].Gmtheta_r = zone_malloc( cuda_device->super.memory, gb->f_data_M * gb->Ep_N * sizeof(cuDoubleComplex) );
            assert(NULL != gb->ws->gpu_workspace[i].stream_workspace[j].Gmtheta_r);

            /* Fmnm buffer: Et1_M * Ep_N elements */
            gb->ws->gpu_workspace[i].stream_workspace[j].Fmnm = zone_malloc( cuda_device->super.memory, gb->Et1_M * gb->Ep_N * sizeof(cuDoubleComplex) );
            assert(NULL != gb->ws->gpu_workspace[i].stream_workspace[j].Fmnm);

            /* tmp1 buffer: Et2_M * P_N elements */
            gb->ws->gpu_workspace[i].stream_workspace[j].tmp1 = zone_malloc( cuda_device->super.memory, gb->Et2_M * gb->P_N * sizeof(cuDoubleComplex) );
            assert(NULL != gb->ws->gpu_workspace[i].stream_workspace[j].tmp1);

            /* tmp2 buffer: Et2_M * Ep_N elements */
            gb->ws->gpu_workspace[i].stream_workspace[j].tmp2 = zone_malloc( cuda_device->super.memory, gb->Et2_M * gb->Ep_N * sizeof(cuDoubleComplex) );
            assert(NULL != gb->ws->gpu_workspace[i].stream_workspace[j].tmp2);
        }
    }
}

/* ============================================================================
 * GPU memory cleanup functions
 * ============================================================================ */

/**
 * @brief Free GPU workspace memory
 * 
 * Destroys cuBLAS handles and frees allocated GPU memory buffers
 * for all devices and streams. This function should be called
 * before program termination to prevent memory leaks.
 * 
 * The cleanup process includes:
 * - Destroying cuBLAS handles for all streams
 * - Freeing GPU memory buffers (Gmtheta_r, Fmnm, tmp1, tmp2)
 * - Freeing CPU memory for workspace structures
 * 
 * @param[in] ws GPU workspace structure to clean up
 */
void climate_emulator_workspace_memory_free( climate_emulator_workspace_t *ws)
{
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( NULL == device ) continue;
        if( device->type != PARSEC_DEV_CUDA && device->type != PARSEC_DEV_HIP ) continue;

        for(int j = 0; j < ws->gpu_workspace[i].cuda_device->super.max_exec_streams; j++) {
            /* Skip streams 0 and 1 (reserved for h2d and d2h operations) */
            if( j <= 1 ) continue;

            /* Destroy cuBLAS handle */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].handle ) {
                cublasStatus_t status = cublasDestroy(ws->gpu_workspace[i].stream_workspace[j].handle);
                assert(status == CUBLAS_STATUS_SUCCESS);
            }

            /* Free GPU memory buffers */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].Gmtheta_r ) {
                zone_free( ws->gpu_workspace[i].cuda_device->super.memory, ws->gpu_workspace[i].stream_workspace[j].Gmtheta_r);
            }

            if( NULL != ws->gpu_workspace[i].stream_workspace[j].Fmnm ) {
                zone_free( ws->gpu_workspace[i].cuda_device->super.memory, ws->gpu_workspace[i].stream_workspace[j].Fmnm);
            }

            if( NULL != ws->gpu_workspace[i].stream_workspace[j].tmp1 ) {
                zone_free( ws->gpu_workspace[i].cuda_device->super.memory, ws->gpu_workspace[i].stream_workspace[j].tmp1);
            }

            if( NULL != ws->gpu_workspace[i].stream_workspace[j].tmp2 ) {
                zone_free( ws->gpu_workspace[i].cuda_device->super.memory, ws->gpu_workspace[i].stream_workspace[j].tmp2 );
            }
        }
    }

    /* Free CPU memory for workspace structures */
    for( int i = 0; i < parsec_nb_devices; i++ ) {
        free( ws->gpu_workspace[i].stream_workspace );
    }

    free( ws->gpu_workspace );
    free( ws );
}

/**
 * @brief Finalize GPU for climate emulator
 * 
 * Cleans up all GPU resources including cuBLAS handles and allocated
 * memory buffers. This function should be called before program
 * termination to prevent memory leaks.
 * 
 * @param[in] gb Climate emulator structure
 */
void climate_emulator_gpu_fini( climate_emulator_struct_t *gb ) {
    if( gb->gpus > 0 ) {
        climate_emulator_workspace_memory_free( gb->ws );
    }
}

/* ============================================================================
 * GPU-accelerated forward computation
 * ============================================================================ */

/**
 * @brief GPU core forward computation function for spherical harmonic transform
 * 
 * GPU-accelerated version of the forward spherical harmonic transform.
 * Performs matrix operations on GPU using cuBLAS for improved performance.
 * 
 * The algorithm follows the same steps as the CPU version:
 * 1. Compute Gmtheta_r = f_data * Ep
 * 2. Compute Fmnm = Et1 * Gmtheta_r + Et2 * P * Gmtheta_r * D
 * 3. Apply spherical harmonic transformations using Slmn, Ie, and Io
 * 4. Reshape and separate real/imaginary parts into flm coefficients
 * 
 * This function leverages GPU parallelism and optimized cuBLAS operations
 * to achieve significant performance improvements over CPU implementations.
 * 
 * @param[in] flm Output spherical harmonic coefficients
 * @param[in] f_data Input climate data matrix
 * @param[in] Et1 ET1 transformation matrix
 * @param[in] Et2 ET2 transformation matrix
 * @param[in] Ep EP transformation matrix
 * @param[in] Slmn SLMN transformation matrix
 * @param[in] Ie IE transformation matrix (even)
 * @param[in] Io IO transformation matrix (odd)
 * @param[in] P P transformation matrix
 * @param[in] D D transformation matrix
 * @param[in] cuda_device CUDA device module
 * @param[in] gpu_task GPU task
 * @param[in] cuda_stream CUDA execution stream
 * @param[in] gb Climate emulator structure
 */
void climate_emulator_geqsht_forward_pre_computed_version_gpu_core(
		double *flm,
		cuDoubleComplex *f_data,
		cuDoubleComplex *Et1,
		cuDoubleComplex *Et2,
		cuDoubleComplex *Ep,
		cuDoubleComplex *Slmn,
		cuDoubleComplex *Ie,
		cuDoubleComplex *Io,
		cuDoubleComplex *P,
		cuDoubleComplex *D,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
		climate_emulator_struct_t *gb) {

	int L = gb->L;
	cuDoubleComplex alpha_complex, beta_complex;
    cublasStatus_t status;

    /* Find the appropriate GPU workspace for this device and stream */
    climate_emulator_workspace_t *ws_gpu = (climate_emulator_workspace_t *)gb->ws;
    climate_emulator_stream_workspace_t *stream_found = climate_emulator_lookup_gpu_workspace(cuda_device, cuda_stream, ws_gpu);

    /* Get cuBLAS handle and set the execution stream */
    cublasHandle_t handle = stream_found->handle;
    cublasSetStream( handle, cuda_stream->cuda_stream );

    /* Get temporary buffer pointers from workspace */
    cuDoubleComplex *Gmtheta_r = (cuDoubleComplex *)stream_found->Gmtheta_r;
    cuDoubleComplex *Fmnm = (cuDoubleComplex *)stream_found->Fmnm;
    cuDoubleComplex *tmp1 = (cuDoubleComplex *)stream_found->tmp1;
    cuDoubleComplex *tmp2 = (cuDoubleComplex *)stream_found->tmp2;

#if CLIMATE_EMULATOR_DEBUG_INFO
    climate_emulator_print_complex_GPU(f_data, 10, 10, gb->f_data_M, cuda_stream->cuda_stream);
    climate_emulator_print_double_GPU((double *)f_data, 10, 10, gb->f_data_M, cuda_stream->cuda_stream);
#endif

    /* ============================================================================
     * Step 1: Compute Gmtheta_r = f_data * Ep
     * ============================================================================ */
    assert(gb->f_data_N == gb->Ep_M);
    alpha_complex = make_cuDoubleComplex(1.0, 0);
    beta_complex = make_cuDoubleComplex(0.0, 0);
    int Gmtheta_r_M = gb->f_data_M;
    int Gmtheta_r_N = gb->Ep_N;
    int Gmtheta_r_K = gb->f_data_N;
    
    /* Perform GPU matrix multiplication using cuBLAS */
    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            Gmtheta_r_M, Gmtheta_r_N, Gmtheta_r_K,
            &alpha_complex, f_data, Gmtheta_r_M,
                            Ep, Gmtheta_r_K,
            &beta_complex, Gmtheta_r, Gmtheta_r_M);

#if CLIMATE_EMULATOR_DEBUG_INFO
    printf("Gmtheta_r\n");
    climate_emulator_sum_complex_core(Gmtheta_r, Gmtheta_r_M, Gmtheta_r_N);
    climate_emulator_print_complex_GPU(Gmtheta_r, 10, 10, Gmtheta_r_M, cuda_stream->cuda_stream);
#endif

	/* ============================================================================
	 * Step 2: Compute Fmnm = Et1 * Gmtheta_r + Et2 * P * Gmtheta_r * D
	 * ============================================================================ */
	
	/* First term: Fmnm = Et1 * Gmtheta_r */
    assert(gb->Et1_N == Gmtheta_r_M);
    int Fmnm_M = gb->Et1_M;
    int Fmnm_N = Gmtheta_r_N;
    int Fmnm_K = Gmtheta_r_M;
    
    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            Fmnm_M, Fmnm_N, Fmnm_K,
            &alpha_complex, Et1, Fmnm_M,
                            Gmtheta_r, Fmnm_K,
            &beta_complex, Fmnm, Fmnm_M);

#if CLIMATE_EMULATOR_DEBUG_INFO
    printf("Et1*Gmtheta_r\n");
    climate_emulator_sum_complex_core(Fmnm, Fmnm_M, Fmnm_N);
#endif

	/* Second term: Fmnm += Et2 * P * Gmtheta_r * D */
	/* Matrix dimensions for the chain of multiplications:
	 * Et2: (2L-1) x (L-1)
	 * P: (L-1) x (L+1)  
	 * Gmtheta_r: (L+1) x (2L-1)
	 * D: (2L-1) x (2L-1) */

    /* Step 2a: tmp1 = Et2 * P */
	assert(gb->Et2_N == gb->P_M);
    int tmp1_M = gb->Et2_M;
    int tmp1_N = gb->P_N;
    int tmp1_K = gb->P_M; 
    
    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            tmp1_M, tmp1_N, tmp1_K,
            &alpha_complex, Et2, tmp1_M,
                           P, tmp1_K,
            &beta_complex, tmp1, tmp1_M); 

#if CLIMATE_EMULATOR_DEBUG_INFO
    printf("Et2*P\n");
    climate_emulator_sum_complex_core(tmp1, tmp1_M, tmp1_N);
#endif

    /* Step 2b: tmp2 = tmp1 * Gmtheta_r */
	assert(tmp1_N == Gmtheta_r_M);
    int tmp2_M = tmp1_M;
    int tmp2_N = Gmtheta_r_N;
    int tmp2_K = Gmtheta_r_M;
    
    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            tmp2_M, tmp2_N, tmp2_K,
            &alpha_complex, tmp1, tmp2_M,
                            Gmtheta_r, tmp2_K,
            &beta_complex,  tmp2, tmp2_M);

#if CLIMATE_EMULATOR_DEBUG_INFO
    printf("Et2*P*Gmtheta_r\n");
    climate_emulator_sum_complex_core(tmp2, tmp2_M, tmp2_N);
#endif

    /* Step 2c: Fmnm += tmp2 * D */
	assert(Fmnm_M == tmp2_M);
	assert(Fmnm_N == gb->D_N);
	assert(tmp2_N == gb->D_M);
	Fmnm_K = tmp2_N;
	beta_complex = make_cuDoubleComplex(1.0, 0);  /* Add to existing Fmnm */
    
    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            Fmnm_M, Fmnm_N, Fmnm_K,
            &alpha_complex, tmp2, Fmnm_M,
                           D, Fmnm_K,
            &beta_complex, Fmnm, Fmnm_M);

#if CLIMATE_EMULATOR_DEBUG_INFO
    printf("Fmnm\n");
    climate_emulator_sum_complex_core(Fmnm, Fmnm_M, Fmnm_N);
    climate_emulator_print_complex_GPU(Fmnm, 10, 10, Fmnm_M, cuda_stream->cuda_stream); 
#endif

	/* ============================================================================
	 * Step 3: Apply spherical harmonic transformations
	 * Compute flmn_matrix using Slmn, Ie, and Io matrices
	 * ============================================================================ */
    assert(gb->Slmn_N == gb->Ie_M);
    assert(gb->Ie_N == Fmnm_M);

	int flmn_matrix_M = gb->L;
	int flmn_matrix_N = gb->L;

	cuDoubleComplex *flmn_matrix = tmp1;  /* Reuse tmp1 buffer for flmn_matrix */
	cuDoubleComplex *Fmnm_tmp;
	cuDoubleComplex *Slmn_tmp;
	cuDoubleComplex *multipy_tmp = tmp2 + Fmnm_M + gb->Slmn_N;

	/* Loop over spherical harmonic orders m */
	for(int m = 0; m < L; m++) {
        Fmnm_tmp = Fmnm + (L+m-1)*Fmnm_M;

		/* Process even and odd orders differently */
		if( 0 == m % 2) { 
			/* Even order: use Ie transformation matrix */
			for(int n = m; n < L; n++) {
                Slmn_tmp = Slmn + climate_emulator_getSingleIndex(n, m);

				/* multipy_tmp = Ie * Fmnm_tmp */
				alpha_complex = make_cuDoubleComplex(1.0, 0);
				beta_complex = make_cuDoubleComplex(0.0, 0);
				cublasZgemv(handle, CUBLAS_OP_N,
						gb->Ie_M, gb->Ie_N,
						&alpha_complex, Ie, gb->Ie_M,
						Fmnm_tmp, 1,
						&beta_complex, multipy_tmp, 1);

				/* flmn_matrix(ell+1,m+1) = Slmn_tmp * multipy_tmp */
				cublasZdotu(handle, gb->Slmn_N, Slmn_tmp, gb->Slmn_M, multipy_tmp, 1, &flmn_matrix[m*flmn_matrix_M+n]);
			}
		} else {
			/* Odd order: use Io transformation matrix */
			for(int n = m; n < L; n++) {
                Slmn_tmp = Slmn + climate_emulator_getSingleIndex(n, m);

                /* multipy_tmp = Io * Fmnm_tmp */
                alpha_complex = make_cuDoubleComplex(1.0, 0);
                beta_complex = make_cuDoubleComplex(0.0, 0);
                cublasZgemv(handle, CUBLAS_OP_N,
                        gb->Io_M, gb->Io_N,
                        &alpha_complex, Io, gb->Io_M,
                        Fmnm_tmp, 1,
                        &beta_complex, multipy_tmp, 1);

                /* flmn_matrix(ell+1,m+1) = Slmn_tmp * multipy_tmp */
                cublasZdotu(handle, gb->Slmn_N, Slmn_tmp, gb->Slmn_M, multipy_tmp, 1, &flmn_matrix[m*flmn_matrix_M+n]);
			}
		}
	}

#if CLIMATE_EMULATOR_DEBUG_INFO
    printf("flmn_matrix\n");
    climate_emulator_sum_complex_core(flmn_matrix, flmn_matrix_M, flmn_matrix_N);
    climate_emulator_print_matrix_col_complex(flmn_matrix, 10, 10, flmn_matrix_M);
#endif

    /* ============================================================================
	 * Step 4: Reshape and separate real and imaginary parts
	 * Convert complex flmn_matrix to real flm coefficients
	 * ============================================================================ */
    climate_emulator_reshape_GPU(flm, flmn_matrix, L, flmn_matrix_M, cuda_stream->cuda_stream);

#if CLIMATE_EMULATOR_DEBUG_INFO
    printf("flm\n");
    climate_emulator_sum_double_core(flm, gb->flm_M, gb->flm_N);
	climate_emulator_print_matrix_col_double(flm, 1, 20, 1); 
#endif
}


// f_spatial  = geqsht_inverse_pre_computed_version(flm,Ylm_matrix_full,Ep_inverse,reshape_matrix,L);
void climate_emulator_geqsht_inverse_pre_computed_version_gpu_core(
        int tile_id,
        double *flm,
        double *f_spatial,
        double *Zlm,
        double *SC,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        climate_emulator_struct_t *gb) {

    int L = gb->L;
    int index_Zlm, index_flm;
    cublasStatus_t status;

    /* Find workspace */
    climate_emulator_workspace_t *ws_gpu = (climate_emulator_workspace_t *)gb->ws;
    climate_emulator_stream_workspace_t *stream_found = climate_emulator_lookup_gpu_workspace(cuda_device, cuda_stream, ws_gpu);

    /* Get handle_cublas */
    //cublasHandle_t handle = stream_found->handle_cublas_tensor;
    cublasHandle_t handle = stream_found->handle;
    cublasSetStream( handle, cuda_stream->cuda_stream );

    // Temp buffer
    int Smt_M = L+1;
    int Smt_N = 2*L-1;
    double *Smt = (double *)stream_found->Gmtheta_r;
    cudaMemset(Smt, 0, Smt_M * Smt_N * sizeof(double));

    //printf("Summation over spherical harmonics\n");
    double alpha_axpy;
    for(int m = -(L-1); m < L; m++) {
        for(int n = abs(m); n < L; n++) {
            //Smt(:,m+L) = Smt(:,m+L) + Zlm_matrix(:,climate_emulator_getSingleIndex(ell, abs(m)))*flm(ell^2+ell+m+1);
            index_Zlm = climate_emulator_getSingleIndex(n, abs(m));
            index_flm = n*n+n+m;
            alpha_axpy = ((double *)gb->desc_flm.super.super.data_of(&gb->desc_flm.super.super, 0, tile_id)->device_copies[0]->device_private)[index_flm];
            cublasDaxpy(handle, Smt_M, &alpha_axpy, Zlm+index_Zlm*gb->Zlm_M, 1, Smt+(m+L-1)*Smt_M, 1);
        }
    }

    //printf("Summation over sines and cosines basis\n");
    int f_spatial_M = Smt_M;
    int f_spatial_N = gb->SC_N;
    int f_spatial_K = Smt_N;
    double alpha = 1.0;
    double beta = 0.0;
    assert(Smt_N == gb->SC_M);
    assert(gb->f_spatial_M == f_spatial_M);
    assert(gb->f_spatial_N == f_spatial_N);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            f_spatial_M, f_spatial_N, f_spatial_K,
            &alpha, Smt, f_spatial_M,
                    SC, f_spatial_K,
            &beta, f_spatial, f_spatial_M);

}

#endif

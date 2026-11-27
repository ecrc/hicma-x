
#include "climate_emulator.h"


int climate_emulator_getSingleIndex(int n, int m) {
    return n * (n + 1) / 2 + m;
}

double climate_emulator_sum_double_core(double *A, int M, int N) {
    double sum = 0.0;
    for(int j = 0; j < N; j++) {
        for(int i = 0; i < M; i++) {
            sum += A[j*M+i];
        }
    }
    return sum;
}

complex double climate_emulator_sum_complex_core(complex double *A, int M, int N) {
    complex double sum = 0.0;
    for(int j = 0; j < N; j++) {
        for(int i = 0; i < M; i++) {
            sum += A[j*M+i];
        }
    }
    return sum;
}

void climate_emulator_print_matrix_col_double(double *data, int M, int N, int lda) {
    printf("\n\n");
    printf("lda %d\n", lda);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.21le ", data[j*lda+i]);
        }
        printf("\n");
    }
    printf("\n\n");
}

void climate_emulator_print_matrix_col_complex(complex double *data, int M, int N, int lda) {
    printf("\n\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.6le + %.6lei ", creal(data[j*lda+i]), cimag(data[j*lda+i]));
        }
        printf("\n");
    }
    printf("\n\n");
}

/* Init climate_emulator */
parsec_context_t *climate_emulator_init( int argc, char ** argv,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_data_t *data) {

    parsec_context_t * parsec = hicma_parsec_init(argc, argv, params, params_kernel, data);

    /* Set default value */
    params->band_size_dense = params->NT;
    params->band_size_dense_dp = params->NT;
    params->band_size_dense_sp = params->NT;
    params->band_size_dense_hp = params->NT;
    params->kind_of_cholesky = DENSE_MP_GPU;
    params->band_size_dist = 0;
    params->auto_band = 0;

    return parsec;
} 


static void climate_emulator_print_initial(climate_emulator_struct_t *gb) {
    if( 0 == gb->rank ) {
        fprintf(stderr, BLU "\nCLIMATE_EMULATOR Parameters:\n");
        fprintf(stderr, "L %d T %d NB %d gpus %d nodes %d time_slot_per_file %d num_file %d file_per_node %d\n",
                gb->L, gb->T, gb->NB, gb->gpus, gb->nodes, gb->time_slot_per_file, gb->num_file, gb->file_per_node);
        fprintf(stderr, "f_data_M %d f_data_N %d Et1_M %d Et1_N %d Et2_M %d Et2_N %d Ep_M %d Ep_N %d\n",
                gb->f_data_M, gb->f_data_N, gb->Et1_M, gb->Et1_N, gb->Et2_M, gb->Et2_N, gb->Ep_M, gb->Ep_N);
        fprintf(stderr, "Slmn_M %d Slmn_N %d Ie_M %d Ie_N %d Io_M %d Io_N %d P_M %d P_N %d\n",
                gb->Slmn_M, gb->Slmn_N, gb->Ie_M, gb->Ie_N, gb->Io_M, gb->Io_N, gb->P_M, gb->P_N);
        fprintf(stderr, "D_M %d D_N %d flm_M %d flm_N %d Zlm_M %d Zlm_N %d SC_M %d SC_N %d\n",
                gb->D_M, gb->D_N, gb->flm_M, gb->flm_N, gb->Zlm_M, gb->Zlm_N, gb->SC_M, gb->SC_N);
        fprintf(stderr, "f_spatial_M %d f_spatial_N %d flmT_M %d flmT_N %d flmT_numNB %d flops_forward %le flops_backward %le\n",
                gb->f_spatial_M, gb->f_spatial_N, gb->flmT_M, gb->flmT_N, gb->flmT_numNB, gb->flops_forward, gb->flops_backward);
        fprintf(stderr, "\n"RESET);
    }
}


void climate_emulator_fini(parsec_context_t *parsec,
        climate_emulator_struct_t *gb,
        hicma_parsec_params_t *params) {

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    if(gb->gpus > 0) {
        climate_emulator_gpu_fini(gb);
    }
#endif

#if !CLIMATE_EMULATOR_READ_FROM_FILE
    parsec_data_free(gb->desc_f_data.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&gb->desc_f_data);

    parsec_data_free(gb->desc_Et1.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&gb->desc_Et1);

    parsec_data_free(gb->desc_Et2.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&gb->desc_Et2);

    parsec_data_free(gb->desc_Ep.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&gb->desc_Ep);

    parsec_data_free(gb->desc_Slmn.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&gb->desc_Slmn);

    parsec_data_free(gb->desc_Ie.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&gb->desc_Ie);

    parsec_data_free(gb->desc_Io.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&gb->desc_Io);

    parsec_data_free(gb->desc_P.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&gb->desc_P);

    parsec_data_free(gb->desc_D.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&gb->desc_D);

    parsec_data_free(gb->desc_flm.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&gb->desc_flm);
#endif

    parsec_data_free(gb->desc_A.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&gb->desc_A);

#if CLIMATE_EMULATOR_ENABLE_INVERSE
    parsec_data_free(gb->desc_flmERA.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&gb->desc_flmERA);

    parsec_data_free(gb->desc_Zlm.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&gb->desc_Zlm);

    parsec_data_free(gb->desc_SC.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&gb->desc_SC);

    parsec_data_free(gb->desc_f_spatial.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&gb->desc_f_spatial);
#endif

    free(gb->phi);
}


static int read_csv_double(const char* filename, double **data, int M, int N) {
    // Allocate memory
    *data = (double *)malloc(M * N * sizeof(double));

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("File opening failed: %s", filename);
        return -1;
    }

    int status = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Assuming the CSV data is separated by commas,
            // fscanf can be used to read directly into the array.
            status = fscanf(fp, "%lf,", &(*data)[i*N+j]);
            if (status != 1) {
                fprintf(stderr, "Error reading file at row %d, column %d\n", i, j);
                fclose(fp);
                return 1;
            }
        }
    }

    fclose(fp);

#if CLIMATE_EMULATOR_DEBUG_INFO
    print_matrix_row_double(*data, hicma_parsec_min(M, 10), hicma_parsec_min(N, 10), N);
#endif

    return 0;
}


void climate_emulator_reading_data(parsec_context_t *parsec, climate_emulator_struct_t *gb, hicma_parsec_params_t *params) {
    gb->L = params->latitude;
    gb->verbose = params->verbose;
    if(0 == gb->L) gb->L = CLIMATE_EMULATOR_L_SIZE;
    gb->T = params->time_slots;
    gb->NB = params->NB;
    gb->gpus = params->gpus;
    gb->nodes = params->nodes;
    gb->rank = params->rank;
    gb->time_slot_per_file = CLIMATE_EMULATOR_TIME_SLOT_PER_FILE;
    gb->num_file = CLIMATE_EMULATOR_NUM_FILE;
    gb->file_per_node = (gb->num_file%gb->nodes)? gb->num_file/gb->nodes+1 : gb->num_file/gb->nodes;
    gb->data_dir = params->mesh_file;
    if(NULL == gb->data_dir) {
        gb->data_dir = "/home/qcao3/data";
        fprintf(stderr, RED"Warning: input files directory is missing! You can download them from https://drive.google.com/drive/u/0/folders/1RnnVJPcoeokI8J0Q5pcytIWqU2muU10t \n"RESET);
        exit(1);
    }

    int L = gb->L;
    char *cwd = gb->data_dir; 
    char filename[PATH_MAX+100];

    VERBOSE_PRINT(params->rank, params->verbose, ("L= %d time_slot= %d\n", gb->L, gb->T));

    /* Forward computation matrix dimensions and data reading */
    gb->f_data_M = L+1;
    gb->f_data_N = 2*L;
#if !CLIMATE_EMULATOR_READ_FROM_FILE
    VERBOSE_PRINT(params->rank, params->verbose, ("Reading f_data\n"));
    sprintf(filename, "%s/%d_era_data_sample.csv", cwd, CLIMATE_EMULATOR_L_SIZE);
    climate_emulator_read_csv_double2complex_timeslot(parsec, filename, &gb->desc_f_data, gb->f_data_M, gb->f_data_N, params);
#endif

    gb->Et1_M = 2*L-1;
    gb->Et1_N = L+1;
#if !CLIMATE_EMULATOR_READ_FROM_FILE
    VERBOSE_PRINT(params->rank, params->verbose, ("Reading Et1\n"));
    sprintf(filename, "%s/%d_Et1.csv", cwd, CLIMATE_EMULATOR_L_SIZE);
    climate_emulator_read_csv_complex(parsec, filename, &gb->desc_Et1, gb->Et1_M, gb->Et1_N, params);
#endif

    gb->Et2_M = 2*L-1;
    gb->Et2_N = L-1;
#if !CLIMATE_EMULATOR_READ_FROM_FILE
    VERBOSE_PRINT(params->rank, params->verbose, ("Reading Et2\n"));
    sprintf(filename, "%s/%d_Et2.csv", cwd, CLIMATE_EMULATOR_L_SIZE);
    climate_emulator_read_csv_complex(parsec, filename, &gb->desc_Et2, gb->Et2_M, gb->Et2_N, params);
#endif

    gb->Ep_M = 2*L;
    gb->Ep_N = 2*L-1;
#if !CLIMATE_EMULATOR_READ_FROM_FILE
    VERBOSE_PRINT(params->rank, params->verbose, ("Reading Ep\n"));
    sprintf(filename, "%s/%d_Ep.csv", cwd, CLIMATE_EMULATOR_L_SIZE);
    climate_emulator_read_csv_complex(parsec, filename, &gb->desc_Ep, gb->Ep_M, gb->Ep_N, params);
#endif

    gb->Slmn_M = (L*L+L)/2;
    gb->Slmn_N = L;
#if !CLIMATE_EMULATOR_READ_FROM_FILE
    VERBOSE_PRINT(params->rank, params->verbose, ("Reading Slmn\n"));
    sprintf(filename, "%s/%d_Slmn.csv", cwd, CLIMATE_EMULATOR_L_SIZE);
    climate_emulator_read_csv_complex(parsec, filename, &gb->desc_Slmn, gb->Slmn_M, gb->Slmn_N, params);
#endif

    gb->Ie_M = L;
    gb->Ie_N = 2*L-1;
#if !CLIMATE_EMULATOR_READ_FROM_FILE
    VERBOSE_PRINT(params->rank, params->verbose, ("Reading Ie\n"));
    sprintf(filename, "%s/%d_Ie.csv", cwd, CLIMATE_EMULATOR_L_SIZE);
    climate_emulator_read_csv_double2complex(parsec, filename, &gb->desc_Ie, gb->Ie_M, gb->Ie_N, params);
#endif

    gb->Io_M = L;
    gb->Io_N = 2*L-1;
#if !CLIMATE_EMULATOR_READ_FROM_FILE
    VERBOSE_PRINT(params->rank, params->verbose, ("Reading Io\n"));
    sprintf(filename, "%s/%d_Io.csv", cwd, CLIMATE_EMULATOR_L_SIZE);
    climate_emulator_read_csv_double2complex(parsec, filename, &gb->desc_Io, gb->Io_M, gb->Io_N, params);
#endif

    gb->P_M = L-1;
    gb->P_N = L+1;
#if !CLIMATE_EMULATOR_READ_FROM_FILE
    VERBOSE_PRINT(params->rank, params->verbose, ("Reading P\n"));
    sprintf(filename, "%s/%d_P.csv", cwd, CLIMATE_EMULATOR_L_SIZE);
    climate_emulator_read_csv_double2complex(parsec, filename, &gb->desc_P, gb->P_M, gb->P_N, params);
#endif

    gb->D_M = 2*L-1;
    gb->D_N = 2*L-1;
#if !CLIMATE_EMULATOR_READ_FROM_FILE
    VERBOSE_PRINT(params->rank, params->verbose, ("Reading D\n"));
    sprintf(filename, "%s/%d_D.csv", cwd, CLIMATE_EMULATOR_L_SIZE);
    climate_emulator_read_csv_double2complex(parsec, filename, &gb->desc_D, gb->D_M, gb->D_N, params);
#endif

    gb->flm_M = L;
    gb->flm_N = L;
#if !CLIMATE_EMULATOR_READ_FROM_FILE
    VERBOSE_PRINT(params->rank, params->verbose, ("Reading flmERA\n"));
    climate_emulator_read_csv_double_timeslot(parsec, NULL, &gb->desc_flm, gb->flm_M, gb->flm_N, params);
    sprintf(filename, "%s/%d_flmERA.csv", cwd, CLIMATE_EMULATOR_L_SIZE);
    climate_emulator_read_csv_double_timeslot(parsec, filename, &gb->desc_flmERA, gb->flm_M, gb->flm_N, params);
#endif

    /* Backward computation matrix dimensions and data reading */
#if CLIMATE_EMULATOR_ENABLE_INVERSE
    gb->Zlm_M = L+1;
    gb->Zlm_N = (L*L+L)/2;
    VERBOSE_PRINT(params->rank, params->verbose, ("Reading Zlm\n"));
    sprintf(filename, "%s/%d_Zlm.csv", cwd, CLIMATE_EMULATOR_L_SIZE);
    climate_emulator_read_csv_double(parsec, filename, &gb->desc_Zlm, gb->Zlm_M, gb->Zlm_N, params);

    gb->SC_M = 2*L-1;
    gb->SC_N = 2*L;
    VERBOSE_PRINT(params->rank, params->verbose, ("Reading SC\n"));
    sprintf(filename, "%s/%d_SC.csv", cwd, CLIMATE_EMULATOR_L_SIZE);
    climate_emulator_read_csv_double(parsec, filename, &gb->desc_SC, gb->SC_M, gb->SC_N, params);

    VERBOSE_PRINT(params->rank, params->verbose, ("f_spatial\n"));
    gb->f_spatial_M = L+1;
    gb->f_spatial_N = 2*L;
    climate_emulator_read_csv_double_timeslot(parsec, NULL, &gb->desc_f_spatial, gb->f_spatial_M, gb->f_spatial_N, params);
#endif

    /* Initialize and allocate memory for desc_flmT */
    gb->flmT_M = L * L;
    gb->flmT_N = gb->T;
    gb->flmT_numNB = (gb->flmT_M/params->nodes%gb->NB) ? gb->flmT_M/params->nodes/gb->NB+1 : gb->flmT_M/params->nodes/gb->NB;   
#if !CLIMATE_EMULATOR_READ_FROM_FILE
    parsec_matrix_block_cyclic_init(&gb->desc_flmT, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
            params->rank, gb->NB*gb->flmT_numNB, gb->flmT_N, gb->flmT_M, gb->flmT_N, 0, 0,
            gb->flmT_M, gb->flmT_N, params->nodes, 1,
            1, 1, 0, 0);
    gb->desc_flmT.mat = parsec_data_allocate((size_t)gb->desc_flmT.super.nb_local_tiles *
                                   (size_t)gb->desc_flmT.super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(gb->desc_flmT.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&gb->desc_flmT, "desc_flmT");
#endif

    /* Initialize and allocate memory for desc_A */
    gb->A_M = L * L;
    gb->A_N = gb->T;
    parsec_matrix_block_cyclic_init(&gb->desc_A, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
            params->rank, gb->NB, gb->NB, gb->A_M, gb->A_N, 0, 0,
            gb->flmT_M, gb->flmT_N, params->P, params->nodes/params->P,
            1, 1, 0, 0);
    gb->desc_A.mat = parsec_data_allocate((size_t)gb->desc_A.super.nb_local_tiles *
                                   (size_t)gb->desc_A.super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(gb->desc_A.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&gb->desc_A, "desc_A");

    /* Initialize phi array */
    gb->phi_M = L * L;
    gb->phi_N = 3;
    gb->phi = calloc(gb->phi_M*gb->phi_N, sizeof(double));

#if CLIMATE_EMULATOR_ENABLE_INVERSE
    gb->ts_test_M = 2000;
    gb->ts_test_N = 1;
    sprintf(filename, "%s/ts_test.csv", cwd);
    read_csv_double(filename, &gb->ts_test, gb->ts_test_M, gb->ts_test_N);
#endif

    // Flops
    gb->flops_forward = 2.0*(L+1)*(2*L-1)*(2*L) // Gmtheta_r = f_data*Ep
        + 2.0*(2*L-1)*(2*L-1)*(L+1) // Fmnm = Et1*Gmtheta_r
        + 2.0*(2*L-1)*(L-1)*(L+1) // tmp1 = Et2*P
        + 2.0*(2*L-1)*(2*L-1)*(L+1)  // tmp2 = tmp1 * Gmtheta_r
        + 2.0*(2*L-1)*(2*L-1)*(2*L-1)  // Fmnm += tmp2 * D 
        + 2.0*L*L/2*(L*(2*L-1)+L);   // flmn_matrix(ell+1,m+1) = Slmn(climate_emulator_getSingleIndex(ell, m),:)*Ie*Fmnm(:,L+m)
    gb->flops_forward *= (2 * gb->T);
    gb->flops_backward = 0; 

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    if( params->gpus > 0 ) {
        climate_emulator_gpu_init(parsec, gb, params);
    }   
#endif

    // Print
    climate_emulator_print_initial(gb);

}


void climate_emulator_geqsht_forward_pre_computed_version_core(
		double *flm,
		complex double *f_data,
		complex double *Et1,
		complex double *Et2,
		complex double *Ep,
		complex double *Slmn,
		complex double *Ie,
		complex double *Io,
		complex double *P,
		complex double *D,
		complex double *Gmtheta_r,
		complex double *Fmnm,
		complex double *tmp1,
		complex double *tmp2,
		climate_emulator_struct_t *gb) {
	int L = gb->L;
	complex double alpha_complex, beta_complex;
	double alpha_double, beta_double;

	// Gmtheta_r = f_data*Ep
    assert(gb->f_data_N == gb->Ep_M);
	alpha_complex = (complex double)1.0;
	beta_complex = (complex double)0.0;
    int Gmtheta_r_M = gb->f_data_M;
    int Gmtheta_r_N = gb->Ep_N;
    int Gmtheta_r_K = gb->f_data_N;
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            Gmtheta_r_M, Gmtheta_r_N, Gmtheta_r_K,
            &alpha_complex, f_data, Gmtheta_r_M,
                            Ep, Gmtheta_r_K,
            &beta_complex, Gmtheta_r, Gmtheta_r_M);

#if CLIMATE_EMULATOR_DEBUG_INFO 
    printf("Gmtheta_r\n");
    climate_emulator_sum_complex_core(Gmtheta_r, Gmtheta_r_M, Gmtheta_r_N);
    climate_emulator_print_matrix_col_complex(Gmtheta_r, 10, 10, Gmtheta_r_M);
#endif

	// Fmnm = Et1*Gmtheta_r + Et2*P*Gmtheta_r*D;
	// Fmnm = Et1*Gmtheta_r
    assert(gb->Et1_N == Gmtheta_r_M);
    int Fmnm_M = gb->Et1_M;
    int Fmnm_N = Gmtheta_r_N;
    int Fmnm_K = Gmtheta_r_M;
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            Fmnm_M, Fmnm_N, Fmnm_K,
            &alpha_complex, Et1, Fmnm_M,
                            Gmtheta_r, Fmnm_K,
            &beta_complex, Fmnm, Fmnm_M);

#if CLIMATE_EMULATOR_DEBUG_INFO
    printf("Et1*Gmtheta_r\n");
    climate_emulator_sum_complex_core(Fmnm, Fmnm_M, Fmnm_N);
#endif

	// fmnm += Et2*P*Gmtheta_r*D
	// Et2: gb->W_M * (gb->Et_N - gb->f_data_M) : 255 * 719 
	// P: (gb->Et_N - gb->f_data_M) * gb->f_data_M  : 719 * 721
	// Gmtheta_r: gb->f_data_M * gb->Ep_N : 721 * 255
	// D: gb->Ep_N * gb->Ep_N : 255  * 255 

    // tmp1 = Et2*P
	assert(gb->Et2_N == gb->P_M);
    int tmp1_M = gb->Et2_M;
    int tmp1_N = gb->P_N;
    int tmp1_K = gb->P_M; 
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            tmp1_M, tmp1_N, tmp1_K,
            &alpha_complex, Et2, tmp1_M,
                           P, tmp1_K,
            &beta_complex, tmp1, tmp1_M); 

#if CLIMATE_EMULATOR_DEBUG_INFO
    printf("Et2*P\n");
    climate_emulator_sum_complex_core(tmp1, tmp1_M, tmp1_N);
#endif

    // tmp2 = tmp1 * Gmtheta_r 
	assert(tmp1_N == Gmtheta_r_M);
    int tmp2_M = tmp1_M;
    int tmp2_N = Gmtheta_r_N;
    int tmp2_K = Gmtheta_r_M;
    //complex double *tmp2 = (complex double *)malloc(tmp2_M * tmp2_N * sizeof(complex double));
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            tmp2_M, tmp2_N, tmp2_K,
            &alpha_complex, tmp1, tmp2_M,
                            Gmtheta_r, tmp2_K,
            &beta_complex,  tmp2, tmp2_M);

#if CLIMATE_EMULATOR_DEBUG_INFO
    printf("Et2*P*Gmtheta_r\n");
    climate_emulator_sum_complex_core(tmp2, tmp2_M, tmp2_N);
#endif

    // Fmnm += tmp2 * D 
	assert(Fmnm_M == tmp2_M);
	assert(Fmnm_N == gb->D_N);
	assert(tmp2_N == gb->D_M);
	Fmnm_K = tmp2_N;
	beta_complex = (complex double)1.0;
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            Fmnm_M, Fmnm_N, Fmnm_K,
            &alpha_complex, tmp2, Fmnm_M,
                           D, Fmnm_K,
            &beta_complex, Fmnm, Fmnm_M);

#if CLIMATE_EMULATOR_DEBUG_INFO
    printf("Fmnm\n");
    climate_emulator_sum_complex_core(Fmnm, Fmnm_M, Fmnm_N);
    climate_emulator_print_matrix_col_complex(Fmnm, 10, 10, Fmnm_M); 
    printf("flmn_matrix(ell+1,m+1) = Slmn(climate_emulator_getSingleIndex(ell, m),:)*Ie*Fmnm(:,L+m);\n");
#endif

    assert(gb->Slmn_N == gb->Ie_M);
    assert(gb->Ie_N == Fmnm_M);

	int flmn_matrix_M = gb->L;
	int flmn_matrix_N = gb->L;

	complex double *flmn_matrix = tmp1;
	complex double *Fmnm_tmp;
	complex double *Slmn_tmp;
	complex double *multipy_tmp = tmp2 + Fmnm_M + gb->Slmn_N;

	for(int m = 0; m < L; m++) {
        Fmnm_tmp = Fmnm + (L+m-1)*Fmnm_M;

#if CLIMATE_EMULATOR_DEBUG_INFO
        print_matrix_row_complex(Fmnm_tmp, 1, 10, Fmnm_N);
#endif

		if( 0 == m % 2) { 
			for(int n = m; n < L; n++) {
                Slmn_tmp = Slmn + climate_emulator_getSingleIndex(n, m);

				// multipy_tmp = Ie * Fmnm_tmp
				alpha_complex = (complex double)1.0;
				beta_complex = (complex double)0.0;
				cblas_zgemv(CblasColMajor, CblasNoTrans,
						gb->Ie_M, gb->Ie_N,
						&alpha_complex, Ie, gb->Ie_M,
						Fmnm_tmp, 1,
						&beta_complex, multipy_tmp, 1);

				// flmn_matrix(ell+1,m+1) = Slmn_tmp * multipy_tmp
				// Slmn_tmp: gb->Slmn_N = L
				// multipy_tmp: gb->Ie_M = L
				cblas_zdotu_sub(gb->Slmn_N, Slmn_tmp, gb->Slmn_M, multipy_tmp, 1, &flmn_matrix[m*flmn_matrix_M+n]);
			}
		} else {
			for(int n = m; n < L; n++) {
                // Slmn_tmp
                Slmn_tmp = Slmn + climate_emulator_getSingleIndex(n, m);

                // multipy_tmp = Io * Fmnm_tmp
                alpha_complex = (complex double)1.0;
                beta_complex = (complex double)0.0;
                cblas_zgemv(CblasColMajor, CblasNoTrans,
                        gb->Io_M, gb->Io_N,
                        &alpha_complex, Io, gb->Io_M,
                        Fmnm_tmp, 1,
                        &beta_complex, multipy_tmp, 1);

                // flmn_matrix(ell+1,m+1) = Slmn_tmp * multipy_tmp
                // Slmn_tmp: gb->Slmn_N = L
                // multipy_tmp: gb->Ie_M = L
                cblas_zdotu_sub(gb->Slmn_N, Slmn_tmp, gb->Slmn_M, multipy_tmp, 1, &flmn_matrix[m*flmn_matrix_M+n]);
			}
		}

	}

#if CLIMATE_EMULATOR_DEBUG_INFO
    printf("flmn_matrix\n");
    climate_emulator_sum_complex_core(flmn_matrix, flmn_matrix_M, flmn_matrix_N);
    climate_emulator_print_matrix_col_complex(flmn_matrix, 10, 10, flmn_matrix_M);
#endif

    // Reshaping and separation of real and imaginary parts
#if CLIMATE_EMULATOR_DEBUG_INFO
	printf("Reshaping and separation of real and imaginary parts\n");
#endif
	for(int n = 0; n < L; n++) {
		for(int m = 0; m <= n; m++) {
			flm[n*n+n+m] = creal(flmn_matrix[m*flmn_matrix_M+n]);
			if( m != 0) {
				flm[n*n+n-m] = cimag(flmn_matrix[m*flmn_matrix_M+n]);
			}
		}
	}

#if CLIMATE_EMULATOR_DEBUG_INFO
    printf("flm\n");
    climate_emulator_sum_double_core(flm, gb->flm_M, gb->flm_N);
	climate_emulator_print_matrix_col_double(flm, 1, 20, 1); 
#endif

}


/* ============================================================================
 * Inverse computation functions
 * ============================================================================ */

/**
 * @brief Core inverse computation function for spherical harmonic transform
 * 
 * This function implements the core algorithm for inverse spherical harmonic
 * transform, converting spectral coefficients back to spatial climate data.
 * 
 * The algorithm performs the following steps:
 * 1. Compute Smt matrix by summing over spherical harmonics
 * 2. Apply final transformation: f_spatial = Smt * SC
 * 
 * This function is the computational kernel that performs the actual
 * matrix operations for the inverse transform.
 * 
 * @param[in] flm Input spherical harmonic coefficients
 * @param[in] f_spatial Output spatial climate data
 * @param[in] Zlm ZLM transformation matrix
 * @param[in] SC SC transformation matrix
 * @param[in] Smt Temporary matrix for intermediate results
 * @param[in] gb Climate emulator structure containing dimensions
 */
void climate_emulator_geqsht_inverse_pre_computed_version_core(
        double *flm,
        double *f_spatial,
        double *Zlm,
        double *SC,
        double *Smt,
        climate_emulator_struct_t *gb) {

    int L = gb->L;
    int index_Zlm, index_flm;
    int Smt_M = L+1;      /* Rows: L+1 */
    int Smt_N = 2*L-1;    /* Columns: 2L-1 */
    
    /* Initialize Smt matrix with zeros */
    memset(Smt, 0, Smt_M * Smt_N * sizeof(double));

    /* ============================================================================
     * Step 1: Compute Smt matrix by summing over spherical harmonics
     * Smt(:,m+L) = Smt(:,m+L) + Zlm_matrix(:,climate_emulator_getSingleIndex(ell, abs(m)))*flm(ell^2+ell+m+1)
     * ============================================================================ */
    for(int m = -(L-1); m < L; m++) {
        for(int n = abs(m); n < L; n++) {
            /* Calculate indices for accessing Zlm and flm matrices */
            index_Zlm = climate_emulator_getSingleIndex(n, abs(m));
            index_flm = n*n+n+m;
            
            /* Add contribution to Smt matrix using BLAS axpy operation */
            cblas_daxpy(Smt_M, flm[index_flm], Zlm+index_Zlm*gb->Zlm_M, 1, Smt+(m+L-1)*Smt_M, 1);
        }
    }

#if CLIMATE_EMULATOR_DEBUG_INFO
    printf("Smt\n");
    climate_emulator_print_matrix_col_double(Smt, 10, 10, Smt_M);
    climate_emulator_sum_double_core(Smt, Smt_M, Smt_N);
#endif

    /* ============================================================================
     * Step 2: Apply final transformation to get spatial field
     * f_spatial = Smt * SC
     * ============================================================================ */
    int f_spatial_M = Smt_M;           /* Rows: L+1 */
    int f_spatial_N = gb->SC_N;        /* Columns: 2L */
    int f_spatial_K = Smt_N;           /* Inner dimension: 2L-1 */
    
    /* Verify matrix dimensions are compatible */
    assert(Smt_N == gb->SC_M);
    assert(gb->f_spatial_M == f_spatial_M);
    assert(gb->f_spatial_N == f_spatial_N);
    
    /* Perform matrix multiplication: f_spatial = Smt * SC */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            f_spatial_M, f_spatial_N, f_spatial_K,
            (double)1.0, Smt, f_spatial_M,
                         SC, f_spatial_K,
            (double)0.0, f_spatial, f_spatial_M);

#if CLIMATE_EMULATOR_DEBUG_INFO
    printf("gb->f_spatial\n");
    climate_emulator_print_matrix_col_double(f_spatial, 10, 10, f_spatial_M);
    climate_emulator_sum_double_core(f_spatial, f_spatial_M, f_spatial_N);
#endif
}

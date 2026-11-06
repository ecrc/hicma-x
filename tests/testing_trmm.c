#include "hicma_parsec.h"
#include "dplasma/tests/common.h"
#include <sys/time.h>

static int check_solution( parsec_context_t *parsec, int loud,
                           dplasma_enum_t side, dplasma_enum_t uplo, dplasma_enum_t trans, dplasma_enum_t diag,
                           double alpha,
                           int Am, int An, int Aseed,
                           int M,  int N,  int Cseed,
                           parsec_matrix_block_cyclic_t *dcCfinal,
                           double fixedacc, hicma_parsec_params_t *params );

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    hicma_parsec_params_t params;
    hicma_parsec_data_t data;
    int ret = 0;
    int Aseed = 3872;
    int Cseed = 2873;
    double alpha = 3.5;
    parsec_tiled_matrix_t *dcA;

#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 4.2;
#endif

    /* Parse command line arguments */
    parse_arguments(&argc, &argv, &params);

    /* Disable adaptive decision in memory */
    params.adaptive_memory = 0;

    /* Initialize HiCMA parameters */
    hicma_parsec_params_init(&params, argv);

    /* A is M x M and its decisions use the NT x NT layout of N */
    if( params.M != params.N ) {
        if( 0 == params.rank )
            fprintf(stderr, "testing_trmm: --M (%d) must equal --N (%d); B is N x K\n", params.M, params.N);
        exit(1);
    }

    /* Initialize PaRSEC */
    parsec = hicma_parsec_setup_parsec(argc, argv, &params);

    /* Print initial parameters */
    hicma_parsec_params_print_initial(&params);

    SYNC_TIME_PRINT(params.rank, ("HiCMA and PaRSEC initialization completed in %.6f seconds\n", sync_time_elapsed));

    int M = params.M;
    int N = params.N;
    int K = params.K;
    int MB = params.MB;
    int NB = params.NB;
    int KB = params.KB;
    int nodes = params.nodes;
    int rank = params.rank;
    int P = params.P;
    int Q = params.Q;
    int KP = params.KP;
    int KQ = params.KQ;
    int IP = 0;
    int JQ = 0;
    int check = params.check;
    int uplo = params.uplo;
    int nruns = params.nruns;
    int loud = params.verbose;
    int gpus = params.gpus;

    assert(MB == NB);
    assert(MB == KB);

    /* initializing matrix structure */
    int Am = max(M, N);
    int LDA = Am;
    int LDC = M;
    //int LDA = max(LDA, Am);
    //int LDC = max(LDC, M);
    PASTE_CODE_ALLOCATE_MATRIX(dcA0, 1,
        parsec_matrix_block_cyclic, (&dcA0, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                               rank, MB, NB, LDA, Am, 0, 0,
                               Am, Am, P, nodes/P, KP, KQ, IP, JQ));
    PASTE_CODE_ALLOCATE_MATRIX(dcC, 1,
        parsec_matrix_block_cyclic, (&dcC, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                               rank, MB, KB, LDC, K, 0, 0,
                               M, K, P, nodes/P, KP, KQ, IP, JQ));

    /* Allocate memory for GPU workspace */
#if (defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)) && GPU_BUFFER_ONCE
    gpu_temporay_buffer_init( &data, &params, MB, NB, 0, params.kind_of_cholesky );
#endif

    if(!check)
    {
        dplasma_enum_t side  = dplasmaLeft;
        dplasma_enum_t trans = dplasmaNoTrans;
        uplo = dplasmaLower;
        int sidx = 0; /* dplasmaLeft */
        int uidx = 1; /* dplasmaLower */
        int tidx = 0; /* dplasmaNoTrans */

        PASTE_CODE_FLOPS(FLOPS_DTRMM, (side, (DagDouble_t)M, (DagDouble_t)N));

        /* Make A square */
        if (side == dplasmaLeft) {
            dcA = parsec_tiled_matrix_submatrix( (parsec_tiled_matrix_t *)&dcA0, 0, 0, M, M );
        } else {
            dcA = parsec_tiled_matrix_submatrix( (parsec_tiled_matrix_t *)&dcA0, 0, 0, N, N );
        }

        /* matrix generation */
        if(loud > 2 && 0 == rank) printf("+++ Generate matrices ... ");
        dplasma_dplgsy( parsec, 0., uplo, dcA, Aseed);
        dplasma_dplrnt( parsec, 0,        (parsec_tiled_matrix_t *)&dcC, Cseed);
        if(loud > 2 && 0 == rank)  printf("Done\n");

        if( params.adaptive_decision ) {
            /* Get norm and decisions */
            SYNC_TIME_START();
            hicma_parsec_matrix_norm_get(parsec, params.uplo, dcA, &params, params.norm_tile, &params.norm_global, "double");
            SYNC_TIME_PRINT(params.rank, ("hicma_parsec_matrix_norm_get A: uplo= %d norm_global %lf\n", params.uplo, params.norm_global));

            SYNC_TIME_START();
            hicma_parsec_decision_make_comp(parsec, params.uplo, dcA, &params, params.norm_tile, params.norm_global, params.decisions);
            SYNC_TIME_PRINT(params.rank, ("hicma_parsec_decision_make_comp A: uplo= %d norm_global %lf\n", params.uplo, params.norm_global));

            SYNC_TIME_START();
            parsec_datatype_convert_dense_adaptive(parsec, &data, &params, params.uplo, dcA, params.decisions, 0);
            SYNC_TIME_PRINT(params.rank, ("parsec_datatype_convert_dense_adaptive A 0: uplo= %d norm_global %lf\n", params.uplo, params.norm_global));

            SYNC_TIME_START();
            hicma_parsec_matrix_norm_get(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcC,
                    &params, params.norm_tileB, &params.norm_globalB, "double");
            SYNC_TIME_PRINT(params.rank, ("hicma_parsec_matrix_norm_get B: uplo= %d norm_global %lf\n", dplasmaUpperLower, params.norm_globalB));

            SYNC_TIME_START();
            hicma_parsec_decision_make_comp(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcC,
                    &params, params.norm_tileB, params.norm_globalB, params.decisionsB);
            SYNC_TIME_PRINT(params.rank, ("hicma_parsec_decision_make_comp B: uplo= %d norm_global %lf\n", dplasmaUpperLower, params.norm_globalB));

            SYNC_TIME_START();
            parsec_datatype_convert_dense_adaptive(parsec, &data, &params, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcC, params.decisionsB, 0);
            SYNC_TIME_PRINT(params.rank, ("parsec_datatype_convert_dense_adaptive B 0: uplo= %d norm_global %lf\n", dplasmaUpperLower, params.norm_globalB));
        }

        if(params.verbose > 9) {
            print_decisions(&params, params.decisions, params.uplo, params.NT, params.NT);
            print_decisions(&params, params.decisionsB, dplasmaUpperLower, params.NT, params.KT);
        }

        int t, d;
        for(t = 0; t < nruns; t++) {
            //parsec_devices_release_memory();
            for(d = 0; d < 2; d++) {
                struct timeval tstart, tend;
                double run_time;

#if defined(PARSEC_HAVE_MPI)
                MPI_Barrier(MPI_COMM_WORLD);
#endif
                gettimeofday(&tstart, NULL);
                if (hicma_parsec_trmm(parsec, side, uplo, trans, diags[d],
                                      alpha, dcA, (parsec_tiled_matrix_t *)&dcC, &data, &params) != 0) {
                    ret |= 1;
                }
#if defined(PARSEC_HAVE_MPI)
                MPI_Barrier(MPI_COMM_WORLD);
#endif
                gettimeofday(&tend, NULL);

                run_time = (tend.tv_sec - tstart.tv_sec) +
                           (tend.tv_usec - tstart.tv_usec) / 1.0e6;
                gflops = (run_time > 0.0) ? (flops * 1e-9) / run_time : 0.0;

                if (rank == 0) {
                    printf("TRMM run %d/%d (%s, %s, %s, %s): %.6f s, %.3f Gflop/s "
                           "(nodes= %d gpus= %d P= %d Q= %d MB= %d NB= %d M= %d N= %d)\n",
                           t + 1, nruns, sidestr[sidx], uplostr[uidx], transstr[tidx], diagstr[d],
                           run_time, gflops, nodes, gpus, P, Q, MB, NB, M, N);
                }
            }

            //parsec_devices_reset_load(parsec);
        }
        free(dcA);
    }
    else
    {
        int d;
        int info_solution;

        PASTE_CODE_ALLOCATE_MATRIX(dcC2, 1,
            parsec_matrix_block_cyclic, (&dcC2, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                                   rank, MB, NB, LDC, K, 0, 0,
                                   M, K, P, nodes/P, KP, KQ, IP, JQ));

        dplasma_dplrnt( parsec, 0, (parsec_tiled_matrix_t *)&dcC2, Cseed);

        dplasma_enum_t side  = dplasmaLeft;
        dplasma_enum_t uplo  = dplasmaLower;
        dplasma_enum_t trans = dplasmaNoTrans;
        int sidx = 0; /* dplasmaLeft  */
        int uidx = 1; /* dplasmaLower */
        int tidx = 0; /* dplasmaNoTrans */

        /* Left-side TRMM uses an MxM triangular matrix A. */
        Am = M;
        dcA = parsec_tiled_matrix_submatrix( (parsec_tiled_matrix_t *)&dcA0, 0, 0, M, M );
        dplasma_dplgsy( parsec, 0., dplasmaUpperLower, dcA, Aseed);

        /* Get norm and decisions */
        if( params.adaptive_decision ) {
            SYNC_TIME_START();
            hicma_parsec_matrix_norm_get(parsec, params.uplo, dcA, &params, params.norm_tile, &params.norm_global, "double");
            SYNC_TIME_PRINT(params.rank, ("hicma_parsec_matrix_norm_get A: uplo= %d norm_global %lf\n", params.uplo, params.norm_global));

            SYNC_TIME_START();
            hicma_parsec_decision_make_comp(parsec, params.uplo, dcA, &params, params.norm_tile, params.norm_global, params.decisions);
            SYNC_TIME_PRINT(params.rank, ("hicma_parsec_decision_make_comp A: uplo= %d norm_global %lf\n", params.uplo, params.norm_global));

            SYNC_TIME_START();
            parsec_datatype_convert_dense_adaptive(parsec, &data, &params, params.uplo, dcA, params.decisions, 0);
            SYNC_TIME_PRINT(params.rank, ("parsec_datatype_convert_dense_adaptive A 0: uplo= %d norm_global %lf\n", params.uplo, params.norm_global));

            if(params.verbose > 9) print_decisions(&params, params.decisions, params.uplo, params.NT, params.NT);
        }

        for (d = 0; d < 2; d++) {
            if ( rank == 0 ) {
                printf("***************************************************\n");
                printf(" ----- TESTING DTRMM (%s, %s, %s, %s) -------- \n",
                        sidestr[sidx], uplostr[uidx], transstr[tidx], diagstr[d]);
            }

            /* matrix generation */
            if(loud > 2 && 0 == rank) printf("Generate matrices ... ");
            dplasma_dlacpy( parsec, dplasmaUpperLower,
                    (parsec_tiled_matrix_t *)&dcC2, (parsec_tiled_matrix_t *)&dcC );
            if(loud > 2 && 0 == rank)  printf("Done\n");

            if( params.adaptive_decision ) {
                /* Get norm and decisions */
                SYNC_TIME_START();
                hicma_parsec_matrix_norm_get(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcC,
                        &params, params.norm_tileB, &params.norm_globalB, "double");
                SYNC_TIME_PRINT(params.rank, ("hicma_parsec_matrix_norm_get B: uplo= %d norm_global %lf\n", dplasmaUpperLower, params.norm_globalB));

                SYNC_TIME_START();
                hicma_parsec_decision_make_comp(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcC,
                        &params, params.norm_tileB, params.norm_globalB, params.decisionsB);
                SYNC_TIME_PRINT(params.rank, ("hicma_parsec_decision_make_comp B: uplo= %d norm_global %lf\n", dplasmaUpperLower, params.norm_globalB));

                SYNC_TIME_START();
                parsec_datatype_convert_dense_adaptive(parsec, &data, &params, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcC, params.decisionsB, 0);
                SYNC_TIME_PRINT(params.rank, ("parsec_datatype_convert_dense_adaptive B 0: uplo= %d norm_global %lf\n", dplasmaUpperLower, params.norm_globalB));

                if(params.verbose > 9) print_decisions(&params, params.decisionsB, dplasmaUpperLower, params.NT, params.KT);
            }

            /* Compute */
            if(loud > 2 && 0 == rank) printf("Compute ... ... ");
            if (hicma_parsec_trmm(parsec, side, uplo, trans, diags[d],
                        alpha, dcA, (parsec_tiled_matrix_t *)&dcC, &data, &params) != 0) {
                if (rank == 0) {
                    printf("hicma_parsec_trmm execution failed\n");
                }
                ret |= 1;
            }
            if(loud > 2 && 0 == rank) printf("Done\n");

            /* Convert back to DP */
            if( params.adaptive_decision ) {
                SYNC_TIME_START();
                parsec_datatype_convert_dense_adaptive(parsec, &data, &params, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcC, params.decisionsB, 1);
                SYNC_TIME_PRINT(params.rank, ("parsec_datatype_convert_dense_adaptive B 1: uplo= %d norm_global %lf\n", dplasmaUpperLower, params.norm_globalB));
            }

            SYNC_TIME_START();
            hicma_parsec_matrix_norm_get(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcC,
                    &params, params.norm_tileB, &params.norm_globalB, "double");
            SYNC_TIME_PRINT(params.rank, ("hicma_parsec_matrix_norm_get B: uplo= %d norm_global %lf\n", dplasmaUpperLower, params.norm_globalB));

            SYNC_TIME_START();
            parsec_datatype_convert_dense_adaptive(parsec, &data, &params, params.uplo, dcA, params.decisions, 1);
            SYNC_TIME_PRINT(params.rank, ("parsec_datatype_convert_dense_adaptive A 1: uplo= %d norm_global %lf\n", params.uplo, params.norm_global));

            SYNC_TIME_START();
            hicma_parsec_matrix_norm_get(parsec, params.uplo, dcA, &params, params.norm_tile, &params.norm_global, "double");
            SYNC_TIME_PRINT(params.rank, ("hicma_parsec_matrix_norm_get A: uplo= %d norm_global %lf\n", params.uplo, params.norm_global));

            SYNC_TIME_START();
            parsec_datatype_convert_dense_adaptive(parsec, &data, &params, params.uplo, dcA, params.decisions, 0);
            SYNC_TIME_PRINT(params.rank, ("parsec_datatype_convert_dense_adaptive A 0: uplo= %d norm_global %lf\n", params.uplo, params.norm_global));

            /* Check the solution */
            info_solution = check_solution(parsec, rank == 0 ? loud : 0,
                    side, uplo, trans, diags[d],
                    alpha, Am, Am, Aseed,
                    M,  K,  Cseed,
                    &dcC, params.fixedacc, &params);
            if ( rank == 0 ) {
                if (info_solution == 0) {
                    printf(" ---- TESTING DTRMM (%s, %s, %s, %s) ...... PASSED !\n",
                           sidestr[sidx], uplostr[uidx], transstr[tidx], diagstr[d]);
                }
                else {
                    printf(" ---- TESTING DTRMM (%s, %s, %s, %s) ... FAILED !\n",
                           sidestr[sidx], uplostr[uidx], transstr[tidx], diagstr[d]);
                    ret |= 1;
                }
                printf("***************************************************\n");
            }
        }
        free(dcA);
        parsec_data_free(dcC2.mat);
        parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&dcC2);
    }

    /* Free parameter arrays */
    free(params.rank_array);
    free(params.op_band);
    free(params.op_offband);
    free(params.op_path);
    free(params.op_offpath);
    free(params.gather_time);
    free(params.gather_time_tmp);
    free(params.decisions);
    free(params.decisions_send);
    free(params.decisions_gemm_gpu);
    free(params.decisionsB);
    free(params.norm_tile);
    free(params.norm_tileB);

    /* GPU workspace */
#if (defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)) && GPU_BUFFER_ONCE
    gpu_temporay_buffer_fini( &data, params.kind_of_cholesky );
#endif

    parsec_data_free(dcA0.mat);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&dcA0);
    parsec_data_free(dcC.mat);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&dcC);

    /* Clean up PaRSEC */
    hicma_parsec_cleanup_parsec(parsec, &params);

    return ret;
}


/**********************************
 * static functions
 **********************************/

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_solution( parsec_context_t *parsec, int loud,
                           dplasma_enum_t side, dplasma_enum_t uplo, dplasma_enum_t trans, dplasma_enum_t diag,
                           double alpha,
                           int Am, int An, int Aseed,
                           int M,  int N,  int Cseed,
                           parsec_matrix_block_cyclic_t *dcCfinal,
                           double fixedacc, hicma_parsec_params_t *params )
{
    int info_solution = 1;
    double Anorm, Cinitnorm, Cdplasmanorm, Clapacknorm, Rnorm, Rfro;
    double eps, result;
    int MB = dcCfinal->super.mb;
    int NB = dcCfinal->super.nb;
    int LDA = Am;
    int LDC = M;
    int rank  = dcCfinal->super.super.myrank;
    int nodes = dcCfinal->super.super.nodes;
    int P = dcCfinal->grid.rows;
    int Q = dcCfinal->grid.cols;
    int KP = dcCfinal->grid.krows;
    int KQ = dcCfinal->grid.kcols;
    int IP = dcCfinal->grid.ip;
    int JQ = dcCfinal->grid.jq;

    eps = LAPACKE_dlamch_work('e');

    /* Tile storage: the reference is distributed on the same P x Q grid as
     * the result (LAPACK storage with the global leading dimension is only
     * valid on a 1 x 1 grid). */
    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
        parsec_matrix_block_cyclic, (&dcA, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                               rank, MB, NB, LDA, An, 0, 0,
                               Am, An, P, nodes/P, KP, KQ, IP, JQ));
    PASTE_CODE_ALLOCATE_MATRIX(dcC, 1,
        parsec_matrix_block_cyclic, (&dcC, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                               rank, MB, NB, LDC, N, 0, 0,
                               M, N, P, nodes/P, KP, KQ, IP, JQ));

    dplasma_dplgsy( parsec, 0., dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcA, Aseed);
    dplasma_dplrnt( parsec, 0, (parsec_tiled_matrix_t *)&dcC, Cseed );

    Anorm        = dplasma_dlange( parsec, dplasmaInfNorm, (parsec_tiled_matrix_t*)&dcA );
    Cinitnorm    = dplasma_dlange( parsec, dplasmaInfNorm, (parsec_tiled_matrix_t*)&dcC );
    Cdplasmanorm = dplasma_dlange( parsec, dplasmaInfNorm, (parsec_tiled_matrix_t*)dcCfinal );

#if 0
    if ( rank == 0 ) {
        cblas_dtrmm(CblasColMajor,
                    (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
                    (CBLAS_TRANSPOSE)trans, (CBLAS_DIAG)diag,
                    M, N,
                    (alpha), dcA.mat, LDA,
                                        dcC.mat, LDC );
    }
#else
    parsec_taskpool_t *parsec_dtrmm = NULL;
    parsec_dtrmm = dplasma_dtrmm_New(side, uplo, trans, diag, alpha, (parsec_tiled_matrix_t*)&dcA, (parsec_tiled_matrix_t*)&dcC);
    if ( parsec_dtrmm != NULL )
    {
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        hicma_parsec_disable_GPU(parsec_dtrmm);
#endif
        parsec_context_add_taskpool( parsec, (parsec_taskpool_t*)parsec_dtrmm);
        dplasma_wait_until_completion(parsec);
        dplasma_dtrmm_Destruct( parsec_dtrmm );
    }
#endif

    if(loud > 99) {
        dplasma_dprint(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t*)dcCfinal);
        printf("\n\n\n");
        dplasma_dprint(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t*)&dcC);
    }

    SYNC_TIME_START();
    hicma_parsec_matrix_norm_get(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcC,
            params, params->norm_tileB, &params->norm_globalB, "double");
    SYNC_TIME_PRINT(params->rank, ("hicma_parsec_matrix_norm_get in check DPLASMA/LAPACK: uplo= %d norm_global %lf\n", dplasmaUpperLower, params->norm_globalB));

    Clapacknorm = dplasma_dlange( parsec, dplasmaInfNorm, (parsec_tiled_matrix_t*)&dcC );

    dplasma_dgeadd( parsec, dplasmaNoTrans,
            -1.0, (parsec_tiled_matrix_t*)dcCfinal,
            1.0, (parsec_tiled_matrix_t*)&dcC );

    Rnorm = dplasma_dlange( parsec, dplasmaMaxNorm, (parsec_tiled_matrix_t*)&dcC );

    /* dlange skips NaN entries; a sum of squares does not */
    Rfro = hicma_parsec_matrix_norm_get(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcC,
            params, params->norm_tileB, &params->norm_globalB, "double");

    result = Rnorm / (Clapacknorm * max(M,N) * fmax(fixedacc, eps));

    if ( rank == 0 ) {
        if ( loud > 2 ) {
            printf("  ||A||_inf = %e, ||C||_inf = %e\n"
                   "  ||lapack(a*A*C)||_inf = %e, ||HiCMA-x(a*A*C)||_inf = %e, ||R||_m = %e, ||R||_F = %e, res = %e\n",
                   Anorm, Cinitnorm, Clapacknorm, Cdplasmanorm, Rnorm, Rfro, result);
        }

        if (  isinf(Clapacknorm) || isinf(Cdplasmanorm) ||
              isnan(Rfro) || isinf(Rfro) ||
              isnan(result) || isinf(result) || (result > 10.0) ) {
            info_solution = 1;
        }
        else {
            info_solution = 0;
        }
    }

#if defined(PARSEC_HAVE_MPI)
    MPI_Bcast(&info_solution, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&dcA);
    parsec_data_free(dcC.mat);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&dcC);

    return info_solution;
}

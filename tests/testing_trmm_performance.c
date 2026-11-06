#include "hicma_parsec.h"
#include "dplasma/tests/common.h"
#include <sys/time.h>

int main(int argc, char **argv)
{
    parsec_context_t *parsec;
    hicma_parsec_params_t params;
    hicma_parsec_data_t data;
    const dplasma_enum_t side  = dplasmaLeft;
    const dplasma_enum_t uplo  = dplasmaLower;
    const dplasma_enum_t trans = dplasmaNoTrans;
    const dplasma_enum_t diag  = dplasmaNonUnit;
    const int Aseed = 3872;
    const int Cseed = 2873;
    double alpha = 3.5;
    int ret = 0;

#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 4.2;
#endif

    parse_arguments(&argc, &argv, &params);

    /* This benchmark measures the explicit mixed-precision decisions only. */
    params.adaptive_memory = 0;

    /* The TRMM has no FP8 tiles: keep FP8 out of the precision decisions
     * (hicma_parsec_trmm_New rejects them) */
    params.fp8 = 0;

    hicma_parsec_params_init(&params, argv);

    /* A is M x M and its decisions use the NT x NT layout of N */
    if (params.M != params.N) {
        if (0 == params.rank)
            fprintf(stderr, "testing_trmm_performance: --M (%d) must equal --N (%d); C is N x K\n",
                    params.M, params.N);
        exit(1);
    }

    parsec = hicma_parsec_setup_parsec(argc, argv, &params);
    hicma_parsec_params_print_initial(&params);

    SYNC_TIME_PRINT(params.rank,
                    ("HiCMA and PaRSEC initialization completed in %.6f seconds\n",
                     sync_time_elapsed));

    const int M = params.M;
    const int K = params.K;
    const int MB = params.MB;
    const int NB = params.NB;
    const int KB = params.KB;
    const int nodes = params.nodes;
    const int rank = params.rank;
    const int P = params.P;
    const int Q = params.Q;
    const int KP = params.KP;
    const int KQ = params.KQ;
    const int IP = 0;
    const int JQ = 0;
    const int nruns = params.nruns;
    const int loud = params.verbose;
    const int gpus = params.gpus;

    assert(MB == NB);
    assert(MB == KB);

    /* Left TRMM uses a square lower-triangular A and an M-by-K C. */
    parsec_matrix_sym_block_cyclic_t dcA;
    parsec_matrix_sym_block_cyclic_init(&dcA, PARSEC_MATRIX_DOUBLE,
            rank, MB, NB, M, M, 0, 0,
            M, M, P, nodes / P, uplo);
    dcA.mat = parsec_data_allocate(
            (size_t)dcA.super.nb_local_tiles *
            (size_t)dcA.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t *)&dcA, "dcA");

    parsec_matrix_block_cyclic_t dcC;
    parsec_matrix_block_cyclic_init(&dcC, PARSEC_MATRIX_DOUBLE,
            PARSEC_MATRIX_TILE, rank, MB, KB, M, K, 0, 0,
            M, K, P, nodes / P, KP, KQ, IP, JQ);
    dcC.mat = parsec_data_allocate(
            (size_t)dcC.super.nb_local_tiles *
            (size_t)dcC.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(dcC.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t *)&dcC, "dcC");

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Pin A and C once. Otherwise PaRSEC registers and unregisters both
     * matrices with every taskpool, inside every timed TRMM. */
    const size_t A_size = (size_t)dcA.super.nb_local_tiles * (size_t)dcA.super.bsiz *
                          (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype);
    const size_t C_size = (size_t)dcC.super.nb_local_tiles * (size_t)dcC.super.bsiz *
                          (size_t)parsec_datadist_getsizeoftype(dcC.super.mtype);
    const int host_registered = (gpus > 0);
    if (host_registered) {
        dcA.super.super.register_memory = NULL;
        dcA.super.super.unregister_memory = NULL;
        dcC.super.super.register_memory = NULL;
        dcC.super.super.unregister_memory = NULL;
        if (cudaSuccess != cudaHostRegister(dcA.mat, A_size, cudaHostRegisterDefault) ||
            cudaSuccess != cudaHostRegister(dcC.mat, C_size, cudaHostRegisterDefault)) {
            fprintf(stderr, "Unable to register the host memory of A and C\n");
            abort();
        }
    }
#endif

#if (defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)) && GPU_BUFFER_ONCE
    gpu_temporay_buffer_init(&data, &params, MB, NB, 0, params.kind_of_cholesky);
#endif

    const double flops = FLOPS_DTRMM(side, (DagDouble_t)M,
                                     (DagDouble_t)K);

    if (loud > 2 && rank == 0) {
        printf("+++ Generate matrices ... ");
    }
    dplasma_dplgsy(parsec, 0., uplo,
                   (parsec_tiled_matrix_t *)&dcA, Aseed);
    dplasma_dplrnt(parsec, 0,
                   (parsec_tiled_matrix_t *)&dcC, Cseed);
    if (loud > 2 && rank == 0) {
        printf("Done\n");
    }

    if (params.adaptive_decision) {
        SYNC_TIME_START();
        hicma_parsec_matrix_norm_get(parsec, uplo,
                (parsec_tiled_matrix_t *)&dcA, &params,
                params.norm_tile, &params.norm_global, "double");
        SYNC_TIME_PRINT(rank,
                ("hicma_parsec_matrix_norm_get A: uplo= %d norm_global %lf\n",
                 uplo, params.norm_global));

        SYNC_TIME_START();
        hicma_parsec_decision_make_comp(parsec, uplo,
                (parsec_tiled_matrix_t *)&dcA, &params,
                params.norm_tile, params.norm_global, params.decisions);
        SYNC_TIME_PRINT(rank,
                ("hicma_parsec_decision_make_comp A: uplo= %d norm_global %lf\n",
                 uplo, params.norm_global));

        SYNC_TIME_START();
        parsec_datatype_convert_dense_adaptive(parsec, &data, &params, uplo,
                (parsec_tiled_matrix_t *)&dcA, params.decisions, 0);
        SYNC_TIME_PRINT(rank,
                ("parsec_datatype_convert_dense_adaptive A: uplo= %d norm_global %lf\n",
                 uplo, params.norm_global));

        SYNC_TIME_START();
        hicma_parsec_matrix_norm_get(parsec, dplasmaUpperLower,
                (parsec_tiled_matrix_t *)&dcC, &params,
                params.norm_tileB, &params.norm_globalB, "double");
        SYNC_TIME_PRINT(rank,
                ("hicma_parsec_matrix_norm_get C: uplo= %d norm_global %lf\n",
                 dplasmaUpperLower, params.norm_globalB));

        SYNC_TIME_START();
        hicma_parsec_decision_make_comp(parsec, dplasmaUpperLower,
                (parsec_tiled_matrix_t *)&dcC, &params,
                params.norm_tileB, params.norm_globalB, params.decisionsB);
        SYNC_TIME_PRINT(rank,
                ("hicma_parsec_decision_make_comp C: uplo= %d norm_global %lf\n",
                 dplasmaUpperLower, params.norm_globalB));

        SYNC_TIME_START();
        parsec_datatype_convert_dense_adaptive(parsec, &data, &params,
                dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcC,
                params.decisionsB, 0);
        SYNC_TIME_PRINT(rank,
                ("parsec_datatype_convert_dense_adaptive C: uplo= %d norm_global %lf\n",
                 dplasmaUpperLower, params.norm_globalB));
    }

    if (loud > 9) {
        print_decisions(&params, params.decisions, uplo,
                        params.MT, params.MT);
        print_decisions(&params, params.decisionsB, dplasmaUpperLower,
                        params.MT, params.KT);
    }

    for (int run = 0; run < nruns; run++) {
        struct timeval tstart, tend;

        /* Every run starts from the same C, regenerated outside the timing:
         * back to FP64, generated, and converted again */
        if (run > 0) {
            if (params.adaptive_decision) {
                parsec_datatype_convert_dense_adaptive(parsec, &data, &params,
                        dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcC,
                        params.decisionsB, 1);
            }
            dplasma_dplrnt(parsec, 0, (parsec_tiled_matrix_t *)&dcC, Cseed);
            if (params.adaptive_decision) {
                parsec_datatype_convert_dense_adaptive(parsec, &data, &params,
                        dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcC,
                        params.decisionsB, 0);
            }
        }

#if defined(PARSEC_HAVE_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        gettimeofday(&tstart, NULL);
        if (hicma_parsec_trmm(parsec, side, uplo, trans, diag,
                              alpha, (parsec_tiled_matrix_t *)&dcA,
                              (parsec_tiled_matrix_t *)&dcC,
                              &data, &params) != 0) {
            ret = 1;
        }
#if defined(PARSEC_HAVE_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        gettimeofday(&tend, NULL);

        const double run_time = (tend.tv_sec - tstart.tv_sec) +
                                (tend.tv_usec - tstart.tv_usec) / 1.0e6;
        const double tflops = run_time > 0.0 ? flops * 1.0e-12 / run_time : 0.0;

        if (rank == 0) {
            printf("TRMM run %d/%d (Left, Lower, NoTrans, NonUnit): "
                   "%.6f s, %.3f Tflop/s "
                   "(nodes= %d gpus= %d P= %d Q= %d MB= %d NB= %d M= %d K= %d)\n",
                   run + 1, nruns, run_time, tflops,
                   nodes, gpus, P, Q, MB, NB, M, K);
        }
    }

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

#if (defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)) && GPU_BUFFER_ONCE
    gpu_temporay_buffer_fini(&data, params.kind_of_cholesky);
#endif

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    if (host_registered) {
        cudaHostUnregister(dcA.mat);
        cudaHostUnregister(dcC.mat);
    }
#endif
    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&dcA);
    parsec_data_free(dcC.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&dcC);

    hicma_parsec_cleanup_parsec(parsec, &params);
    return ret;
}

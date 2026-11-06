#include "hicma_parsec.h"
#include "dplasma/tests/common.h"
#include "dplasmaaux.h"

#include <errno.h>
#include <sys/time.h>

static int parse_percentage(const char *text, double *value)
{
    char *end = NULL;
    double parsed;

    errno = 0;
    parsed = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' ||
        !isfinite(parsed) || parsed < 0.0 || parsed > 100.0) {
        return -1;
    }
    *value = parsed;
    return 0;
}

static int parse_binary_flag(const char *text, int *value)
{
    char *end = NULL;
    long parsed;

    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        (parsed != 0 && parsed != 1)) {
        return -1;
    }
    *value = (int)parsed;
    return 0;
}

static int parse_portion_arguments(int *argc, char ***argv,
                                   double *portion_dp,
                                   double *portion_sp,
                                   double *portion_hp,
                                   int *pinned_memory)
{
    char **args = *argv;
    int write_index = 1;

    *portion_dp = 20.0;
    *portion_sp = 30.0;
    *portion_hp = 50.0;
    *pinned_memory = 0;

    for (int read_index = 1; read_index < *argc; read_index++) {
        char *argument = args[read_index];
        const char *value_text = NULL;
        double *target = NULL;

        if (strcmp(argument, "--help") == 0) {
            printf("Static TRMM precision-map options:\n"
                   "  --portion_dp PERCENT   Random off-diagonal DP portion "
                   "(default: 20)\n"
                   "  --portion_sp PERCENT   Random off-diagonal SP portion "
                   "(default: 30)\n"
                   "  --portion_hp PERCENT   Random off-diagonal HP portion "
                   "(default: 50)\n"
                   "  --pinned_memory 0|1    Use registered/pinned host memory "
                   "(default: 0)\n"
                   "  --adaptive_decision 0  Use the band-size decision map\n"
                   "  --adaptive_decision 1  Use the randomized decision map\n\n");
            args[write_index++] = argument;
            continue;
        } else if (strcmp(argument, "--pinned_memory") == 0 ||
                   strncmp(argument, "--pinned_memory=", 16) == 0) {
            if (argument[15] == '=') {
                value_text = argument + 16;
            } else {
                if (++read_index >= *argc) {
                    fprintf(stderr, "%s requires 0 or 1\n", argument);
                    return -1;
                }
                value_text = args[read_index];
            }
            if (parse_binary_flag(value_text, pinned_memory) != 0) {
                fprintf(stderr, "Invalid value for %s: %s (expected 0 or 1)\n",
                        argument, value_text);
                return -1;
            }
            continue;
        } else if (strcmp(argument, "--portion_dp") == 0) {
            target = portion_dp;
        } else if (strcmp(argument, "--portion_sp") == 0) {
            target = portion_sp;
        } else if (strcmp(argument, "--portion_hp") == 0) {
            target = portion_hp;
        } else if (strncmp(argument, "--portion_dp=", 13) == 0) {
            target = portion_dp;
            value_text = argument + 13;
        } else if (strncmp(argument, "--portion_sp=", 13) == 0) {
            target = portion_sp;
            value_text = argument + 13;
        } else if (strncmp(argument, "--portion_hp=", 13) == 0) {
            target = portion_hp;
            value_text = argument + 13;
        } else {
            args[write_index++] = argument;
            continue;
        }

        if (value_text == NULL) {
            if (++read_index >= *argc) {
                fprintf(stderr, "%s requires a percentage value\n", argument);
                return -1;
            }
            value_text = args[read_index];
        }
        if (parse_percentage(value_text, target) != 0) {
            fprintf(stderr, "Invalid percentage for %s: %s\n",
                    argument, value_text);
            return -1;
        }
    }

    args[write_index] = NULL;
    *argc = write_index;

    if (fabs(*portion_dp + *portion_sp + *portion_hp - 100.0) > 1.0e-9) {
        fprintf(stderr,
                "--portion_dp + --portion_sp + --portion_hp must equal 100%% "
                "(received %.12g%%)\n",
                *portion_dp + *portion_sp + *portion_hp);
        return -1;
    }
    return 0;
}

static uint64_t decision_rng_next(uint64_t *state)
{
    uint64_t value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    return value * UINT64_C(2685821657736338717);
}

static int initialize_random_A_decisions(hicma_parsec_params_t *params,
                                         double portion_dp,
                                         double portion_sp,
                                         double portion_hp,
                                         uint64_t seed)
{
    const int nt = params->NT;
    const size_t off_diagonal_tiles = (size_t)nt * (size_t)(nt - 1) / 2;
    size_t count_dp = (size_t)llround(off_diagonal_tiles * portion_dp / 100.0);
    size_t count_sp = (size_t)llround(off_diagonal_tiles * portion_sp / 100.0);
    uint16_t *choices;
    size_t position = 0;

    (void)portion_hp;

    if (count_dp > off_diagonal_tiles) {
        count_dp = off_diagonal_tiles;
    }
    if (count_sp > off_diagonal_tiles - count_dp) {
        count_sp = off_diagonal_tiles - count_dp;
    }

    choices = (uint16_t *)malloc(off_diagonal_tiles * sizeof(uint16_t));
    if (off_diagonal_tiles != 0 && choices == NULL) {
        return -1;
    }

    for (; position < count_dp; position++) {
        choices[position] = DENSE_DP;
    }
    for (; position < count_dp + count_sp; position++) {
        choices[position] = DENSE_SP;
    }
    for (; position < off_diagonal_tiles; position++) {
        choices[position] = DENSE_HP;
    }

    for (size_t i = off_diagonal_tiles; i > 1; i--) {
        const size_t j = (size_t)(decision_rng_next(&seed) % i);
        const uint16_t temporary = choices[i - 1];
        choices[i - 1] = choices[j];
        choices[j] = temporary;
    }

    position = 0;
    for (int m = 0; m < nt; m++) {
        params->decisions[m * nt + m] = DENSE_DP;
        for (int n = 0; n < m; n++) {
            params->decisions[n * nt + m] = choices[position++];
        }
    }

    free(choices);
    return 0;
}

static int initialize_band_A_decisions(hicma_parsec_params_t *params)
{
    const int nt = params->NT;
    const int band_dp = params->band_size_dense_dp;
    const int band_sp = params->band_size_dense_sp;
    const int band_hp = params->band_size_dense_hp;
    const int band_dense = params->band_size_dense;

    if (band_dp < 0 || band_sp < band_dp || band_hp < band_sp ||
        band_dense < band_hp) {
        if (params->rank == 0) {
            fprintf(stderr,
                    "Band sizes must satisfy 0 <= DP <= SP <= HP <= dense; "
                    "received %d, %d, %d, %d\n",
                    band_dp, band_sp, band_hp, band_dense);
        }
        return -1;
    }

    for (int m = 0; m < nt; m++) {
        for (int n = 0; n <= m; n++) {
            const int distance = m - n;
            uint16_t decision;

            if (distance < band_dp) {
                decision = DENSE_DP;
            } else if (distance < band_sp) {
                decision = DENSE_SP;
            } else if (distance < band_hp) {
                decision = DENSE_HP;
            } else if (distance < band_dense) {
                /* TRMM has no FP8 path, so the remaining dense band is HP. */
                decision = DENSE_HP;
            } else {
                decision = DENSE_HP;
            }
            params->decisions[n * nt + m] = decision;
        }
    }
    return 0;
}

static int initialize_decision_maps(hicma_parsec_params_t *params,
                                    double portion_dp,
                                    double portion_sp,
                                    double portion_hp)
{
    int rc;

    if (params->adaptive_decision == 0) {
        rc = initialize_band_A_decisions(params);
    } else {
        rc = initialize_random_A_decisions(params,
                                           portion_dp,
                                           portion_sp,
                                           portion_hp,
                                           UINT64_C(0x9e3779b97f4a7c15));
    }
    if (rc != 0) {
        return rc;
    }

    for (int n = 0; n < params->KT; n++) {
        for (int m = 0; m < params->NT; m++) {
            params->decisionsB[n * params->NT + m] = DENSE_HP;
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    parsec_context_t *parsec;
    hicma_parsec_params_t params = {0};
    hicma_parsec_data_t data = {0};
    const dplasma_enum_t side = dplasmaLeft;
    const dplasma_enum_t uplo = dplasmaLower;
    const dplasma_enum_t trans = dplasmaNoTrans;
    const dplasma_enum_t diag = dplasmaNonUnit;
    const unsigned long long Aseed = 3872;
    const unsigned long long Bseed = 2873;
    const double alpha = 3.5;
    double portion_dp, portion_sp, portion_hp;
    int pinned_memory;
    int ret = 0;

    if (parse_portion_arguments(&argc, &argv,
                                &portion_dp, &portion_sp, &portion_hp,
                                &pinned_memory) != 0) {
        return 1;
    }

    parse_arguments(&argc, &argv, &params);
    params.adaptive_memory = 1;
    hicma_parsec_params_init(&params, argv);

    if (params.MT != params.NT) {
        if (params.rank == 0) {
            fprintf(stderr,
                    "Static TRMM requires M == N so the A decision-map stride "
                    "matches the square A descriptor\n");
        }
        ret = 1;
        goto free_parameters;
    }
    if (initialize_decision_maps(&params,
                                 portion_dp, portion_sp, portion_hp) != 0) {
        ret = 1;
        goto free_parameters;
    }

    parsec = hicma_parsec_setup_parsec(argc, argv, &params);
    hicma_parsec_params_print_initial(&params);

#if !HAVE_HP_CPU
    if (params.nruns > 0 && params.gpus == 0) {
        if (params.rank == 0) {
            fprintf(stderr,
                    "Static TRMM stores B in HP and therefore requires an "
                    "enabled GPU when HAVE_HP_CPU=0. Use --nruns 0 for a "
                    "CPU-only allocation/free test.\n");
        }
        hicma_parsec_cleanup_parsec(parsec, &params);
        ret = 1;
        goto free_parameters;
    }
#endif

    const int M = params.M;
    const int K = params.K;
    const int MB = params.MB;
    const int NB = params.NB;
    const int KB = params.KB;
    const int rank = params.rank;
    const int nodes = params.nodes;
    const int P = params.P;
    size_t A_tiles_dp = 0;
    size_t A_tiles_sp = 0;
    size_t A_tiles_hp = 0;

    for (int m = 0; m < params.NT; m++) {
        for (int n = 0; n <= m; n++) {
            switch (params.decisions[n * params.NT + m]) {
            case DENSE_DP: A_tiles_dp++; break;
            case DENSE_SP: A_tiles_sp++; break;
            case DENSE_HP: A_tiles_hp++; break;
            default: break;
            }
        }
    }
    const size_t B_tiles_hp = (size_t)params.NT * (size_t)params.KT;
    const size_t tile_elements = (size_t)MB * (size_t)NB;
    const size_t A_capacity_bytes = tile_elements *
        (A_tiles_dp * sizeof(double) +
         (A_tiles_sp + A_tiles_hp) * sizeof(float));
    const size_t A_payload_bytes = tile_elements *
        (A_tiles_dp * sizeof(double) + A_tiles_sp * sizeof(float) +
         A_tiles_hp * sizeof(uint16_t));

    assert(MB == NB);
    assert(MB == KB);

    parsec_matrix_sym_block_cyclic_t dcA;
    parsec_matrix_sym_block_cyclic_init(&dcA, PARSEC_MATRIX_DOUBLE,
            rank, MB, NB, M, M, 0, 0,
            M, M, P, nodes / P, uplo);
    dcA.mat = NULL;
    parsec_data_collection_set_key((parsec_data_collection_t *)&dcA, "dcA_static");

    parsec_matrix_block_cyclic_t dcB;
    parsec_matrix_block_cyclic_init(&dcB, PARSEC_MATRIX_BYTE,
            PARSEC_MATRIX_TILE, rank, MB, KB, M, K, 0, 0,
            M, K, P, nodes / P,
            params.KP, params.KQ, 0, 0);
    /* PARSEC_MATRIX_BYTE plus a two-byte tile stride represents binary16;
     * with --accumulation_fp32, B's FP16 tiles are stored in FP32. */
    dcB.super.bsiz *= params.accumulation_fp32 ? (int)sizeof(float)
                                               : (int)sizeof(uint16_t);
    dcB.mat = NULL;
    parsec_data_collection_set_key((parsec_data_collection_t *)&dcB, "dcB_static");

#if (defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)) && GPU_BUFFER_ONCE
    gpu_temporay_buffer_init(&data, &params, MB, NB, 0, params.kind_of_cholesky);
#endif

    SYNC_TIME_START();
    const size_t B_allocate_size =
        (size_t)dcB.super.nb_local_tiles * (size_t)dcB.super.bsiz;
    dcB.mat = parsec_data_allocate(B_allocate_size);
    if (dcB.mat == NULL) {
        fprintf(stderr, "Contiguous allocation for B failed (%zu bytes)\n",
                B_allocate_size);
        abort();
    }

    int B_memory_registered = 0;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    if (pinned_memory && params.gpus > 0) {
        dcB.super.super.register_memory = NULL;
        dcB.super.super.unregister_memory = NULL;
        if (cudaSuccess != cudaHostRegister(
                    dcB.mat, B_allocate_size, cudaHostRegisterDefault)) {
            fprintf(stderr, "Unable to register the contiguous B allocation\n");
            abort();
        }
        B_memory_registered = 1;
    }
#endif

    hicma_parsec_memory_allocation_dense_decision(
            parsec, dplasmaLower, (parsec_tiled_matrix_t *)&dcA,
            params.decisions, 1, 1, 0, pinned_memory, Aseed);
    hicma_parsec_memory_allocation_dense_decision(
            parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcB,
            params.decisionsB, 0, 0, params.accumulation_fp32, 0, Bseed);

#if defined(DPLASMA_HAVE_CUDA) || defined(DPLASMA_HAVE_HIP)
    /* Keep each local tile on its preferred GPU across the TRMM taskpool.
     * This is placement advice only; the JDF control flows above bound the
     * number of host-side broadcasts that may be active concurrently. */
    if (params.gpus > 0) {
/*
        dplasma_advise_data_on_device(
                parsec, dplasmaLower, (parsec_tiled_matrix_t *)&dcA,
                (parsec_tiled_matrix_unary_op_t)
                    dplasma_advise_data_on_device_ops_2D,
                NULL);
*/
        dplasma_advise_data_on_device(
                parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcB,
                (parsec_tiled_matrix_unary_op_t)
                    dplasma_advise_data_on_device_ops_2D,
                NULL);
    }
#endif
    SYNC_TIME_PRINT(rank,
            ("Static mixed-precision allocation and initialization "
             "host_memory= %s A_capacity_bytes= %zu A_payload_bytes= %zu "
             "B_bytes= %zu\n",
             pinned_memory ? "pinned" : "pageable",
             A_capacity_bytes, A_payload_bytes, B_allocate_size));
    const double initialization_time = sync_time_elapsed;

    if (rank == 0 && params.verbose > 0) {
        printf("A decision mode: %s",
               params.adaptive_decision == 0 ? "band" : "random off-diagonal");
        if (params.adaptive_decision != 0) {
            printf(" (DP %.2f%%, SP %.2f%%, HP %.2f%%)",
                   portion_dp, portion_sp, portion_hp);
        }
        printf("; B precision: HP%s; host memory: %s\n",
               params.accumulation_fp32 ? " (stored in FP32)" : "",
               pinned_memory ? "pinned/registered" : "pageable");
    }
    if (params.verbose > 9) {
        print_decisions(&params, params.decisions, uplo,
                        params.NT, params.NT);
        print_decisions(&params, params.decisionsB, dplasmaUpperLower,
                        params.NT, params.KT);
    }

    const double flops = FLOPS_DTRMM(side, (DagDouble_t)M,
                                     (DagDouble_t)K);
    for (int run = 0; run < params.nruns; run++) {
        struct timeval start, end;

        /* Every run starts from the same B, initialized again outside the
         * timing (in place: the tiles are already allocated) */
        if (run > 0) {
            hicma_parsec_memory_allocation_dense_decision(
                    parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcB,
                    params.decisionsB, 0, 0, params.accumulation_fp32, 0, Bseed);
        }

#if defined(PARSEC_HAVE_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        gettimeofday(&start, NULL);
        if (hicma_parsec_trmm(parsec, side, uplo, trans, diag,
                              alpha, (parsec_tiled_matrix_t *)&dcA,
                              (parsec_tiled_matrix_t *)&dcB,
                              &data, &params) != 0) {
            ret = 1;
        }
#if defined(PARSEC_HAVE_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        gettimeofday(&end, NULL);

        const double elapsed = (end.tv_sec - start.tv_sec) +
                               (end.tv_usec - start.tv_usec) / 1.0e6;
        const double tflops = elapsed > 0.0 ? flops * 1.0e-12 / elapsed : 0.0;
        if (rank == 0) {
            printf("TRMM_STATIC_RESULT run= %d nruns= %d time_s= %.9f "
                   "tflops= %.6f initialization_s= %.9f flops= %.0f "
                   "nodes= %d cores= %d gpus= %d gpu_type= %d P= %d Q= %d "
                   "M= %d N= %d K= %d MB= %d NB= %d KB= %d NT= %d KT= %d "
                   "side= left uplo= lower trans= notrans diag= nonunit "
                   "alpha= %.17g decision_mode= %s adaptive_decision= %d "
                   "portion_dp= %.6f portion_sp= %.6f portion_hp= %.6f "
                   "band_dp= %d band_sp= %d band_hp= %d band_dense= %d "
                   "A_tiles_dp= %zu A_tiles_sp= %zu A_tiles_hp= %zu "
                   "B_tiles_hp= %zu A_capacity_bytes= %zu "
                   "A_payload_bytes= %zu B_bytes= %zu "
                   "A_hp_capacity= sp B_precision= hp host_memory= %s "
                   "trmm_window= %d\n",
                   run + 1, params.nruns, elapsed, tflops,
                   initialization_time, flops,
                   params.nodes, params.cores, params.gpus, params.gpu_type,
                   params.P, params.Q, M, params.N, K, MB, NB, KB,
                   params.NT, params.KT,
                   alpha,
                   params.adaptive_decision == 0 ? "band" : "random",
                   params.adaptive_decision,
                   portion_dp, portion_sp, portion_hp,
                   params.band_size_dense_dp, params.band_size_dense_sp,
                   params.band_size_dense_hp, params.band_size_dense,
                   A_tiles_dp, A_tiles_sp, A_tiles_hp, B_tiles_hp,
                   A_capacity_bytes, A_payload_bytes, B_allocate_size,
                   pinned_memory ? "pinned" : "pageable",
                   params.trmm_window);
            fflush(stdout);
        }
    }

    hicma_parsec_memory_free_dense_decision(
            parsec, dplasmaLower,
            (parsec_tiled_matrix_t *)&dcA, &params, pinned_memory);

#if (defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)) && GPU_BUFFER_ONCE
    gpu_temporay_buffer_fini(&data, params.kind_of_cholesky);
#endif

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    if (B_memory_registered) {
        cudaHostUnregister(dcB.mat);
    }
#endif
    parsec_data_free(dcB.mat);
    dcB.mat = NULL;
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&dcB);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&dcA);
    hicma_parsec_cleanup_parsec(parsec, &params);

free_parameters:
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
    return ret;
}

/**
 * @file testing_matrix_sum.c
 * @brief Test of hicma_parsec_matrix_sum over tiles stored in different precisions
 *
 * The entries are small integers, so FP32 and FP64 sums are exact and do not
 * depend on the summation order: every result is compared exactly with the
 * value computed here, for every process grid of the run, for a double and a
 * float matrix without a decision map, for a mixed map (FP64, FP32, half and
 * FP8 kept in FP32, low-rank FP64/FP32 with their ranks) and for a NaN.
 */
#include "hicma_parsec.h"

static int f(int i, int j) { return (i * 7 + j * 13) % 101 - 50; }

/* storage decision of tile (m, n) in the mixed case */
static uint16_t decision(int m, int n)
{
    static const uint16_t d[6] = { DENSE_DP, DENSE_SP, DENSE_HP, DENSE_FP8, LOW_RANK_DP, LOW_RANK_SP };
    return (m == n) ? DENSE_DP : d[(m + 2 * n) % 6];
}

static void *tile(parsec_tiled_matrix_t *A, int m, int n)
{
    parsec_data_t *d = A->super.data_of(&A->super, m, n);
    return parsec_data_copy_get_ptr(parsec_data_get_copy(d, 0));
}

/* Fill the local tiles and return the expected sum over all tiles (each rank
 * computes it for the whole matrix). single: FP32 tiles without a map. */
static double fill(parsec_tiled_matrix_t *A, parsec_tiled_matrix_t *Ar, const uint16_t *dec,
                   int single, int lower)
{
    double expected = 0.0;
    for( int m = 0; m < A->mt; m++ )
        for( int n = 0; n <= (lower ? m : A->nt - 1); n++ ) {
            int rows = (m == A->mt-1) ? A->m - m * A->mb : A->mb;
            int cols = (n == A->nt-1) ? A->n - n * A->nb : A->nb;
            uint16_t d = dec ? dec[n * A->lmt + m] : 0;
            int fp32 = dec ? (DENSE_SP == d || DENSE_HP == d || DENSE_FP8 == d || LOW_RANK_SP == d) : single;
            int mine = A->super.rank_of(&A->super, m, n) == (uint32_t)A->super.myrank;
            /* dense: one rows x cols block; low rank: U (rows x rank), then V
             * (cols x rank) mb*rank elements later, leading dimension mb. In the
             * last tile row V has more rows than U. */
            size_t off[2] = { 0, 0 };
            int nrow[2] = { rows, 0 }, ncol[2] = { cols, 0 };
            if( LOW_RANK_DP == d || LOW_RANK_SP == d ) {
                int rank = (m + n) % 3;
                if( mine ) *(int *)tile(Ar, m, n) = rank;
                ncol[0] = rank;
                off[1] = (size_t)A->mb * rank; nrow[1] = cols; ncol[1] = rank;
            }
            void *t = mine ? tile(A, m, n) : NULL;
            for( int b = 0; b < 2; b++ )
                for( int j = 0; j < ncol[b]; j++ )
                    for( int i = 0; i < nrow[b]; i++ ) {
                        int v = f(m * A->mb + i, n * A->nb + b * A->nb / 2 + j);
                        size_t idx = off[b] + (size_t)j * A->mb + i;
                        expected += v;
                        if( t ) {
                            if( fp32 ) ((float *)t)[idx] = (float)v;
                            else      ((double *)t)[idx] = (double)v;
                        }
                    }
        }
    return expected;
}

int main(int argc, char **argv)
{
    int provided, rank, nodes, failures = 0, cases = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    parsec_context_t *parsec = parsec_init(2, &argc, &argv);
    const int M = 37, MB = 8;           /* partial edge tiles */

    for( int P = 1; P <= nodes; P++ ) {
        if( nodes % P ) continue;
        int Q = nodes / P;
        for( int c = 0; c < 4; c++ ) {   /* 0: double, 1: float, 2: mixed map, 3: NaN */
            int lower = (c != 1);
            parsec_matrix_block_cyclic_t A, Ar;
            parsec_matrix_block_cyclic_init(&A, 1 == c ? PARSEC_MATRIX_FLOAT : PARSEC_MATRIX_DOUBLE,
                                            PARSEC_MATRIX_TILE, rank, MB, MB, M, M, 0, 0, M, M, P, Q, 1, 1, 0, 0);
            A.mat = parsec_data_allocate((size_t)A.super.nb_local_tiles * A.super.bsiz
                                         * parsec_datadist_getsizeoftype(A.super.mtype));
            parsec_data_collection_set_key(&A.super.super, "A");
            parsec_matrix_block_cyclic_init(&Ar, PARSEC_MATRIX_INTEGER, PARSEC_MATRIX_TILE, rank, 1, 1,
                                            A.super.mt, A.super.nt, 0, 0, A.super.mt, A.super.nt, P, Q, 1, 1, 0, 0);
            Ar.mat = parsec_data_allocate((size_t)Ar.super.nb_local_tiles * sizeof(int));
            parsec_data_collection_set_key(&Ar.super.super, "Ar");

            uint16_t *dec = NULL;
            if( 2 == c ) {
                dec = malloc((size_t)A.super.lmt * A.super.lnt * sizeof(uint16_t));
                for( int m = 0; m < A.super.lmt; m++ )
                    for( int n = 0; n < A.super.lnt; n++ ) dec[n * A.super.lmt + m] = decision(m, n);
            }
            double expected = fill(&A.super, &Ar.super, dec, 1 == c, lower);
            if( 3 == c && A.super.super.rank_of(&A.super.super, 3, 1) == (uint32_t)rank )
                ((double *)tile(&A.super, 3, 1))[5] = NAN;

            double got = hicma_parsec_matrix_sum(parsec, &A.super, dec ? &Ar.super : NULL, dec, lower);
            int bad = (3 == c) ? !isnan(got) : (got != expected);
            MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if( 0 == rank && bad )
                printf("FAILED: grid %dx%d case %d: sum %.17g, expected %.17g\n", P, Q, c, got, 3 == c ? NAN : expected);
            failures += bad; cases++;

            free(dec);
            parsec_data_free(Ar.mat); parsec_tiled_matrix_destroy(&Ar.super);
            parsec_data_free(A.mat);  parsec_tiled_matrix_destroy(&A.super);
        }
    }
    if( 0 == rank )
        printf("%s: %d cases on %d process(es), %d failure(s)\n", failures ? "FAILED" : "PASSED", cases, nodes, failures);
    parsec_fini(&parsec);
    MPI_Finalize();
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

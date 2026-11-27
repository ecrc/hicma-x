/**
 * @file testing_lapack_tile.c
 * @brief Round-trip test of hicma_parsec_lapack_tile_convert and its named variants
 *
 * For every process grid P x Q of the run, every element type (double, float,
 * int, int8), both directions, full and lower-triangle copies, 2D block-cyclic
 * and symmetric block-cyclic descriptors, and partial edge tiles:
 *   1. fill the local LAPACK buffer with f(i, j) and copy it into the tiles;
 *   2. check every owned tile element against f (and that the tiles above
 *      the diagonal of a lower copy kept their sentinel);
 *   3. clear the buffer, copy the tiles back, check the buffer.
 * Then a descriptor on a P x Q grid converted with Q x P must be refused on
 * every rank, and nothing copied. Exit code 0 when everything matches.
 *
 * Also a small example of the conversion API for code that holds its matrix
 * in LAPACK layout.
 */
#include "hicma_parsec.h"

#define SENTINEL 77

static int f(int i, int j) { return (i * 7 + j * 13) % 101; }

static void put(void *base, size_t idx, parsec_matrix_type_t t, int v)
{
    switch(t) {
    case PARSEC_MATRIX_DOUBLE:  ((double *)base)[idx] = (double)v + 0.25; break;
    case PARSEC_MATRIX_FLOAT:   ((float  *)base)[idx] = (float)v + 0.5f; break;
    case PARSEC_MATRIX_INTEGER: ((int    *)base)[idx] = v - 50;          break;
    default:                    ((int8_t *)base)[idx] = (int8_t)v;       break;
    }
}

static int same(const void *base, size_t idx, parsec_matrix_type_t t, int v)
{
    switch(t) {
    case PARSEC_MATRIX_DOUBLE:  return ((const double *)base)[idx] == (double)v + 0.25;
    case PARSEC_MATRIX_FLOAT:   return ((const float  *)base)[idx] == (float)v + 0.5f;
    case PARSEC_MATRIX_INTEGER: return ((const int    *)base)[idx] == v - 50;
    default:                    return ((const int8_t *)base)[idx] == (int8_t)v;
    }
}

static parsec_datatype_t dtt_of(parsec_matrix_type_t t)
{
    switch(t) {
    case PARSEC_MATRIX_DOUBLE:  return parsec_datatype_double_t;
    case PARSEC_MATRIX_FLOAT:   return parsec_datatype_float_t;
    case PARSEC_MATRIX_INTEGER: return parsec_datatype_int_t;
    default:                    return parsec_datatype_int8_t;
    }
}

static void *tile_ptr(parsec_tiled_matrix_t *A, int m, int n)
{
    parsec_data_collection_t *dc = &A->super;
    parsec_data_t *d = dc->data_of(dc, m, n);
    return parsec_data_copy_get_ptr(parsec_data_get_copy(d, 0));
}

/* Visit every element of every tile this process owns: global (i, j), its
 * index in the tile and in the local LAPACK buffer, and whether the tile is
 * part of a lower copy */
typedef int (*visit_fn)(parsec_tiled_matrix_t *A, void *tile, size_t ti, size_t li,
                        int i, int j, int in_copy, void *buf, parsec_matrix_type_t t);

static int visit(parsec_tiled_matrix_t *A, int P, int Q, int lower, void *buf,
                 parsec_matrix_type_t t, visit_fn fn)
{
    int errors = 0;
    for(int m = 0; m < A->mt; m++)
        for(int n = 0; n < A->nt; n++) {
            if( A->super.rank_of(&A->super, m, n) != (uint32_t)A->super.myrank ) continue;
            if( NULL == A->super.data_of(&A->super, m, n) ) continue;
            int tempmm = (m == A->mt-1) ? A->m - m * A->mb : A->mb;
            int tempnn = (n == A->nt-1) ? A->n - n * A->nb : A->nb;
            void *tile = tile_ptr(A, m, n);
            size_t mm = (size_t)m / P, nn = (size_t)n / Q;
            for(int jj = 0; jj < tempnn; jj++)
                for(int ii = 0; ii < tempmm; ii++) {
                    size_t ti = (size_t)jj * A->mb + ii;
                    size_t li = ((size_t)nn * A->nb + jj) * A->llm + mm * A->mb + ii;
                    errors += fn(A, tile, ti, li, m * A->mb + ii, n * A->nb + jj,
                                 !lower || n <= m, buf, t);
                }
        }
    return errors;
}

static int fill_buf(parsec_tiled_matrix_t *A, void *tile, size_t ti, size_t li, int i, int j, int in_copy, void *buf, parsec_matrix_type_t t)
{ (void)A; (void)in_copy; put(tile, ti, t, SENTINEL); put(buf, li, t, f(i, j)); return 0; }

static int check_tile(parsec_tiled_matrix_t *A, void *tile, size_t ti, size_t li, int i, int j, int in_copy, void *buf, parsec_matrix_type_t t)
{ (void)A; (void)li; (void)buf; return !same(tile, ti, t, in_copy ? f(i, j) : SENTINEL); }

static int clear_buf(parsec_tiled_matrix_t *A, void *tile, size_t ti, size_t li, int i, int j, int in_copy, void *buf, parsec_matrix_type_t t)
{ (void)A; (void)tile; (void)ti; (void)i; (void)j; (void)in_copy; put(buf, li, t, SENTINEL); return 0; }

static int check_buf(parsec_tiled_matrix_t *A, void *tile, size_t ti, size_t li, int i, int j, int in_copy, void *buf, parsec_matrix_type_t t)
{ (void)A; (void)tile; (void)ti; return !same(buf, li, t, in_copy ? f(i, j) : SENTINEL); }

static int round_trip(parsec_context_t *parsec, parsec_tiled_matrix_t *A, int P, int Q,
                      int lower, parsec_matrix_type_t t)
{
    size_t elt = parsec_datadist_getsizeoftype(t);
    void *buf = malloc((size_t)A->llm * (size_t)A->lln * elt + 1);
    int errors = 0;
    errors += visit(A, P, Q, lower, buf, t, fill_buf);
    errors += 0 != hicma_parsec_lapack_tile_convert(parsec, A, buf, P, Q, dtt_of(t), lower, 1);
    errors += visit(A, P, Q, lower, buf, t, check_tile);
    errors += visit(A, P, Q, lower, buf, t, clear_buf);
    errors += 0 != hicma_parsec_lapack_tile_convert(parsec, A, buf, P, Q, dtt_of(t), lower, 0);
    errors += visit(A, P, Q, lower, buf, t, check_buf);
    free(buf);
    return errors;
}

int main(int argc, char **argv)
{
    int provided, rank, nodes, total = 0, cases = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    int cores = 2;
    parsec_context_t *parsec = parsec_init(cores, &argc, &argv);

    const parsec_matrix_type_t types[] = { PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_FLOAT,
                                           PARSEC_MATRIX_INTEGER, PARSEC_MATRIX_BYTE };
    const char *tname[] = { "byte", "int", "float", "double" };
    const int M = 37, N = 29, MB = 8, NB = 6;   /* partial edge tiles in both dimensions */

    for(int P = 1; P <= nodes; P++) {
        if( nodes % P ) continue;
        int Q = nodes / P;
        for(int ti = 0; ti < 4; ti++) {
            parsec_matrix_type_t t = types[ti];
            size_t elt = parsec_datadist_getsizeoftype(t);

            /* 2D block-cyclic, rectangular, full and lower copies */
            for(int lower = 0; lower <= 1; lower++) {
                parsec_matrix_block_cyclic_t dc;
                parsec_matrix_block_cyclic_init(&dc, t, PARSEC_MATRIX_TILE, rank, MB, NB, M, N,
                                                0, 0, M, N, P, Q, 1, 1, 0, 0);
                dc.mat = parsec_data_allocate((size_t)dc.super.nb_local_tiles * dc.super.bsiz * elt);
                parsec_data_collection_set_key(&dc.super.super, "A");
                int e = round_trip(parsec, &dc.super, P, Q, lower, t);
                MPI_Allreduce(MPI_IN_PLACE, &e, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                if( 0 == rank && e )
                    printf("FAILED: block-cyclic %dx%d %s lower=%d: %d mismatches\n", P, Q, tname[t], lower, e);
                total += e; cases++;
                parsec_data_free(dc.mat);
                parsec_tiled_matrix_destroy(&dc.super);
            }

            /* symmetric block-cyclic (lower storage), lower copy */
            {
                parsec_matrix_sym_block_cyclic_t dc;
                parsec_matrix_sym_block_cyclic_init(&dc, t, rank, MB, MB, M, M, 0, 0, M, M,
                                                    P, Q, PARSEC_MATRIX_LOWER);
                dc.mat = parsec_data_allocate((size_t)dc.super.nb_local_tiles * dc.super.bsiz * elt);
                parsec_data_collection_set_key(&dc.super.super, "S");
                int e = round_trip(parsec, &dc.super, P, Q, 1, t);
                MPI_Allreduce(MPI_IN_PLACE, &e, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                if( 0 == rank && e )
                    printf("FAILED: symmetric %dx%d %s: %d mismatches\n", P, Q, tname[t], e);
                total += e; cases++;
                parsec_data_free(dc.mat);
                parsec_tiled_matrix_destroy(&dc.super);
            }
        }

        /* A P x Q descriptor converted as Q x P must be refused everywhere */
        if( P != Q ) {
            parsec_matrix_block_cyclic_t dc;
            parsec_matrix_block_cyclic_init(&dc, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE, rank, MB, NB, M, N,
                                            0, 0, M, N, P, Q, 1, 1, 0, 0);
            dc.mat = parsec_data_allocate((size_t)dc.super.nb_local_tiles * dc.super.bsiz * sizeof(double));
            parsec_data_collection_set_key(&dc.super.super, "W");
            double *buf = calloc((size_t)dc.super.llm * dc.super.lln + 1, sizeof(double));
            int rc = hicma_parsec_lapack_tile_convert(parsec, &dc.super, buf, Q, P,
                                                      parsec_datatype_double_t, 0, 0);
            int e = (rc != -1);
            MPI_Allreduce(MPI_IN_PLACE, &e, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if( 0 == rank && e )
                printf("FAILED: a %dx%d descriptor converted as %dx%d was not refused\n", P, Q, Q, P);
            total += e; cases++;
            free(buf);
            parsec_data_free(dc.mat);
            parsec_tiled_matrix_destroy(&dc.super);
        }
    }

    /* the named variants are the generic function with fixed arguments: one call each */
    {
        parsec_matrix_block_cyclic_t dc;
        int P = 1, Q = nodes;
        parsec_matrix_block_cyclic_init(&dc, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE, rank, MB, NB, M, N,
                                        0, 0, M, N, P, Q, 1, 1, 0, 0);
        dc.mat = parsec_data_allocate((size_t)dc.super.nb_local_tiles * dc.super.bsiz * sizeof(double));
        parsec_data_collection_set_key(&dc.super.super, "V");
        double *buf = calloc((size_t)dc.super.llm * dc.super.lln + 1, sizeof(double));
        int e = 0;
        e += 0 != hicma_parsec_Lapack_to_Tile(parsec, &dc.super, buf, P, Q);
        e += 0 != hicma_parsec_Tile_to_Lapack(parsec, &dc.super, buf, P, Q);
        e += 0 != hicma_parsec_Tile_to_Lapack_sym(parsec, &dc.super, buf, P, Q);
        e += 0 != hicma_parsec_Lapack_to_Tile_sym(parsec, &dc.super, buf, P, Q);
        MPI_Allreduce(MPI_IN_PLACE, &e, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if( 0 == rank && e ) printf("FAILED: named variants returned errors (%d)\n", e);
        total += e; cases++;
        free(buf);
        parsec_data_free(dc.mat);
        parsec_tiled_matrix_destroy(&dc.super);
    }

    if( 0 == rank )
        printf("%s: %d cases on %d process(es), %d failure(s)\n", total ? "FAILED" : "PASSED", cases, nodes, total);
    parsec_fini(&parsec);
    MPI_Finalize();
    return total ? EXIT_FAILURE : EXIT_SUCCESS;
}

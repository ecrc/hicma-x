/**
 * @file hicma_parsec_timing.h
 * @brief Wall-clock timing macros used by the library and the test drivers
 *
 * Same conventions as DPLASMA's test helpers (common_timing.h), which are not
 * installed with DPLASMA: TIME_* measure on the calling rank, SYNC_TIME_*
 * measure between two MPI barriers and print on rank 0. The elapsed-time
 * variables are defined in hicma_parsec_internal.c.
 *
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#ifndef HICMA_PARSEC_TIMING_H
#define HICMA_PARSEC_TIMING_H

#include <stdio.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

extern double time_elapsed;
extern double sync_time_elapsed;

/* DPLASMA's test helpers define the same names; the in-tree TRMM drivers
 * include both, so define ours last and unconditionally */
#undef TIME_START
#undef TIME_STOP
#undef TIME_PRINT
#undef SYNC_TIME_START
#undef SYNC_TIME_STOP
#undef SYNC_TIME_PRINT

#define TIME_START() do { time_elapsed = MPI_Wtime(); } while(0)
#define TIME_STOP()  do { time_elapsed = MPI_Wtime() - time_elapsed; } while(0)
#define TIME_PRINT(rank, print) do {                            \
        TIME_STOP();                                            \
        printf("[%4d] TIME(s) %12.5f : ", (rank), time_elapsed); \
        printf print;                                           \
        fflush(stdout);                                         \
    } while(0)

#define SYNC_TIME_START() do {                                  \
        MPI_Barrier(MPI_COMM_WORLD);                            \
        sync_time_elapsed = MPI_Wtime();                        \
    } while(0)
#define SYNC_TIME_STOP() do {                                   \
        MPI_Barrier(MPI_COMM_WORLD);                            \
        sync_time_elapsed = MPI_Wtime() - sync_time_elapsed;    \
    } while(0)
#define SYNC_TIME_PRINT(rank, print) do {                       \
        SYNC_TIME_STOP();                                       \
        if( 0 == (rank) ) {                                     \
            printf("[****] TIME(s) %12.5f : ", sync_time_elapsed); \
            printf print;                                       \
            fflush(stdout);                                     \
            fflush(stderr);                                     \
        }                                                       \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif /* HICMA_PARSEC_TIMING_H */

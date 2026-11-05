/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#ifndef HICMA_KERNEL_TIME_H
#define HICMA_KERNEL_TIME_H

#ifndef PRINT_KERNEL_TIME
#define PRINT_KERNEL_TIME 0
#endif

#include <pthread.h>
#include "parsec/mca/device/device.h"
#include <stdint.h>
#include <stdlib.h>

#define HICMA_KERNEL_TIME_BUCKETS 16384

/*
 * Start-time table and per-device accumulators. The state lives in
 * hicma_kernel_time.c (one table and one lock for the whole library); it
 * used to be `static` in this header, which gave every including file its
 * own private table and lock.
 */
void hicma_kernel_time_record(const void *task, double start_time);
double hicma_kernel_time_take(const void *task, double fallback_start_time);
double hicma_kernel_time_accumulate(const parsec_task_t *task,
                                    hicma_parsec_params_t *params_tlr,
                                    int cpu_id,
                                    double start_time,
                                    double end_time,
                                    const char **sum_scope,
                                    int *sum_id);

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
int hicma_kernel_time_gpu_event_begin(const void *task,
                                      cudaStream_t stream,
                                      cudaEvent_t *start_event,
                                      cudaEvent_t *stop_event);
void hicma_kernel_time_gpu_event_record(const void *task,
                                        int gpu_id,
                                        cudaStream_t stream,
                                        cudaEvent_t start_event,
                                        cudaEvent_t stop_event);
void hicma_kernel_time_gpu_event_abort(cudaEvent_t start_event,
                                       cudaEvent_t stop_event);
#endif

int hicma_kernel_time_gpu_event_take(const void *task,
                                     hicma_parsec_params_t *params_tlr,
                                     int *gpu_id,
                                     double *gpu_exe_time,
                                     double *gpu_sum_time);

static inline void hicma_kernel_time_print_task(const char *task_name,
                                                int band_size_dense,
                                                int nodes,
                                                int matrix,
                                                int m,
                                                int n,
                                                int k,
                                                double end_time,
                                                double start_time,
                                                double task_exe_time,
                                                const char *sum_time_scope,
                                                int sum_time_id,
                                                double sum_time,
                                                int has_gpu_time,
                                                int gpu_time_id,
                                                double gpu_exe_time,
                                                double gpu_sum_time)
{
    if(has_gpu_time) {
        fprintf(stderr, "band_size_dense %d Nodes %d Matrix %d %s %d %d %d end_time %lf start_time %lf task_exe_time %lf task_sum_time_%s_%d %lf gpu_exe_time %lf gpu_sum_time_%d %lf\n",
                band_size_dense, nodes, matrix, task_name, m, n, k,
                end_time, start_time, task_exe_time, sum_time_scope, sum_time_id, sum_time,
                gpu_exe_time, gpu_time_id, gpu_sum_time);
    } else {
        fprintf(stderr, "band_size_dense %d Nodes %d Matrix %d %s %d %d %d end_time %lf start_time %lf task_exe_time %lf task_sum_time_%s_%d %lf\n",
                band_size_dense, nodes, matrix, task_name, m, n, k,
                end_time, start_time, task_exe_time, sum_time_scope, sum_time_id, sum_time);
    }
}

static inline void hicma_kernel_time_print_gemm(int band_size_dense,
                                                int nodes,
                                                int matrix,
                                                int m,
                                                int n,
                                                int k,
                                                double end_time,
                                                double start_time,
                                                double task_exe_time,
                                                const char *sum_time_scope,
                                                int sum_time_id,
                                                double sum_time,
                                                int has_gpu_time,
                                                int gpu_time_id,
                                                double gpu_exe_time,
                                                double gpu_sum_time)
{
    hicma_kernel_time_print_task("GEMM", band_size_dense, nodes, matrix, m, n, k,
                                 end_time, start_time, task_exe_time, sum_time_scope, sum_time_id, sum_time,
                                 has_gpu_time, gpu_time_id, gpu_exe_time, gpu_sum_time);
}

#endif /* HICMA_KERNEL_TIME_H */

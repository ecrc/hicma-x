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

typedef struct hicma_kernel_time_entry_s {
    const void *task;
    double start_time;
    struct hicma_kernel_time_entry_s *next;
} hicma_kernel_time_entry_t;

static pthread_mutex_t hicma_kernel_time_lock = PTHREAD_MUTEX_INITIALIZER;
static hicma_kernel_time_entry_t *hicma_kernel_time_table[HICMA_KERNEL_TIME_BUCKETS];

static inline size_t hicma_kernel_time_hash(const void *task)
{
    uintptr_t key = (uintptr_t)task;
    key >>= 4;
    return (size_t)(key % HICMA_KERNEL_TIME_BUCKETS);
}

static inline void hicma_kernel_time_record(const void *task, double start_time)
{
    size_t bucket = hicma_kernel_time_hash(task);
    hicma_kernel_time_entry_t *entry;

    pthread_mutex_lock(&hicma_kernel_time_lock);
    for(entry = hicma_kernel_time_table[bucket]; entry != NULL; entry = entry->next) {
        if(entry->task == task) {
            entry->start_time = start_time;
            pthread_mutex_unlock(&hicma_kernel_time_lock);
            return;
        }
    }

    entry = (hicma_kernel_time_entry_t *)malloc(sizeof(*entry));
    if(entry != NULL) {
        entry->task = task;
        entry->start_time = start_time;
        entry->next = hicma_kernel_time_table[bucket];
        hicma_kernel_time_table[bucket] = entry;
    }
    pthread_mutex_unlock(&hicma_kernel_time_lock);
}

static inline double hicma_kernel_time_take(const void *task, double fallback_start_time)
{
    size_t bucket = hicma_kernel_time_hash(task);
    hicma_kernel_time_entry_t **entryp;
    hicma_kernel_time_entry_t *entry;
    double start_time = fallback_start_time;

    pthread_mutex_lock(&hicma_kernel_time_lock);
    for(entryp = &hicma_kernel_time_table[bucket]; *entryp != NULL; entryp = &(*entryp)->next) {
        entry = *entryp;
        if(entry->task == task) {
            start_time = entry->start_time;
            *entryp = entry->next;
            free(entry);
            break;
        }
    }
    pthread_mutex_unlock(&hicma_kernel_time_lock);

    return start_time;
}

static inline double hicma_kernel_time_add_interval(hicma_kernel_time_interval_t **intervals,
                                                    double *sum_array,
                                                    int id,
                                                    double start_time,
                                                    double end_time)
{
    hicma_kernel_time_interval_t **intervalp;
    hicma_kernel_time_interval_t *interval;
    hicma_kernel_time_interval_t *new_interval;
    double merged_start = start_time;
    double merged_end = end_time;

    if(end_time <= start_time) {
        return sum_array[id];
    }

    intervalp = &intervals[id];
    while(NULL != *intervalp && (*intervalp)->end_time < merged_start) {
        intervalp = &(*intervalp)->next;
    }

    while(NULL != *intervalp && (*intervalp)->start_time <= merged_end) {
        interval = *intervalp;
        if(interval->start_time < merged_start) {
            merged_start = interval->start_time;
        }
        if(interval->end_time > merged_end) {
            merged_end = interval->end_time;
        }
        sum_array[id] -= interval->end_time - interval->start_time;
        *intervalp = interval->next;
        free(interval);
    }

    new_interval = (hicma_kernel_time_interval_t *)malloc(sizeof(*new_interval));
    if(NULL == new_interval) {
        sum_array[id] += end_time - start_time;
        return sum_array[id];
    }

    new_interval->start_time = merged_start;
    new_interval->end_time = merged_end;
    new_interval->next = *intervalp;
    *intervalp = new_interval;

    sum_array[id] += merged_end - merged_start;
    return sum_array[id];
}

static inline double hicma_kernel_time_accumulate(const parsec_task_t *task,
                                                  hicma_parsec_params_t *params_tlr,
                                                  int cpu_id,
                                                  double start_time,
                                                  double end_time,
                                                  const char **sum_scope,
                                                  int *sum_id)
{
    const parsec_device_module_t *device = (NULL != task) ? task->selected_device : NULL;
    double *sum_array = (NULL != params_tlr) ? params_tlr->kernel_time_cpu : NULL;
    hicma_kernel_time_interval_t **intervals = (NULL != params_tlr) ? params_tlr->kernel_time_cpu_intervals : NULL;
    int sum_count = (NULL != params_tlr) ? params_tlr->kernel_time_cpu_count : 0;
    int id = cpu_id;
    int is_gpu = 0;
    double sum_time;

    *sum_scope = "cpu";
    if(NULL != device && PARSEC_DEV_IS_GPU(device->type)) {
        *sum_scope = "gpu";
        is_gpu = 1;
        sum_array = (NULL != params_tlr) ? params_tlr->kernel_time_gpu : NULL;
        intervals = (NULL != params_tlr) ? params_tlr->kernel_time_gpu_intervals : NULL;
        sum_count = (NULL != params_tlr) ? params_tlr->kernel_time_gpu_count : 0;
        id = device->device_index;
    }

    if(id < 0 || id >= sum_count || NULL == sum_array || NULL == intervals) {
        id = 0;
    }

    pthread_mutex_lock(&hicma_kernel_time_lock);
    if(NULL != sum_array && NULL != intervals && sum_count > 0) {
        if(is_gpu) {
            sum_time = hicma_kernel_time_add_interval(intervals, sum_array, id, start_time, end_time);
        } else {
            sum_array[id] += end_time - start_time;
            sum_time = sum_array[id];
        }
    } else {
        sum_time = end_time - start_time;
    }
    pthread_mutex_unlock(&hicma_kernel_time_lock);

    *sum_id = id;
    return sum_time;
}

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

/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"
#include "hicma_kernel_time.h"

typedef struct hicma_kernel_gpu_event_entry_s {
    const void *task;
    int gpu_id;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    int cuda_device;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
#endif
    struct hicma_kernel_gpu_event_entry_s *next;
} hicma_kernel_gpu_event_entry_t;

static pthread_mutex_t hicma_kernel_gpu_event_lock = PTHREAD_MUTEX_INITIALIZER;
static hicma_kernel_gpu_event_entry_t *hicma_kernel_gpu_event_table[HICMA_KERNEL_TIME_BUCKETS];

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
int hicma_kernel_time_gpu_event_begin(const void *task,
                                      cudaStream_t stream,
                                      cudaEvent_t *start_event,
                                      cudaEvent_t *stop_event)
{
    if(NULL == task || NULL == start_event || NULL == stop_event) {
        return 0;
    }

    *start_event = NULL;
    *stop_event = NULL;
    if(cudaSuccess != cudaEventCreate(start_event)) {
        return 0;
    }
    if(cudaSuccess != cudaEventCreate(stop_event)) {
        cudaEventDestroy(*start_event);
        *start_event = NULL;
        return 0;
    }
    if(cudaSuccess != cudaEventRecord(*start_event, stream)) {
        cudaEventDestroy(*start_event);
        cudaEventDestroy(*stop_event);
        *start_event = NULL;
        *stop_event = NULL;
        return 0;
    }

    return 1;
}

void hicma_kernel_time_gpu_event_abort(cudaEvent_t start_event,
                                       cudaEvent_t stop_event)
{
    if(NULL != start_event) {
        cudaEventDestroy(start_event);
    }
    if(NULL != stop_event) {
        cudaEventDestroy(stop_event);
    }
}

void hicma_kernel_time_gpu_event_record(const void *task,
                                        int gpu_id,
                                        cudaStream_t stream,
                                        cudaEvent_t start_event,
                                        cudaEvent_t stop_event)
{
    size_t bucket;
    hicma_kernel_gpu_event_entry_t *entry;
    int cuda_device = -1;

    if(NULL == task || NULL == start_event || NULL == stop_event) {
        hicma_kernel_time_gpu_event_abort(start_event, stop_event);
        return;
    }

    cudaGetDevice(&cuda_device);

    if(cudaSuccess != cudaEventRecord(stop_event, stream)) {
        hicma_kernel_time_gpu_event_abort(start_event, stop_event);
        return;
    }

    bucket = hicma_kernel_time_hash(task);
    pthread_mutex_lock(&hicma_kernel_gpu_event_lock);
    for(entry = hicma_kernel_gpu_event_table[bucket]; entry != NULL; entry = entry->next) {
        if(entry->task == task) {
            hicma_kernel_time_gpu_event_abort(entry->start_event, entry->stop_event);
            entry->gpu_id = gpu_id;
            entry->cuda_device = cuda_device;
            entry->start_event = start_event;
            entry->stop_event = stop_event;
            pthread_mutex_unlock(&hicma_kernel_gpu_event_lock);
            return;
        }
    }

    entry = (hicma_kernel_gpu_event_entry_t *)malloc(sizeof(*entry));
    if(NULL == entry) {
        pthread_mutex_unlock(&hicma_kernel_gpu_event_lock);
        hicma_kernel_time_gpu_event_abort(start_event, stop_event);
        return;
    }

    entry->task = task;
    entry->gpu_id = gpu_id;
    entry->cuda_device = cuda_device;
    entry->start_event = start_event;
    entry->stop_event = stop_event;
    entry->next = hicma_kernel_gpu_event_table[bucket];
    hicma_kernel_gpu_event_table[bucket] = entry;
    pthread_mutex_unlock(&hicma_kernel_gpu_event_lock);
}
#endif

int hicma_kernel_time_gpu_event_take(const void *task,
                                     hicma_parsec_params_t *params_tlr,
                                     int *gpu_id,
                                     double *gpu_exe_time,
                                     double *gpu_sum_time)
{
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    size_t bucket;
    hicma_kernel_gpu_event_entry_t **entryp;
    hicma_kernel_gpu_event_entry_t *entry = NULL;
    float elapsed_ms = 0.0f;
    int id;
    int old_cuda_device = -1;
    int restore_cuda_device = 0;

    if(NULL == task || NULL == gpu_id || NULL == gpu_exe_time || NULL == gpu_sum_time) {
        return 0;
    }

    bucket = hicma_kernel_time_hash(task);
    pthread_mutex_lock(&hicma_kernel_gpu_event_lock);
    for(entryp = &hicma_kernel_gpu_event_table[bucket]; *entryp != NULL; entryp = &(*entryp)->next) {
        if((*entryp)->task == task) {
            entry = *entryp;
            *entryp = entry->next;
            break;
        }
    }
    pthread_mutex_unlock(&hicma_kernel_gpu_event_lock);

    if(NULL == entry) {
        return 0;
    }

    if(entry->cuda_device >= 0 &&
       cudaSuccess == cudaGetDevice(&old_cuda_device) &&
       old_cuda_device != entry->cuda_device &&
       cudaSuccess == cudaSetDevice(entry->cuda_device)) {
        restore_cuda_device = 1;
    }

    if(cudaSuccess != cudaEventElapsedTime(&elapsed_ms, entry->start_event, entry->stop_event)) {
        cudaEventSynchronize(entry->stop_event);
        if(cudaSuccess != cudaEventElapsedTime(&elapsed_ms, entry->start_event, entry->stop_event)) {
            hicma_kernel_time_gpu_event_abort(entry->start_event, entry->stop_event);
            if(restore_cuda_device) {
                cudaSetDevice(old_cuda_device);
            }
            free(entry);
            return 0;
        }
    }

    id = entry->gpu_id;
    if(NULL == params_tlr || NULL == params_tlr->kernel_time_gpu_event ||
       id < 0 || id >= params_tlr->kernel_time_gpu_event_count) {
        id = 0;
    }

    *gpu_exe_time = (double)elapsed_ms / 1000.0;
    pthread_mutex_lock(&hicma_kernel_gpu_event_lock);
    if(NULL != params_tlr && NULL != params_tlr->kernel_time_gpu_event &&
       params_tlr->kernel_time_gpu_event_count > 0) {
        params_tlr->kernel_time_gpu_event[id] += *gpu_exe_time;
        *gpu_sum_time = params_tlr->kernel_time_gpu_event[id];
    } else {
        *gpu_sum_time = *gpu_exe_time;
    }
    pthread_mutex_unlock(&hicma_kernel_gpu_event_lock);

    *gpu_id = id;
    hicma_kernel_time_gpu_event_abort(entry->start_event, entry->stop_event);
    if(restore_cuda_device) {
        cudaSetDevice(old_cuda_device);
    }
    free(entry);
    return 1;
#else
    (void)task;
    (void)params_tlr;
    (void)gpu_id;
    (void)gpu_exe_time;
    (void)gpu_sum_time;
    return 0;
#endif
}

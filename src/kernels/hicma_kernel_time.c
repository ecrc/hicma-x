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
#include <stdatomic.h>

typedef struct hicma_kernel_time_entry_s {
    const void *task;
    double start_time;
    struct hicma_kernel_time_entry_s *next;
} hicma_kernel_time_entry_t;

static pthread_mutex_t hicma_kernel_time_lock = PTHREAD_MUTEX_INITIALIZER;
static hicma_kernel_time_entry_t *hicma_kernel_time_table[HICMA_KERNEL_TIME_BUCKETS];
/* Entries in the locked overflow table; lets take() skip the lock when empty. */
static _Atomic long hicma_kernel_time_overflow = 0;

static size_t hicma_kernel_time_hash(const void *task)
{
    uintptr_t key = (uintptr_t)task;
    key >>= 4;
    return (size_t)(key % HICMA_KERNEL_TIME_BUCKETS);
}

static void hicma_kernel_time_record_locked(const void *task, double start_time)
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
        atomic_fetch_add(&hicma_kernel_time_overflow, 1);
    }
    pthread_mutex_unlock(&hicma_kernel_time_lock);
}

static double hicma_kernel_time_take_locked(const void *task, double fallback_start_time)
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
            atomic_fetch_sub(&hicma_kernel_time_overflow, 1);
            break;
        }
    }
    pthread_mutex_unlock(&hicma_kernel_time_lock);

    return start_time;
}

/*
 * Lock-free start-time table used on every task begin/complete (the GEMM
 * wrappers time every GEMM for gather_time). A task claims a slot in a
 * bounded window by CAS on the key and releases it when it completes, so
 * the common path takes no lock and allocates nothing; the locked list
 * above only catches the rare overflow when a window is full.
 */
#define HICMA_KT_SLOTS  16384   /* power of two, >> tasks in flight */
#define HICMA_KT_WINDOW 64

static _Atomic(const void *) hicma_kt_keys[HICMA_KT_SLOTS];
static _Atomic double hicma_kt_times[HICMA_KT_SLOTS];

static inline size_t hicma_kt_slot(const void *task)
{
    uintptr_t x = (uintptr_t)task;
    x ^= x >> 17;
    x *= (uintptr_t)0x9E3779B97F4A7C15ULL;
    return (size_t)(x >> 13) & (HICMA_KT_SLOTS - 1);
}

void hicma_kernel_time_record(const void *task, double start_time)
{
    size_t base = hicma_kt_slot(task);
    /* Same task recorded again (e.g. a re-executed hook): update in place. */
    for( size_t i = 0; i < HICMA_KT_WINDOW; i++ ) {
        size_t idx = (base + i) & (HICMA_KT_SLOTS - 1);
        if( atomic_load_explicit(&hicma_kt_keys[idx], memory_order_acquire) == task ) {
            atomic_store_explicit(&hicma_kt_times[idx], start_time, memory_order_release);
            return;
        }
    }
    for( size_t i = 0; i < HICMA_KT_WINDOW; i++ ) {
        size_t idx = (base + i) & (HICMA_KT_SLOTS - 1);
        const void *expected = NULL;
        if( atomic_compare_exchange_strong_explicit(&hicma_kt_keys[idx], &expected, task,
                                                    memory_order_acq_rel, memory_order_relaxed) ) {
            atomic_store_explicit(&hicma_kt_times[idx], start_time, memory_order_release);
            return;
        }
    }
    hicma_kernel_time_record_locked(task, start_time);
}

double hicma_kernel_time_take(const void *task, double fallback_start_time)
{
    size_t base = hicma_kt_slot(task);
    for( size_t i = 0; i < HICMA_KT_WINDOW; i++ ) {
        size_t idx = (base + i) & (HICMA_KT_SLOTS - 1);
        if( atomic_load_explicit(&hicma_kt_keys[idx], memory_order_acquire) == task ) {
            double t = atomic_load_explicit(&hicma_kt_times[idx], memory_order_acquire);
            atomic_store_explicit(&hicma_kt_keys[idx], NULL, memory_order_release);
            return t;
        }
    }
    /* Tasks whose begin was never recorded (e.g. CPU chores of wrapped GPU
     * hooks) end here: skip the lock unless something overflowed. */
    if( 0 == atomic_load(&hicma_kernel_time_overflow) ) {
        return fallback_start_time;
    }
    return hicma_kernel_time_take_locked(task, fallback_start_time);
}

static double hicma_kernel_time_add_interval(hicma_kernel_time_interval_t **intervals,
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

double hicma_kernel_time_accumulate(const parsec_task_t *task,
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

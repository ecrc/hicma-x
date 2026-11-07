/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

/**
 * @file potrf_common.c
 * @brief Code shared by the Cholesky wrappers, see potrf_common.h
 */

#include "potrf_common.h"
#include "hicma_kernel_time.h"
#include <pthread.h>

/* ============================================================================
 * Taskpool registry: the generic hooks find their parameters here
 * ============================================================================ */

#define HICMA_POTRF_MAX_TASKPOOLS 16

typedef struct hicma_potrf_reg_s {
    const parsec_taskpool_t *tp;
    hicma_parsec_params_t   *params;
    parsec_tiled_matrix_t   *descA;
} hicma_potrf_reg_t;

static hicma_potrf_reg_t hicma_potrf_registry[HICMA_POTRF_MAX_TASKPOOLS];
static pthread_mutex_t   hicma_potrf_registry_lock = PTHREAD_MUTEX_INITIALIZER;

/* Registration happens in New() and removal in Destruct(), both outside of
 * parsec_context_start/wait, so the hooks read the table without a lock. */
static int hicma_potrf_remember(parsec_taskpool_t *tp, hicma_parsec_params_t *params,
                                parsec_tiled_matrix_t *descA)
{
    int slot = -1;
    pthread_mutex_lock(&hicma_potrf_registry_lock);
    for( int i = 0; i < HICMA_POTRF_MAX_TASKPOOLS; i++ ) {
        if( hicma_potrf_registry[i].tp == tp ) { slot = i; break; }
        if( NULL == hicma_potrf_registry[i].tp && slot < 0 ) slot = i;
    }
    if( slot >= 0 ) {
        hicma_potrf_registry[slot].tp = tp;
        hicma_potrf_registry[slot].params = params;
        hicma_potrf_registry[slot].descA = descA;
    }
    pthread_mutex_unlock(&hicma_potrf_registry_lock);
    return slot < 0 ? -1 : 0;
}

void hicma_potrf_forget(parsec_taskpool_t *tp)
{
    pthread_mutex_lock(&hicma_potrf_registry_lock);
    for( int i = 0; i < HICMA_POTRF_MAX_TASKPOOLS; i++ ) {
        if( hicma_potrf_registry[i].tp == tp ) {
            hicma_potrf_registry[i].tp = NULL;
            hicma_potrf_registry[i].params = NULL;
            hicma_potrf_registry[i].descA = NULL;
        }
    }
    pthread_mutex_unlock(&hicma_potrf_registry_lock);
}

static inline const hicma_potrf_reg_t *hicma_potrf_lookup(const parsec_task_t *task)
{
    for( int i = 0; i < HICMA_POTRF_MAX_TASKPOOLS; i++ ) {
        if( hicma_potrf_registry[i].tp == task->taskpool ) {
            return &hicma_potrf_registry[i];
        }
    }
    fprintf(stderr, "Fatal: task %s belongs to a taskpool that hicma_potrf_install_hooks does not know\n",
            task->task_class->name);
    abort();
    return NULL;
}

/* ============================================================================
 * Timing hooks: identical for every Cholesky JDF
 *
 * Locals: potrf(k) -> [0]; trsm(m, k) -> m [0], k [1]; syrk(k, m) -> k [0],
 * m [1]; gemm(m, n, k) -> [0], [1], [2].
 * ============================================================================ */

static int hicma_potrf_wrap_potrf(parsec_execution_stream_t *es, parsec_task_t *task)
{
    hicma_parsec_params_t *p = hicma_potrf_lookup(task)->params;
    double start_time = MPI_Wtime();
#if PRINT_KERNEL_TIME
    /* Only taken back when PRINT_KERNEL_TIME is on; recording it
     * unconditionally leaves stale entries that a reused task object
     * (e.g. a later GEMM) would pick up. */
    hicma_kernel_time_record(task, start_time);
#endif
    p->potrf_time_temp = start_time;
    p->gather_time_tmp[es->th_id] = start_time;
    return p->wrap_potrf(es, task);
}

static int hicma_potrf_wrap_potrf_complete(parsec_execution_stream_t *es, parsec_task_t *task)
{
    const hicma_potrf_reg_t *r = hicma_potrf_lookup(task);
    hicma_parsec_params_t *p = r->params;
    int val = p->wrap_potrf_complete(es, task);
    double end_time = MPI_Wtime();
    p->potrf_time += end_time - p->potrf_time_temp;
    p->gather_time[es->th_id] += end_time - p->gather_time_tmp[es->th_id];

#if PRINT_CRITICAL_PATH_TIME
    fprintf(stderr, "OUT_critical_path_time band_size_dense %d Nodes %d Matrix %d POTRF %d end_time %lf start_time %lf task_exe_time %lf sum_time %lf\n",
            p->band_size_dense, r->descA->super.nodes, r->descA->lm,
            task->locals[0].value, end_time, p->potrf_time_temp,
            end_time - p->potrf_time_temp, p->potrf_time);
#endif
#if PRINT_KERNEL_TIME
    hicma_kernel_time_report(task, p, r->descA, es->th_id,
                             "POTRF", task->locals[0].value, -1, -1,
                             hicma_kernel_time_take(task, end_time), end_time);
#endif
    (void)r;
    return val;
}

/* TRSM(m, k) and SYRK(k, m): only the first task of each column (m == k+1)
 * contributes to the critical-path time, every task to the per-thread time */
static int hicma_potrf_wrap_trsm(parsec_execution_stream_t *es, parsec_task_t *task)
{
    hicma_parsec_params_t *p = hicma_potrf_lookup(task)->params;
    int m = task->locals[0].value, k = task->locals[1].value;
    double start_time = MPI_Wtime();
#if PRINT_KERNEL_TIME
    hicma_kernel_time_record(task, start_time);
#endif
    p->gather_time_tmp[es->th_id] = start_time;
    if( m == k + 1 ) {
        p->trsm_time_temp = start_time;
    }
    return p->wrap_trsm(es, task);
}

static int hicma_potrf_wrap_trsm_complete(parsec_execution_stream_t *es, parsec_task_t *task)
{
    const hicma_potrf_reg_t *r = hicma_potrf_lookup(task);
    hicma_parsec_params_t *p = r->params;
    int m = task->locals[0].value, k = task->locals[1].value;
    int val = p->wrap_trsm_complete(es, task);
    double end_time = MPI_Wtime();
    if( m == k + 1 ) {
        p->trsm_time += end_time - p->trsm_time_temp;
#if PRINT_CRITICAL_PATH_TIME
        fprintf(stderr, "OUT_critical_path_time band_size_dense %d Nodes %d Matrix %d TRSM %d end_time %lf start_time %lf task_exe_time %lf sum_time %lf\n",
                p->band_size_dense, r->descA->super.nodes, r->descA->lm,
                k, end_time, p->trsm_time_temp, end_time - p->trsm_time_temp, p->trsm_time);
#endif
    }
    p->gather_time[es->th_id] += end_time - p->gather_time_tmp[es->th_id];
#if PRINT_KERNEL_TIME
    hicma_kernel_time_report(task, p, r->descA, es->th_id,
                             "TRSM", m, k, -1, hicma_kernel_time_take(task, end_time), end_time);
#endif
    (void)r;
    return val;
}

static int hicma_potrf_wrap_syrk(parsec_execution_stream_t *es, parsec_task_t *task)
{
    hicma_parsec_params_t *p = hicma_potrf_lookup(task)->params;
    int k = task->locals[0].value, m = task->locals[1].value;
    double start_time = MPI_Wtime();
#if PRINT_KERNEL_TIME
    hicma_kernel_time_record(task, start_time);
#endif
    p->gather_time_tmp[es->th_id] = start_time;
    if( m == k + 1 ) {
        p->syrk_time_temp = start_time;
    }
    return p->wrap_syrk(es, task);
}

static int hicma_potrf_wrap_syrk_complete(parsec_execution_stream_t *es, parsec_task_t *task)
{
    const hicma_potrf_reg_t *r = hicma_potrf_lookup(task);
    hicma_parsec_params_t *p = r->params;
    int k = task->locals[0].value, m = task->locals[1].value;
    int val = p->wrap_syrk_complete(es, task);
    double end_time = MPI_Wtime();
    if( m == k + 1 ) {
        p->syrk_time += end_time - p->syrk_time_temp;
#if PRINT_CRITICAL_PATH_TIME
        fprintf(stderr, "OUT_critical_path_time band_size_dense %d Nodes %d Matrix %d SYRK %d end_time %lf start_time %lf task_exe_time %lf sum_time %lf\n",
                p->band_size_dense, r->descA->super.nodes, r->descA->lm,
                k, end_time, p->syrk_time_temp, end_time - p->syrk_time_temp, p->syrk_time);
#endif
    }
    p->gather_time[es->th_id] += end_time - p->gather_time_tmp[es->th_id];
#if PRINT_KERNEL_TIME
    hicma_kernel_time_report(task, p, r->descA, es->th_id,
                             "SYRK", m, k, -1, hicma_kernel_time_take(task, end_time), end_time);
#endif
    (void)r;
    return val;
}

static int hicma_potrf_wrap_gemm(parsec_execution_stream_t *es, parsec_task_t *task)
{
    hicma_parsec_params_t *p = hicma_potrf_lookup(task)->params;
    hicma_kernel_time_record(task, MPI_Wtime());
    return p->wrap_gemm(es, task);
}

static int hicma_potrf_wrap_gemm_complete(parsec_execution_stream_t *es, parsec_task_t *task)
{
    const hicma_potrf_reg_t *r = hicma_potrf_lookup(task);
    hicma_parsec_params_t *p = r->params;
    double start_time = hicma_kernel_time_take(task, MPI_Wtime());
    int val = p->wrap_gemm_complete(es, task);
    double end_time = MPI_Wtime();
    p->gather_time[es->th_id] += end_time - start_time;
#if PRINT_KERNEL_TIME
    hicma_kernel_time_report(task, p, r->descA, es->th_id,
                             "GEMM", task->locals[0].value, task->locals[1].value, task->locals[2].value,
                             start_time, end_time);
#endif
    (void)r;
    return val;
}

/* ============================================================================
 * Hook installation
 * ============================================================================ */

/* The chore of tc whose device type matches `want`; the generated table ends
 * with a PARSEC_DEV_NONE entry */
static __parsec_chore_t *hicma_potrf_find_chore(const parsec_task_class_t *tc, uint8_t want)
{
    for( const __parsec_chore_t *c = tc->incarnations; PARSEC_DEV_NONE != c->type; c++ ) {
        if( c->type & want ) {
            return (__parsec_chore_t *)c;
        }
    }
    return NULL;
}

int hicma_potrf_install_hooks(parsec_taskpool_t *tp,
                              hicma_parsec_params_t *params,
                              parsec_tiled_matrix_t *descA,
                              int nb_gpus, int hnb, int mb,
                              const hicma_potrf_evaluate_t *eval_gpu)
{
    static const char *names[4] = { "potrf_dpotrf", "potrf_dtrsm", "potrf_dsyrk", "potrf_dgemm" };
    parsec_hook_t *const wraps[4] = { hicma_potrf_wrap_potrf, hicma_potrf_wrap_trsm,
                                      hicma_potrf_wrap_syrk, hicma_potrf_wrap_gemm };
    parsec_hook_t *const completes[4] = { hicma_potrf_wrap_potrf_complete, hicma_potrf_wrap_trsm_complete,
                                          hicma_potrf_wrap_syrk_complete, hicma_potrf_wrap_gemm_complete };
    parsec_hook_t **orig_hook[4] = { &params->wrap_potrf, &params->wrap_trsm,
                                     &params->wrap_syrk, &params->wrap_gemm };
    parsec_hook_t **orig_complete[4] = { &params->wrap_potrf_complete, &params->wrap_trsm_complete,
                                         &params->wrap_syrk_complete, &params->wrap_gemm_complete };
    parsec_evaluate_function_t *evals[4] = { NULL, NULL, NULL, NULL };
    parsec_task_class_t *tc[4] = { NULL, NULL, NULL, NULL };
    parsec_task_class_t *tc_send = NULL;
    const uint8_t gpu_types = PARSEC_DEV_CUDA | PARSEC_DEV_HIP;

    if( NULL != eval_gpu ) {
        evals[0] = eval_gpu->potrf;
        evals[1] = eval_gpu->trsm;
        evals[2] = eval_gpu->syrk;
        evals[3] = eval_gpu->gemm;
    }

    for( uint32_t i = 0; i < tp->nb_task_classes; i++ ) {
        parsec_task_class_t *c = (parsec_task_class_t *)tp->task_classes_array[i];
        for( int j = 0; j < 4; j++ ) {
            if( 0 == strcmp(c->name, names[j]) ) tc[j] = c;
        }
        if( 0 == strcmp(c->name, "potrf_dtrsm_send") ) tc_send = c;
    }
    for( int j = 0; j < 4; j++ ) {
        if( NULL == tc[j] ) {
            fprintf(stderr, "hicma_potrf_install_hooks: taskpool %s has no task class %s\n",
                    tp->taskpool_name ? tp->taskpool_name : "?", names[j]);
            return -1;
        }
    }
    if( 0 != hicma_potrf_remember(tp, params, descA) ) {
        fprintf(stderr, "hicma_potrf_install_hooks: more than %d Cholesky taskpools are live\n",
                HICMA_POTRF_MAX_TASKPOOLS);
        return -1;
    }

    for( int j = 0; j < 4; j++ ) {
        __parsec_chore_t *chore = NULL;
        if( nb_gpus > 0 ) {
            chore = hicma_potrf_find_chore(tc[j], gpu_types);
        }
        if( NULL == chore && hnb < mb ) {
            chore = hicma_potrf_find_chore(tc[j], PARSEC_DEV_RECURSIVE);
        }
        if( NULL == chore ) {
            chore = hicma_potrf_find_chore(tc[j], PARSEC_DEV_CPU);
        }
        if( NULL == chore ) {
            chore = (__parsec_chore_t *)tc[j]->incarnations;
        }
        *orig_hook[j] = chore->hook;
        chore->hook = wraps[j];
        if( nb_gpus > 0 && (chore->type & gpu_types) && NULL != evals[j] ) {
            chore->evaluate = evals[j];
        }
        *orig_complete[j] = tc[j]->complete_execution;
        tc[j]->complete_execution = completes[j];
    }

    if( NULL != tc_send && nb_gpus > 0 && NULL != eval_gpu && NULL != eval_gpu->trsm_send ) {
        __parsec_chore_t *chore = hicma_potrf_find_chore(tc_send, gpu_types);
        if( NULL != chore ) {
            chore->evaluate = eval_gpu->trsm_send;
        }
    }

    return 0;
}

/* ============================================================================
 * Memory pools, GPU devices, driver
 * ============================================================================ */

size_t hicma_potrf_hcore_workspace_doubles(int mb, int compmaxrank)
{
    /* From hicma/compute/pzpotrf.c: side-by-side U and V copies, QR taus,
     * the rank x rank products, the new U and V, and the SVD of the 2r x 2r
     * block (R, sigma, superb) */
    size_t r = (size_t)compmaxrank;
    return 2 * (size_t)mb * 2 * r          /* CU, CV */
         + 2 * (size_t)mb                  /* qrtauA, qrtauB */
         + r * r                           /* qrb_aubut, AcolBcolT */
         + 2 * (size_t)mb * 2 * r          /* newU, newV */
         + (2 * r) * (2 * r)               /* svd_rA, _rA */
         + (2 * r)                         /* sigma */
         + (2 * r);                        /* superb */
}

parsec_memory_pool_t *hicma_potrf_pool_new(size_t bytes)
{
    parsec_memory_pool_t *pool = (parsec_memory_pool_t *)malloc(sizeof(parsec_memory_pool_t));
    if( NULL == pool ) {
        fprintf(stderr, "Fatal: out of memory creating a workspace pool\n");
        abort();
    }
    parsec_private_memory_init(pool, bytes);
    return pool;
}

void hicma_potrf_pool_free(parsec_memory_pool_t *pool)
{
    if( NULL == pool ) return;
    parsec_private_memory_fini(pool);
    free(pool);
}

void hicma_potrf_gpu_devices(int *nb, int **dev_index)
{
    *nb = 0;
    *dev_index = NULL;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    hicma_parsec_find_cuda_devices(dev_index, nb);
#endif
}

int hicma_potrf_run(parsec_context_t *parsec,
                    parsec_taskpool_t *tp,
                    void (*destruct)(parsec_taskpool_t *),
                    hicma_parsec_params_t *params,
                    const char *label)
{
    if( 0 == params->rank && params->verbose ) {
        printf(MAG "%s start\n" RESET, label);
        fflush(stdout);
    }
    if( NULL == tp ) {
        if( 0 == params->rank ) {
            fprintf(stderr, RED "%s: the taskpool could not be created (see the message above)\n" RESET, label);
        }
        return -1;
    }

    parsec_context_add_taskpool(parsec, tp);
    parsec_context_start(parsec);
    parsec_context_wait(parsec);

    /* Host buffers replaced by FP64 promotions (runtime-decision mode); a
     * no-op for the other kinds */
    hicma_parsec_host_deferred_free_flush();

    destruct(tp);

    /* Recursive DAGs are not synchronous on taskpool ids */
    if( params->HNB > 0 && params->HNB < params->NB ) {
        parsec_taskpool_sync_ids();
    }
    return 0;
}

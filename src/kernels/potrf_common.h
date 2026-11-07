/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

/**
 * @file potrf_common.h
 * @brief Code shared by the Cholesky wrappers (potrf_L_*_wrapper.c)
 *
 * Every Cholesky variant has its own JDF, but the wrappers around them do the
 * same things: time the four kernels, pick the chore (GPU, recursive or CPU)
 * whose hook is wrapped, create and release the memory pools, discover the
 * GPUs and run the taskpool. That code lives here once. The JDF bodies are not
 * touched by any of it.
 *
 * The timing hooks are generic: they read the task's indices through
 * parsec_task_t::locals, which every Cholesky JDF declares in the same order
 * (potrf(k), trsm(m, k), syrk(k, m), gemm(m, n, k)), and find the parameters
 * of their taskpool in a small registry filled by hicma_potrf_install_hooks.
 */

#ifndef HICMA_POTRF_COMMON_H
#define HICMA_POTRF_COMMON_H

#include "hicma_parsec.h"

#ifdef __cplusplus
extern "C" {
#endif

/** GPU evaluate hooks installed on the GPU chore of each task class; a NULL
 *  entry leaves the generated default (always run on the GPU). trsm_send is
 *  only used by the kinds that have a potrf_dtrsm_send task. */
typedef struct hicma_potrf_evaluate_s {
    parsec_evaluate_function_t *potrf;
    parsec_evaluate_function_t *trsm;
    parsec_evaluate_function_t *trsm_send;
    parsec_evaluate_function_t *syrk;
    parsec_evaluate_function_t *gemm;
} hicma_potrf_evaluate_t;

/**
 * Wrap the potrf/trsm/syrk/gemm task classes of tp with the timing hooks.
 *
 * The wrapped chore is the GPU one when nb_gpus > 0 and the class has one,
 * else the recursive one when hnb < mb and the class has one, else the CPU
 * one (the chore types are read from the generated table, not assumed). The
 * original hooks are kept in params->wrap_* and params->wrap_*_complete.
 * eval_gpu may be NULL; its entries are installed on the GPU chores only.
 *
 * @return 0, or -1 if a task class is missing or too many taskpools are live
 */
int hicma_potrf_install_hooks(parsec_taskpool_t *tp,
                              hicma_parsec_params_t *params,
                              parsec_tiled_matrix_t *descA,
                              int nb_gpus, int hnb, int mb,
                              const hicma_potrf_evaluate_t *eval_gpu);

/** Forget tp (call from the taskpool's Destruct, before parsec_taskpool_free) */
void hicma_potrf_forget(parsec_taskpool_t *tp);

/** Doubles of HCORE workspace one worker needs for the two-step low-rank GEMM
 *  (from hicma/compute/pzpotrf.c) */
size_t hicma_potrf_hcore_workspace_doubles(int mb, int compmaxrank);

/** A per-thread memory pool of elements of `bytes` bytes (malloc + init) */
parsec_memory_pool_t *hicma_potrf_pool_new(size_t bytes);

/** Release a pool made by hicma_potrf_pool_new (NULL is fine) */
void hicma_potrf_pool_free(parsec_memory_pool_t *pool);

/** Number and PaRSEC indices of the CUDA/HIP devices (nb = 0, dev_index = NULL
 *  without GPU support). dev_index is free()d by the caller. */
void hicma_potrf_gpu_devices(int *nb, int **dev_index);

/**
 * Run a Cholesky taskpool: print the start message, add it to the context,
 * start, wait, flush the host buffers that FP64 promotions replaced, destruct,
 * and resynchronize the taskpool ids if recursive DAGs may have run.
 *
 * @return 0, or -1 if tp is NULL (the New() of the kind refused the parameters)
 */
int hicma_potrf_run(parsec_context_t *parsec,
                    parsec_taskpool_t *tp,
                    void (*destruct)(parsec_taskpool_t *),
                    hicma_parsec_params_t *params,
                    const char *label);

#ifdef __cplusplus
}
#endif

#endif /* HICMA_POTRF_COMMON_H */

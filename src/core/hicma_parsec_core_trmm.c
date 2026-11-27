/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

/*
 * Tile kernels of the mixed-precision TRMM (Left, Lower, NoTrans):
 * B = alpha * A * B, A an MT x MT lower triangular tile matrix.
 *
 * Every tile has a precision decision (params->decisions for A,
 * params->decisionsB for B) and is stored in the precision returned by
 * hicma_parsec_trmm_storage_precision(): the decision itself, or FP32 for an
 * FP16 tile of B with --accumulation_fp32 (A's FP16 tiles stay in FP16).
 *
 *  - TRMM on B(i,n) computes in the precision of B(i,n): FP64 or FP32 (there
 *    is no FP16 TRMM, an FP16 tile is computed in FP32 and converted back to
 *    its storage precision). On the GPU it runs out of place through a
 *    workspace, as a GEMM with tril(A) in FP32 for tiles up to 1024.
 *  - GEMM C(i,n) += A(i,k) * B(k,n) computes in the precision of C(i,n). An
 *    FP16 GEMM accumulates in FP16, or with --accumulation_fp32 takes FP16
 *    operands and accumulates in FP32 into the FP32 C (on the CPU, which has
 *    no such GEMM, it is an FP32 GEMM). A CPU without FP16 (HAVE_HP_CPU)
 *    computes an FP16 GEMM in FP32 too, converting C to FP32 and back.
 *  - The GEMMs read B(k,n) in place, or from an operand copy in its
 *    precision decision (copy_b): the GEMMs on other ranks, and all of them
 *    when --accumulation_fp32 stores the FP16 tile in FP32, which is then
 *    converted to FP16 once instead of in every GEMM that reads it (see
 *    read_B in trmm_mp_LLN.jdf).
 *
 * Operands in another precision are converted into a workspace buffer first,
 * with stochastic rounding for down-conversions if enabled (except the CPU's
 * FP16 conversions, which round to nearest, in software without HAVE_HP_CPU).
 * Every conversion keeps the operand's leading dimension, so the BLAS calls
 * use the tile's.
 *
 * The tasks have CPU and GPU versions, and PaRSEC runs FP16 tiles on the CPU
 * too when it picks the CPU version (no GPU enabled for the taskpool, or
 * device_load_balance_allow_cpu), so the CPU kernels handle every precision.
 */

#include "hicma_parsec.h"

static void hicma_parsec_trmm_bad_precision(const char *where, uint16_t from, uint16_t to)
{
    fprintf(stderr, "%s: no conversion from precision %u to %u on this device\n",
            where, (unsigned)from, (unsigned)to);
    exit(1);
}

/* ======================================================================
 * CPU
 * ====================================================================== */

typedef struct {
    parsec_memory_pool_t *pool;
    void *buf;
} hicma_parsec_trmm_tmp_t;

/* Convert X (rows x cols, leading dimension ld, precision from) into Y
 * (precision to, same leading dimension); from != to. Stochastic rounding
 * applies to FP64 -> FP32 on the CPU, as before. */
static void hicma_parsec_trmm_cpu_convert_into(const void *X, uint16_t from, uint16_t to,
        int rows, int cols, int ld, int stochastic_rounding, uint64_t seed, void *Y)
{
    if( !((DENSE_DP == from || DENSE_SP == from || DENSE_HP == from) &&
          (DENSE_DP == to || DENSE_SP == to || DENSE_HP == to)) || from == to ) {
        hicma_parsec_trmm_bad_precision( "TRMM CPU", from, to );
    }
    const int sr = stochastic_rounding && DENSE_DP == from && DENSE_SP == to;
    HICMA_OMC( omc_convert_2d(hicma_parsec_omc_dtype(from), X, ld, hicma_parsec_omc_dtype(to), Y, ld,
                              rows, cols, HICMA_OMC_SR(sr, seed)) );
}

/* Return X (rows x cols, leading dimension ld, precision from) in precision
 * to, converting into a buffer from the pool of precision to if needed. */
static void *hicma_parsec_trmm_cpu_convert(void *X, uint16_t from, uint16_t to,
        int rows, int cols, int ld, int stochastic_rounding, uint64_t seed,
        parsec_memory_pool_t *p_work_full_dp,
        parsec_memory_pool_t *p_work_full_sp,
        parsec_memory_pool_t *p_work_full_hp,
        hicma_parsec_trmm_tmp_t *tmp)
{
    tmp->pool = NULL;
    tmp->buf = NULL;
    if( from == to ) return X;

    tmp->pool = (DENSE_DP == to) ? p_work_full_dp : (DENSE_SP == to) ? p_work_full_sp : p_work_full_hp;
    tmp->buf = parsec_private_memory_pop( tmp->pool );
    hicma_parsec_trmm_cpu_convert_into( X, from, to, rows, cols, ld, stochastic_rounding, seed, tmp->buf );
    return tmp->buf;
}

static void hicma_parsec_trmm_cpu_release(hicma_parsec_trmm_tmp_t *tmp)
{
    if( NULL != tmp->buf ) {
        parsec_private_memory_push( tmp->pool, tmp->buf );
    }
}

static size_t hicma_parsec_trmm_precision_size(uint16_t precision)
{
    return (DENSE_DP == precision) ? sizeof(double) : (DENSE_SP == precision) ? sizeof(float) : 2;
}

void hicma_parsec_trmm_core_copy_b_cpu(
        hicma_parsec_params_t *params_tlr,
        void *B, void *Bop, int rows, int cols, int ld,
        int k, int n)
{
    int NT = params_tlr->NT;
    uint16_t from = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisionsB, n*NT+k);
    uint16_t to = params_tlr->decisionsB[n*NT+k];

    if( from == to ) {
        memcpy( Bop, B, (size_t)ld * cols * hicma_parsec_trmm_precision_size(to) );
    } else {
        hicma_parsec_trmm_cpu_convert_into( B, from, to, rows, cols, ld,
                params_tlr->enable_stochastic_rounding, hicma_parsec_sr_seed(k, n), Bop );
    }
}

void hicma_parsec_trmm_core_trmm_lln_cpu(
        hicma_parsec_params_t *params_tlr,
        parsec_execution_stream_t *es,
        parsec_memory_pool_t *p_work_full_dp,
        parsec_memory_pool_t *p_work_full_sp,
        parsec_memory_pool_t *p_work_full_hp,
        dplasma_enum_t side,
        dplasma_enum_t uplo,
        dplasma_enum_t trans,
        dplasma_enum_t diag,
        int mb, int nb,
        double lalpha, void *A, int lda,
        void *B, int ldb,
        int m, int n)
{
    (void)es;
    int NT = params_tlr->NT;
    int sr = params_tlr->enable_stochastic_rounding;
    uint64_t seedA = hicma_parsec_sr_seed((NT-1)-m, (NT-1)-m), seedB = hicma_parsec_sr_seed((NT-1)-m, n);
    uint16_t storageA = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisions, ((NT-1)-m)*NT+(NT-1)-m);
    uint16_t decisionB = params_tlr->decisionsB[n*NT+(NT-1)-m];
    uint16_t storageB = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisionsB, n*NT+(NT-1)-m);
    hicma_parsec_trmm_tmp_t tA, tB;

    if( DENSE_DP == decisionB ) {
        void *A_use = hicma_parsec_trmm_cpu_convert( A, storageA, DENSE_DP, mb, mb, lda, sr, seedA,
                p_work_full_dp, p_work_full_sp, p_work_full_hp, &tA );
        CORE_dtrmm( side, uplo, trans, diag, mb, nb,
                    lalpha, A_use, lda,
                            B,     ldb );
        hicma_parsec_trmm_cpu_release( &tA );
    } else if( DENSE_SP == decisionB || DENSE_HP == decisionB ) {
        /* No FP16 TRMM: compute an FP16 tile in FP32 */
        void *A_use = hicma_parsec_trmm_cpu_convert( A, storageA, DENSE_SP, mb, mb, lda, sr, seedA,
                p_work_full_dp, p_work_full_sp, p_work_full_hp, &tA );
        void *B_use = hicma_parsec_trmm_cpu_convert( B, storageB, DENSE_SP, mb, nb, ldb, sr, seedB,
                p_work_full_dp, p_work_full_sp, p_work_full_hp, &tB );
        CORE_strmm( side, uplo, trans, diag, mb, nb,
                    (float)lalpha, A_use, lda,
                                   B_use, ldb );
        if( B_use != B ) {
            hicma_parsec_trmm_cpu_convert_into( B_use, DENSE_SP, storageB, mb, nb, ldb, sr, seedB, B );
        }
        hicma_parsec_trmm_cpu_release( &tA );
        hicma_parsec_trmm_cpu_release( &tB );
    } else {
        hicma_parsec_trmm_bad_precision( "hicma_parsec_trmm_core_trmm_lln_cpu", decisionB, decisionB );
    }
}

void hicma_parsec_trmm_core_gemm_lln_cpu(
        hicma_parsec_params_t *params_tlr,
        parsec_execution_stream_t *es,
        parsec_memory_pool_t *p_work_full_dp,
        parsec_memory_pool_t *p_work_full_sp,
        parsec_memory_pool_t *p_work_full_hp,
        dplasma_enum_t transA,
        dplasma_enum_t transB,
        int mb, int nb, int kb,
        double lalpha, void *A, int lda,
        void *B, int ldb,
        double lbeta, void *C, int ldc,
        int m, int n, int k)
{
    (void)es;
    int NT = params_tlr->NT;
    int sr = params_tlr->enable_stochastic_rounding;
    uint64_t seedA = hicma_parsec_sr_seed((NT-1)-m, k), seedB = hicma_parsec_sr_seed(k, n),
             seedC = hicma_parsec_sr_seed((NT-1)-m, n);
    uint16_t storageA = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisions, k*NT+(NT-1)-m);
    /* B is the operand copy of B(k,n), in its precision decision */
    uint16_t operandB = params_tlr->decisionsB[n*NT+k];
    uint16_t storageC = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisionsB, n*NT+(NT-1)-m);
    hicma_parsec_trmm_tmp_t tA, tB, tC;

    /* A is mb x kb, B is kb x nb, C is mb x nb; compute in C's storage
     * precision (FP32 for an FP16 C with --accumulation_fp32), or in FP32
     * for an FP16 C on a CPU without FP16 */
    uint16_t compute = storageC;
#if !HAVE_HP_CPU
    if( DENSE_HP == compute ) {
        compute = DENSE_SP;
    }
#endif
    void *A_use = hicma_parsec_trmm_cpu_convert( A, storageA, compute, mb, kb, lda, sr, seedA,
            p_work_full_dp, p_work_full_sp, p_work_full_hp, &tA );
    void *B_use = hicma_parsec_trmm_cpu_convert( B, operandB, compute, kb, nb, ldb, sr, seedB,
            p_work_full_dp, p_work_full_sp, p_work_full_hp, &tB );
    void *C_use = hicma_parsec_trmm_cpu_convert( C, storageC, compute, mb, nb, ldc, sr, seedC,
            p_work_full_dp, p_work_full_sp, p_work_full_hp, &tC );

    if( DENSE_DP == compute ) {
        CORE_dgemm( transA, transB, mb, nb, kb,
                    lalpha, A_use, lda,
                            B_use, ldb,
                    lbeta,  C_use, ldc );
    } else if( DENSE_SP == compute ) {
        CORE_sgemm( transA, transB, mb, nb, kb,
                    (float)lalpha, A_use, lda,
                                   B_use, ldb,
                    (float)lbeta,  C_use, ldc );
    }
#if HAVE_HP_CPU
    else if( DENSE_HP == compute ) {
        fjcblas_gemm_r16( CblasColMajor, transA, transB,
                          mb, nb, kb,
                          (__fp16)lalpha, A_use, lda,
                                          B_use, ldb,
                          (__fp16)lbeta,  C_use, ldc );
    }
#endif
    else {
        hicma_parsec_trmm_bad_precision( "hicma_parsec_trmm_core_gemm_lln_cpu", compute, compute );
    }

    if( C_use != C ) {
        hicma_parsec_trmm_cpu_convert_into( C_use, compute, storageC, mb, nb, ldc, sr, seedC, C );
    }

    hicma_parsec_trmm_cpu_release( &tA );
    hicma_parsec_trmm_cpu_release( &tB );
    hicma_parsec_trmm_cpu_release( &tC );
}


#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

/* ======================================================================
 * GPU
 * ====================================================================== */

/* Return X (rows x cols, leading dimension ld, precision from) in precision
 * to, converting into buf on the stream if needed. Stochastic rounding
 * applies to the narrowing conversions. */
static void *hicma_parsec_trmm_gpu_convert(void *X, uint16_t from, uint16_t to,
        int rows, int cols, int ld, int stochastic_rounding, uint64_t seed,
        void *buf, cudaStream_t stream)
{
    if( from == to ) return X;
    if( !((DENSE_DP == from || DENSE_SP == from || DENSE_HP == from) &&
          (DENSE_DP == to || DENSE_SP == to || DENSE_HP == to)) ) {
        hicma_parsec_trmm_bad_precision( "TRMM GPU", from, to );
        return NULL;
    }
    const omc_dtype_t st = hicma_parsec_omc_dtype(from), dt = hicma_parsec_omc_dtype(to);
    const int sr = stochastic_rounding && omc_dtype_size(dt) < omc_dtype_size(st);
    HICMA_OMC( omc_gpu_convert_2d(st, X, ld, dt, buf, ld, rows, cols, HICMA_OMC_SR(sr, seed), stream) );
    return buf;
}

/* Copy X (ld x cols, precision precision) into Y on the stream */
static void hicma_parsec_trmm_gpu_copy(void *Y, void *X, uint16_t precision,
        int ld, int cols, cudaStream_t stream)
{
    const omc_dtype_t t = hicma_parsec_omc_dtype(precision);
    HICMA_OMC( omc_gpu_convert(t, X, t, Y, (size_t)ld * cols, NULL, stream) );
}

void hicma_parsec_trmm_core_copy_b_gpu(
        hicma_parsec_params_t *params_tlr,
        parsec_potrf_workspace_t *ws_gpu,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        void *B, void *Bop, int rows, int cols, int ld,
        int k, int n)
{
    (void)ws_gpu; (void)cuda_device; (void)gpu_task;
    int NT = params_tlr->NT;
    uint16_t from = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisionsB, n*NT+k);
    uint16_t to = params_tlr->decisionsB[n*NT+k];

    if( from == to ) {
        hicma_parsec_trmm_gpu_copy( Bop, B, to, ld, cols, cuda_stream->cuda_stream );
    } else {
        hicma_parsec_trmm_gpu_convert( B, from, to, rows, cols, ld,
                params_tlr->enable_stochastic_rounding, hicma_parsec_sr_seed(k, n), Bop, cuda_stream->cuda_stream );
    }
}

/* Largest diagonal tile computed in FP32 as a GEMM with the lower triangle of
 * A rather than with cuBLAS trmm (measured faster up to 1024 on H100 and V100,
 * slower from 1536 on the H100) */
#define HICMA_PARSEC_TRMM_SGEMM_MAX_NB 1024

void hicma_parsec_trmm_core_trmm_lln_gpu(
        hicma_parsec_params_t *params_tlr,
        parsec_potrf_workspace_t *ws_gpu,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        dplasma_enum_t side,
        dplasma_enum_t uplo,
        dplasma_enum_t trans,
        dplasma_enum_t diag,
        int mb, int nb,
        double lalpha, void *A, int lda,
        void *B, int ldb,
        int m, int n)
{
    (void)gpu_task;
    int NT = params_tlr->NT;
    int sr = params_tlr->enable_stochastic_rounding;
    uint64_t seedA = hicma_parsec_sr_seed((NT-1)-m, (NT-1)-m), seedB = hicma_parsec_sr_seed((NT-1)-m, n);
    uint16_t storageA = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisions, ((NT-1)-m)*NT+(NT-1)-m);
    uint16_t decisionB = params_tlr->decisionsB[n*NT+(NT-1)-m];
    uint16_t storageB = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisionsB, n*NT+(NT-1)-m);
    cudaStream_t stream = cuda_stream->cuda_stream;
    cublasStatus_t status;
    uint16_t prec;

    /* Compute in FP64 or FP32; there is no FP16 TRMM, an FP16 tile is computed in FP32 */
    if( DENSE_DP == decisionB ) {
        prec = DENSE_DP;
    } else if( DENSE_SP == decisionB || DENSE_HP == decisionB ) {
        prec = DENSE_SP;
    } else {
        hicma_parsec_trmm_bad_precision( "hicma_parsec_trmm_core_trmm_lln_gpu", decisionB, decisionB );
        return;
    }

    parsec_potrf_stream_workspace_t *stream_found = lookup_gpu_workspace(cuda_device, cuda_stream, ws_gpu);
    cublasHandle_t handle = stream_found->handle_cublas;

    /* cuBLAS trmm is several times slower in place than out of place: read B
     * from a workspace (copied, or converted to the compute precision) and
     * write the product into B, or into another workspace to convert back */
    void *B_in, *out;
    if( storageB == prec ) {
        hicma_parsec_trmm_gpu_copy( stream_found->gpu_buffer_B, B, prec, ldb, nb, stream );
        B_in = stream_found->gpu_buffer_B;
        out = B;
    } else {
        B_in = hicma_parsec_trmm_gpu_convert( B, storageB, prec, mb, nb, ldb, sr, seedB,
                stream_found->gpu_buffer_B, stream );
        out = stream_found->gpu_buffer_C;
    }

    if( DENSE_DP == prec ) {
        void *A_use = hicma_parsec_trmm_gpu_convert( A, storageA, DENSE_DP, mb, mb, lda, sr, seedA,
                stream_found->gpu_buffer_A, stream );
        double alpha = lalpha;
        status = cublasDtrmm( handle,
                dplasma_cublas_side(side), dplasma_cublas_fill(uplo),
                dplasma_cublas_op(trans), dplasma_cublas_diag(diag),
                mb, nb,
                &alpha, A_use, lda,
                        B_in,  ldb,
                        out,   ldb );
        PARSEC_CUDA_CHECK_ERROR( "cublasDtrmm", status, {exit(1);} );
    } else if( mb <= HICMA_PARSEC_TRMM_SGEMM_MAX_NB && dplasmaLeft == side
               && dplasmaLower == uplo && dplasmaNoTrans == trans ) {
        /* GEMM with tril(A): the upper triangle zeroed, and 1 on the diagonal if unit */
        void *L = hicma_parsec_trmm_gpu_convert( A, storageA, DENSE_SP, mb, mb, lda, sr, seedA,
                stream_found->gpu_buffer_A, stream );
        if( L == A ) {
            HICMA_OMC( omc_gpu_convert(OMC_FP32, A, OMC_FP32, stream_found->gpu_buffer_A, (size_t)lda * mb, NULL, stream) );
            L = stream_found->gpu_buffer_A;
        }
        hicma_parsec_tril_mask_float_GPU( mb, (float *)L, lda, dplasmaUnit == diag, stream );
        float alpha = (float)lalpha, beta = 0.0f;
        status = cublasSgemm( handle,
                dplasma_cublas_op(dplasmaNoTrans), dplasma_cublas_op(dplasmaNoTrans),
                mb, nb, mb,
                &alpha, L,    lda,
                        B_in, ldb,
                &beta,  out,  ldb );
        PARSEC_CUDA_CHECK_ERROR( "TRMM diagonal tile SGEMM", status, {exit(1);} );
    } else {
        void *A_use = hicma_parsec_trmm_gpu_convert( A, storageA, DENSE_SP, mb, mb, lda, sr, seedA,
                stream_found->gpu_buffer_A, stream );
        float alpha = (float)lalpha;
        status = cublasStrmm( handle,
                dplasma_cublas_side(side), dplasma_cublas_fill(uplo),
                dplasma_cublas_op(trans), dplasma_cublas_diag(diag),
                mb, nb,
                &alpha, A_use, lda,
                        B_in,  ldb,
                        out,   ldb );
        PARSEC_CUDA_CHECK_ERROR( "cublasStrmm", status, {exit(1);} );
    }

    /* Back to the storage precision of B */
    if( out != B ) {
        hicma_parsec_trmm_gpu_convert( out, prec, storageB, mb, nb, ldb, sr, seedB, B, stream );
    }
}

void hicma_parsec_trmm_core_gemm_lln_gpu(
        hicma_parsec_params_t *params_tlr,
        parsec_potrf_workspace_t *ws_gpu,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        dplasma_enum_t transA,
        dplasma_enum_t transB,
        int mb, int nb, int kb,
        double lalpha, void *A, int lda,
        void *B, int ldb,
        double lbeta, void *C, int ldc,
        int m, int n, int k)
{
    (void)gpu_task;
    int NT = params_tlr->NT;
    int sr = params_tlr->enable_stochastic_rounding;
    uint64_t seedA = hicma_parsec_sr_seed((NT-1)-m, k), seedB = hicma_parsec_sr_seed(k, n);
    uint16_t storageA = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisions, k*NT+(NT-1)-m);
    /* B is the operand copy of B(k,n), in its precision decision */
    uint16_t operandB = params_tlr->decisionsB[n*NT+k];
    uint16_t decisionC = params_tlr->decisionsB[n*NT+(NT-1)-m];
    uint16_t storageC = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisionsB, n*NT+(NT-1)-m);
    cudaStream_t stream = cuda_stream->cuda_stream;
    cublasStatus_t status;

    parsec_potrf_stream_workspace_t *stream_found = lookup_gpu_workspace(cuda_device, cuda_stream, ws_gpu);
    cublasHandle_t handle = stream_found->handle_cublas;

    /* A is mb x kb, B is kb x nb, C is mb x nb; the operands take the
     * precision of C's decision (FP16 also with --accumulation_fp32) */
    void *A_use = hicma_parsec_trmm_gpu_convert( A, storageA, decisionC, mb, kb, lda, sr, seedA,
            stream_found->gpu_buffer_A, stream );
    void *B_use = hicma_parsec_trmm_gpu_convert( B, operandB, decisionC, kb, nb, ldb, sr, seedB,
            stream_found->gpu_buffer_B, stream );

    if( DENSE_DP == decisionC ) {
        double alpha = lalpha, beta = lbeta;
        status = hicma_parsec_dgemm_gpu( handle, dplasma_cublas_op(transA), dplasma_cublas_op(transB),
                mb, nb, kb,
                &alpha, A_use, lda,
                        B_use, ldb,
                &beta,  C,     ldc );
        PARSEC_CUDA_CHECK_ERROR( "hicma_parsec_dgemm_gpu", status, {exit(1);} );
    } else if( DENSE_SP == decisionC ) {
        float alpha = (float)lalpha, beta = (float)lbeta;
        status = cublasGemmEx( handle, dplasma_cublas_op(transA), dplasma_cublas_op(transB),
                mb, nb, kb,
                &alpha, A_use, CUDA_R_32F, lda,
                        B_use, CUDA_R_32F, ldb,
                &beta,  C,     CUDA_R_32F, ldc,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT );
        PARSEC_CUDA_CHECK_ERROR( "TRMM GEMM FP32", status, {exit(1);} );
    } else if( DENSE_HP == decisionC && DENSE_SP == storageC ) {
        /* --accumulation_fp32: FP16 operands, FP32 accumulation into FP32 C */
        float alpha = (float)lalpha, beta = (float)lbeta;
        status = cublasGemmEx( handle, dplasma_cublas_op(transA), dplasma_cublas_op(transB),
                mb, nb, kb,
                &alpha, A_use, CUDA_R_16F, lda,
                        B_use, CUDA_R_16F, ldb,
                &beta,  C,     CUDA_R_32F, ldc,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT );
        PARSEC_CUDA_CHECK_ERROR( "TRMM GEMM FP16, FP32 accumulation", status, {exit(1);} );
    } else if( DENSE_HP == decisionC && MASK_TF16_A16_B16_C16_OP32 == HICMA_FP16_GEMM(params_tlr) ) {
        /* --tensor_gemm 0x80: FP16 operands and C, FP32 compute */
        float alpha = (float)lalpha, beta = (float)lbeta;
        status = cublasGemmEx( handle, dplasma_cublas_op(transA), dplasma_cublas_op(transB),
                mb, nb, kb,
                &alpha, A_use, CUDA_R_16F, lda,
                        B_use, CUDA_R_16F, ldb,
                &beta,  C,     CUDA_R_16F, ldc,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT );
        PARSEC_CUDA_CHECK_ERROR( "TRMM GEMM FP16, FP32 compute", status, {exit(1);} );
    } else if( DENSE_HP == decisionC ) {
        char alpha[16], beta[16];
        hicma_parsec_fp32_to_fp16_host( (float)lalpha, &alpha[0] );
        hicma_parsec_fp32_to_fp16_host( (float)lbeta, &beta[0] );
        status = cublasGemmEx( handle, dplasma_cublas_op(transA), dplasma_cublas_op(transB),
                mb, nb, kb,
                &alpha[0], A_use, CUDA_R_16F, lda,
                           B_use, CUDA_R_16F, ldb,
                &beta[0],  C,     CUDA_R_16F, ldc,
                CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT );
        PARSEC_CUDA_CHECK_ERROR( "TRMM GEMM FP16", status, {exit(1);} );
    } else {
        hicma_parsec_trmm_bad_precision( "hicma_parsec_trmm_core_gemm_lln_gpu", decisionC, decisionC );
    }
}

#endif

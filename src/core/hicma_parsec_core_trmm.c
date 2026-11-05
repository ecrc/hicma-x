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
 * FP16 decision with --accumulation_fp32.
 *
 *  - TRMM on B(i,n) computes in the precision of B(i,n): FP64 or FP32 (there
 *    is no FP16 TRMM, an FP16 tile is computed in FP32 and converted back).
 *  - GEMM C(i,n) += A(i,k) * B(k,n) computes in the precision of C(i,n). An
 *    FP16 GEMM accumulates in FP16, or in FP32 into an FP32 C with
 *    --accumulation_fp32.
 *
 * Operands in another precision are converted into a workspace buffer first,
 * with stochastic rounding for down-conversions if enabled. Every conversion
 * keeps the operand's leading dimension, so the BLAS calls use the tile's.
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

/* Return X (rows x cols, leading dimension ld, precision from) in precision
 * to, converting into a buffer from the pool of precision to if needed. */
static void *hicma_parsec_trmm_cpu_convert(void *X, uint16_t from, uint16_t to,
        int rows, int cols, int ld, int stochastic_rounding,
        parsec_memory_pool_t *p_work_full_dp,
        parsec_memory_pool_t *p_work_full_sp,
        parsec_memory_pool_t *p_work_full_hp,
        hicma_parsec_trmm_tmp_t *tmp)
{
    tmp->pool = NULL;
    tmp->buf = NULL;
    if( from == to ) return X;

    if( DENSE_DP == from && DENSE_SP == to ) {
        float *Y = parsec_private_memory_pop( p_work_full_sp );
        if( stochastic_rounding ) {
            double2float_round_CPU( rows, cols, (const double *)X, ld, Y, ld );
        } else {
            const double *D = (const double *)X;
            for( int j = 0; j < cols; j++ )
                for( int i = 0; i < rows; i++ )
                    Y[j*ld+i] = (float)D[j*ld+i];
        }
        tmp->pool = p_work_full_sp; tmp->buf = Y;
        return Y;
    }

    if( DENSE_SP == from && DENSE_DP == to ) {
        double *Y = parsec_private_memory_pop( p_work_full_dp );
        const float *F = (const float *)X;
        for( int j = 0; j < cols; j++ )
            for( int i = 0; i < rows; i++ )
                Y[j*ld+i] = (double)F[j*ld+i];
        tmp->pool = p_work_full_dp; tmp->buf = Y;
        return Y;
    }

#if HAVE_HP_CPU
    /* The binary converters take contiguous data: convert ld x cols */
    if( DENSE_HP == to ) {
        void *Y = parsec_private_memory_pop( p_work_full_hp );
        if( DENSE_DP == from ) convert_d2h_binary_CPU( Y, (double *)X, ld, cols );
        else                   convert_s2h_binary_CPU( Y, (float *)X, ld, cols );
        tmp->pool = p_work_full_hp; tmp->buf = Y;
        return Y;
    }
    if( DENSE_HP == from && DENSE_SP == to ) {
        void *Y = parsec_private_memory_pop( p_work_full_sp );
        convert_h2s_binary_CPU( Y, (__fp16 *)X, ld, cols );
        tmp->pool = p_work_full_sp; tmp->buf = Y;
        return Y;
    }
    if( DENSE_HP == from && DENSE_DP == to ) {
        void *Y = parsec_private_memory_pop( p_work_full_dp );
        convert_h2d_binary_CPU( Y, (__fp16 *)X, ld, cols );
        tmp->pool = p_work_full_dp; tmp->buf = Y;
        return Y;
    }
#endif

    hicma_parsec_trmm_bad_precision( "TRMM CPU", from, to );
    return NULL;
}

static void hicma_parsec_trmm_cpu_release(hicma_parsec_trmm_tmp_t *tmp)
{
    if( NULL != tmp->buf ) {
        parsec_private_memory_push( tmp->pool, tmp->buf );
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
    uint16_t storageA = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisions[((NT-1)-m)*NT+(NT-1)-m]);
    uint16_t decisionB = params_tlr->decisionsB[n*NT+(NT-1)-m];
    uint16_t storageB = hicma_parsec_trmm_storage_precision(params_tlr, decisionB);
    hicma_parsec_trmm_tmp_t tA, tB;

    if( DENSE_DP == decisionB ) {
        void *A_use = hicma_parsec_trmm_cpu_convert( A, storageA, DENSE_DP, mb, mb, lda, sr,
                p_work_full_dp, p_work_full_sp, p_work_full_hp, &tA );
        CORE_dtrmm( side, uplo, trans, diag, mb, nb,
                    lalpha, A_use, lda,
                            B,     ldb );
        hicma_parsec_trmm_cpu_release( &tA );
    } else if( DENSE_SP == decisionB || DENSE_HP == decisionB ) {
        /* No FP16 TRMM: compute an FP16 tile in FP32 */
        void *A_use = hicma_parsec_trmm_cpu_convert( A, storageA, DENSE_SP, mb, mb, lda, sr,
                p_work_full_dp, p_work_full_sp, p_work_full_hp, &tA );
        void *B_use = hicma_parsec_trmm_cpu_convert( B, storageB, DENSE_SP, mb, nb, ldb, sr,
                p_work_full_dp, p_work_full_sp, p_work_full_hp, &tB );
        CORE_strmm( side, uplo, trans, diag, mb, nb,
                    (float)lalpha, A_use, lda,
                                   B_use, ldb );
#if HAVE_HP_CPU
        if( B_use != B ) {
            convert_s2h_binary_CPU( (__fp16 *)B, (float *)B_use, ldb, nb );
        }
#endif
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
    uint16_t storageA = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisions[k*NT+(NT-1)-m]);
    uint16_t storageB = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisionsB[n*NT+k]);
    uint16_t storageC = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisionsB[n*NT+(NT-1)-m]);
    hicma_parsec_trmm_tmp_t tA, tB;

    /* A is mb x kb, B is kb x nb, C is mb x nb; compute in C's storage precision */
    void *A_use = hicma_parsec_trmm_cpu_convert( A, storageA, storageC, mb, kb, lda, sr,
            p_work_full_dp, p_work_full_sp, p_work_full_hp, &tA );
    void *B_use = hicma_parsec_trmm_cpu_convert( B, storageB, storageC, kb, nb, ldb, sr,
            p_work_full_dp, p_work_full_sp, p_work_full_hp, &tB );

    if( DENSE_DP == storageC ) {
        CORE_dgemm( transA, transB, mb, nb, kb,
                    lalpha, A_use, lda,
                            B_use, ldb,
                    lbeta,  C,     ldc );
    } else if( DENSE_SP == storageC ) {
        CORE_sgemm( transA, transB, mb, nb, kb,
                    (float)lalpha, A_use, lda,
                                   B_use, ldb,
                    (float)lbeta,  C,     ldc );
    }
#if HAVE_HP_CPU
    else if( DENSE_HP == storageC ) {
        fjcblas_gemm_r16( CblasColMajor, transA, transB,
                          mb, nb, kb,
                          (__fp16)lalpha, A_use, lda,
                                          B_use, ldb,
                          (__fp16)lbeta,  C,     ldc );
    }
#endif
    else {
        hicma_parsec_trmm_bad_precision( "hicma_parsec_trmm_core_gemm_lln_cpu", storageC, storageC );
    }

    hicma_parsec_trmm_cpu_release( &tA );
    hicma_parsec_trmm_cpu_release( &tB );
}


#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

/* ======================================================================
 * GPU
 * ====================================================================== */

/* Return X (rows x cols, leading dimension ld, precision from) in precision
 * to, converting into buf on the stream if needed. */
static void *hicma_parsec_trmm_gpu_convert(void *X, uint16_t from, uint16_t to,
        int rows, int cols, int ld, int stochastic_rounding,
        void *buf, cudaStream_t stream)
{
    if( from == to ) return X;

    if( DENSE_DP == to ) {
        if( DENSE_SP == from ) {
            float2double_GPU( rows, cols, (const float *)X, ld, (double *)buf, ld, stream );
            return buf;
        }
        if( DENSE_HP == from ) {
            half2double_GPU( rows, cols, X, ld, (double *)buf, ld, stream );
            return buf;
        }
    } else if( DENSE_SP == to ) {
        if( DENSE_DP == from ) {
            if( stochastic_rounding )
                double2float_round_GPU( rows, cols, (const double *)X, ld, (float *)buf, ld, stream );
            else
                double2float_GPU( rows, cols, (const double *)X, ld, (float *)buf, ld, stream );
            return buf;
        }
        if( DENSE_HP == from ) {
            half2float_GPU( rows, cols, X, ld, (float *)buf, ld, stream );
            return buf;
        }
    } else if( DENSE_HP == to ) {
        if( DENSE_DP == from ) {
            if( stochastic_rounding )
                double2half_round_GPU( rows, cols, (const double *)X, ld, buf, ld, stream );
            else
                double2half_GPU( rows, cols, (const double *)X, ld, buf, ld, stream );
            return buf;
        }
        if( DENSE_SP == from ) {
            if( stochastic_rounding )
                float2half_round_GPU( rows, cols, (const float *)X, ld, buf, ld, stream );
            else
                float2half_GPU( rows, cols, (const float *)X, ld, buf, ld, stream );
            return buf;
        }
    }

    hicma_parsec_trmm_bad_precision( "TRMM GPU", from, to );
    return NULL;
}

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
    uint16_t storageA = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisions[((NT-1)-m)*NT+(NT-1)-m]);
    uint16_t decisionB = params_tlr->decisionsB[n*NT+(NT-1)-m];
    uint16_t storageB = hicma_parsec_trmm_storage_precision(params_tlr, decisionB);
    cudaStream_t stream = cuda_stream->cuda_stream;
    cublasStatus_t status;

    parsec_potrf_stream_workspace_t *stream_found = lookup_gpu_workspace(cuda_device, cuda_stream, ws_gpu);
    cublasHandle_t handle = stream_found->handle_cublas;

    if( DENSE_DP == decisionB ) {
        void *A_use = hicma_parsec_trmm_gpu_convert( A, storageA, DENSE_DP, mb, mb, lda, sr,
                stream_found->gpu_buffer_A, stream );
        double alpha = lalpha;
        status = cublasDtrmm( handle,
                dplasma_cublas_side(side), dplasma_cublas_fill(uplo),
                dplasma_cublas_op(trans), dplasma_cublas_diag(diag),
                mb, nb,
                &alpha, A_use, lda,
                        B,     ldb,
                        B,     ldb );
        PARSEC_CUDA_CHECK_ERROR( "cublasDtrmm", status, {exit(1);} );
    } else if( DENSE_SP == decisionB || DENSE_HP == decisionB ) {
        /* No FP16 TRMM: compute an FP16 tile in FP32 */
        void *A_use = hicma_parsec_trmm_gpu_convert( A, storageA, DENSE_SP, mb, mb, lda, sr,
                stream_found->gpu_buffer_A, stream );
        void *B_use = hicma_parsec_trmm_gpu_convert( B, storageB, DENSE_SP, mb, nb, ldb, sr,
                stream_found->gpu_buffer_B, stream );
        float alpha = (float)lalpha;
        status = cublasStrmm( handle,
                dplasma_cublas_side(side), dplasma_cublas_fill(uplo),
                dplasma_cublas_op(trans), dplasma_cublas_diag(diag),
                mb, nb,
                &alpha, A_use, lda,
                        B_use, ldb,
                        B_use, ldb );
        PARSEC_CUDA_CHECK_ERROR( "cublasStrmm", status, {exit(1);} );

        /* Back to the storage precision of B */
        if( B_use != B ) {
            hicma_parsec_trmm_gpu_convert( B_use, DENSE_SP, storageB, mb, nb, ldb, sr, B, stream );
        }
    } else {
        hicma_parsec_trmm_bad_precision( "hicma_parsec_trmm_core_trmm_lln_gpu", decisionB, decisionB );
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
    uint16_t storageA = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisions[k*NT+(NT-1)-m]);
    uint16_t storageB = hicma_parsec_trmm_storage_precision(params_tlr, params_tlr->decisionsB[n*NT+k]);
    uint16_t decisionC = params_tlr->decisionsB[n*NT+(NT-1)-m];
    uint16_t storageC = hicma_parsec_trmm_storage_precision(params_tlr, decisionC);
    cudaStream_t stream = cuda_stream->cuda_stream;
    cublasStatus_t status;

    parsec_potrf_stream_workspace_t *stream_found = lookup_gpu_workspace(cuda_device, cuda_stream, ws_gpu);
    cublasHandle_t handle = stream_found->handle_cublas;

    /* A is mb x kb, B is kb x nb, C is mb x nb; the operands take the
     * precision of C's decision (FP16 also with --accumulation_fp32) */
    void *A_use = hicma_parsec_trmm_gpu_convert( A, storageA, decisionC, mb, kb, lda, sr,
            stream_found->gpu_buffer_A, stream );
    void *B_use = hicma_parsec_trmm_gpu_convert( B, storageB, decisionC, kb, nb, ldb, sr,
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
    } else if( DENSE_HP == decisionC ) {
        char alpha[16], beta[16];
        float2half_host( (float)lalpha, &alpha[0] );
        float2half_host( (float)lbeta, &beta[0] );
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

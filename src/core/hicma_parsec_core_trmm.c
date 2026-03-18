/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"


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
    void *A_use = A;
    void *B_use = B;
    void *A_d, *A_s, *B_d, *B_s;
    int NT = params_tlr->NT;
    int enable_stochastic_rounding = params_tlr->enable_stochastic_rounding;
    uint16_t decisionA = params_tlr->decisions[((NT-1)-m)*NT+(NT-1)-m];
    uint16_t decisionB = params_tlr->decisionsB[n*NT+(NT-1)-m];

    /* DP */
    if(DENSE_DP == decisionB) {
        /* Convert A to DP */
        if(DENSE_SP == decisionA) {
            A_d = parsec_private_memory_pop( p_work_full_dp );
            LAPACKE_slag2d( LAPACK_COL_MAJOR, mb, mb, A, mb, A_d, mb );
            A_use = A_d;
        }
#if HAVE_HP_CPU
        else if(DENSE_HP == decisionA) {
            A_d = parsec_private_memory_pop( p_work_full_dp );
            convert_h2d_binary_CPU(A_d, (__fp16 *)A, mb, mb); 
            A_use = A_d;
        }
#endif
        double alpha = (double)lalpha;
        CORE_dtrmm(side, uplo, trans,
                   diag, mb, nb,
                   alpha, A_use /* descA((descB->mt-1)-m,(descB->mt-1)-m) */, lda,
                          B_use /* descB((descB->mt-1)-m,n) */, ldb );

        /* Push back to mempool */
        if(DENSE_DP != decisionA) {
            parsec_private_memory_push( p_work_full_dp, A_d );
        }

    /* SP */
    } else if(DENSE_SP == decisionB) {
        if(DENSE_DP == decisionA) {
            A_s = parsec_private_memory_pop( p_work_full_sp );
            if(enable_stochastic_rounding) {
                double2float_round_CPU(mb, mb, A, mb, A_s, mb);
            } else {
                convert_d2s_binary_CPU( A_s, A, mb, mb );
            }
            A_use = A_s;
        }
#if HAVE_HP_CPU
        else if(DENSE_HP == decisionA) {
            A_s = parsec_private_memory_pop( p_work_full_sp );
            convert_h2s_binary_CPU(A_s, (__fp16 *)A, mb, mb);
            A_use = A_s;
        }
#endif

        //printf("SGEMM %d %d %d\n", m, n, k);

        float alpha = (float)lalpha;
        CORE_strmm(side, uplo, trans,
                   diag, mb, nb,
                   alpha, A_use /* descA((descB->mt-1)-m,(descB->mt-1)-m) */, lda,
                          B_use /* descB((descB->mt-1)-m,n) */, ldb );

        /* Push back to mempool */
        if(DENSE_SP != decisionA) {
            parsec_private_memory_push( p_work_full_sp, A_s );
        }
    }
#if HAVE_HP_CPU
    /* No half-precision trmm, so do strmm */
    else if(DENSE_HP == decisionB) {
        if(DENSE_DP == decisionA) {
            A_s = parsec_private_memory_pop( p_work_full_sp );
            if(enable_stochastic_rounding) {
                double2float_round_CPU(mb, mb, A, mb, A_s, mb);
            } else {
                convert_d2s_binary_CPU( A_s, A, mb, mb );
            }
            A_use = A_s;
        }
        else if(DENSE_HP == decisionA) {
            A_s = parsec_private_memory_pop( p_work_full_sp );
            convert_h2s_binary_CPU(A_s, (__fp16 *)A, mb, mb);
            A_use = A_s;
        }

        /* Convert B to SP */
        B_s = parsec_private_memory_pop( p_work_full_sp );
        convert_h2s_binary_CPU(B_s, (__fp16 *)B, mb, mb);

        float alpha = (float)lalpha;
        CORE_strmm(side, uplo, trans,
                   diag, mb, nb,
                   alpha, A_use /* descA((descB->mt-1)-m,(descB->mt-1)-m) */, lda,
                          B_use /* descB((descB->mt-1)-m,n) */, ldb );

        /* Convert to HP, as GEMM may use in HP */
        convert_s2h_binary_CPU(B, B_s, mb, mb);

        /* Push back to mempool */
        if(DENSE_SP != decisionA) {
            parsec_private_memory_push( p_work_full_hp, A_h );
        }

        parsec_private_memory_push( p_work_full_hp, B_h );
    }
#endif
    else {
        fprintf(stderr, "hicma_parsec_trmm_core_trmm_lln_cpu: decisionC is wrong!\n");
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
    void *A_use = A;
    void *B_use = B;
    void *A_d, *A_s, *A_h, *B_d, *B_s, *B_h;
    int NT = params_tlr->NT;
    int enable_stochastic_rounding = params_tlr->enable_stochastic_rounding;
    uint16_t decisionA = params_tlr->decisions[k*NT+(NT-1)-m];
    uint16_t decisionB = params_tlr->decisionsB[n*NT+k];
    uint16_t decisionC = params_tlr->decisionsB[n*NT+(NT-1)-m];

    /* DP */
    if(DENSE_DP == decisionC) {
        /* Convert A to DP */
        if(DENSE_SP == decisionA) {
            A_d = parsec_private_memory_pop( p_work_full_dp );
            LAPACKE_slag2d( LAPACK_COL_MAJOR, mb, mb, A, mb, A_d, mb );
            A_use = A_d;
        }
#if HAVE_HP_CPU
        else if(DENSE_HP == decisionA) {
            A_d = parsec_private_memory_pop( p_work_full_dp );
            convert_h2d_binary_CPU(A_d, (__fp16 *)A, mb, mb); 
            A_use = A_d;
        }
#endif

        /* Convert B to DP */
        if(DENSE_SP == decisionB) {
            B_d = parsec_private_memory_pop( p_work_full_dp );
            LAPACKE_slag2d( LAPACK_COL_MAJOR, mb, mb, B, mb, B_d, mb );
            B_use = B_d;
        } 
#if HAVE_HP_CPU
        else if(DENSE_HP == decisionB) {
            B_d = parsec_private_memory_pop( p_work_full_dp );
            convert_h2d_binary_CPU(B_d, (__fp16 *)B, mb, mb); 
            B_use = B_d;
        }
#endif

        CORE_dgemm(transA, transB, mb, nb, kb,
                lalpha, A_use /* descA((descB->mt-1)-m,k) */, lda,
                        B_use /* descB(k,n) */,               ldb,
                1.0,    C /* descB((descB->mt-1)-m,n) */, ldc );

        /* Push back to mempool */
        if(DENSE_DP != decisionA) {
            parsec_private_memory_push( p_work_full_dp, A_d );
        }

        if(DENSE_DP != decisionB) {
            parsec_private_memory_push( p_work_full_dp, B_d );
        }

    /* SP */
    } else if(DENSE_SP == decisionC) {
        /* Convert A to DP */
        if(DENSE_DP == decisionA) {
            A_s = parsec_private_memory_pop( p_work_full_sp );
            if(enable_stochastic_rounding) {
                double2float_round_CPU(mb, mb, A, mb, A_s, mb);
            } else {
                convert_d2s_binary_CPU( A_s, A, mb, mb );
            }
            A_use = A_s;
        }
#if HAVE_HP_CPU
        else if(DENSE_HP == decisionA) {
            A_s = parsec_private_memory_pop( p_work_full_sp );
            convert_h2s_binary_CPU(A_s, (__fp16 *)A, mb, mb);
            A_use = A_s;
        }
#endif

        /* Convert B to DP */
        if(DENSE_DP == decisionB) {
            B_s = parsec_private_memory_pop( p_work_full_sp );
            if(enable_stochastic_rounding) {
                double2float_round_CPU(mb, mb, B, mb, B_s, mb);
            } else {
                convert_d2s_binary_CPU(B_s, B, mb, mb );
            }
            B_use = B_s;
        }
#if HAVE_HP_CPU
        else if(DENSE_HP == decisionB) {
            B_s = parsec_private_memory_pop( p_work_full_dp );
            convert_h2s_binary_CPU(B_s, (__fp16 *)B, mb, mb);
            B_use = B_s;
        }
#endif
        //printf("SGEMM %d %d %d\n", m, n, k);

        CORE_sgemm(transA, transB, mb, nb, kb,
                lalpha, A_use /* descA((descB->mt-1)-m,k) */, lda,
                        B_use /* descB(k,n) */,               ldb,
                1.0,    C /* descB((descB->mt-1)-m,n) */, ldc );

        /* Push back to mempool */
        if(DENSE_SP != decisionA) {
            parsec_private_memory_push( p_work_full_sp, A_s );
        }

        if(DENSE_SP != decisionB) {
            parsec_private_memory_push( p_work_full_sp, B_s );
        }

    }
#if HAVE_HP_CPU
    /* HP */
    else if(DENSE_HP == decisionC) {
        /* Convert A to DP */
        if(DENSE_DP == decisionA) {
            A_h = parsec_private_memory_pop( p_work_full_hp );
            convert_d2h_binary_CPU( A_h, A, mb, mb );
            A_use = A_h;
        }
        else if(DENSE_SP == decisionA) {
            A_h = parsec_private_memory_pop( p_work_full_hp );
            convert_s2h_binary_CPU(A_h, (__fp16 *)A, mb, mb);
            A_use = A_h;
        }

        /* Convert B to DP */
        if(DENSE_DP == decisionB) {
            B_h = parsec_private_memory_pop( p_work_full_hp );
            convert_d2h_binary_CPU(B_h, B, mb, mb );
            B_use = B_h;
        }
        else if(DENSE_SP == decisionB) {
            B_h = parsec_private_memory_pop( p_work_full_hp );
            convert_s2h_binary_CPU(B_h, (__fp16 *)B, mb, mb);
            B_use = B_h;
        }

        //printf("SGEMM %d %d %d\n", m, n, k);

        fjcblas_gemm_r16(CblasColMajor, transA, transB,
                mb, nb, kb,
                (__fp16)lalpha, A_use, lda,
                                B_use, ldb,
                (__fp16) 1.0,   C, ldc);

        /* Push back to mempool */
        if(DENSE_HP != decisionA) {
            parsec_private_memory_push( p_work_full_hp, A_h );
        }

        if(DENSE_HP != decisionB) {
            parsec_private_memory_push( p_work_full_hp, B_h );
        }
    }
#endif
    else {
        fprintf(stderr, "hicma_parsec_trmm_core_gemm_lln_cpu: decisionC is wrong!\n");
    }

}


#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

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
    void *A_use = A;
    void *B_use = B;
    void *A_d, *A_s, *B_d, *B_s;
    int NT = params_tlr->NT;
    uint16_t decisionA = params_tlr->decisions[((NT-1)-m)*NT+(NT-1)-m];
    uint16_t decisionB = params_tlr->decisionsB[n*NT+(NT-1)-m];

    cublasStatus_t status;
    int enable_stochastic_rounding = params_tlr->enable_stochastic_rounding;

    /* Find workspace */
    parsec_potrf_workspace_t *_ws_gpu = (parsec_potrf_workspace_t *)ws_gpu;
    parsec_potrf_stream_workspace_t *stream_found = lookup_gpu_workspace(cuda_device, cuda_stream, _ws_gpu);

    /* Get handle_cublas */
    cublasHandle_t handle = stream_found->handle_cublas;
    //cublasSetStream( handle, cuda_stream->cuda_stream );

    /* Get the temporary buffer on GPU */
    A_d = (double *)stream_found->gpu_buffer_A;
    A_s = (float *)stream_found->gpu_buffer_A;
    B_d = (double *)stream_found->gpu_buffer_B;
    B_s = (float *)stream_found->gpu_buffer_B;

    /* DP */
    if(DENSE_DP == decisionB) {
        /* Convert A to DP */
        if(DENSE_SP == decisionA) {
            float2double_GPU(mb, mb, A, mb, A_d, mb, cuda_stream->cuda_stream);
            A_use = A_d;
        }
        else if(DENSE_HP == decisionA) {
            half2double_GPU(mb, mb, A, mb, A_d, mb, cuda_stream->cuda_stream);
            A_use = A_d;
        }

        double alpha = lalpha;
        status = cublasDtrmm( handle,
                dplasma_cublas_side(side),
                dplasma_cublas_fill(uplo),
                dplasma_cublas_op(trans),
                dplasma_cublas_diag(diag),
                mb, nb,
                &alpha, A_use, lda,
                        B_use, ldb,
                        B_use, ldb);
        PARSEC_CUDA_CHECK_ERROR( "cublasDtrmm", status, {exit(PARSEC_HOOK_RETURN_ERROR);} );

    /* SP */
    } else if(DENSE_SP == decisionB) {
        /* Convert A to SP */
        if(DENSE_DP == decisionA) {
            if(enable_stochastic_rounding) {
                double2float_round_GPU(mb, mb, A, mb, A_s, mb, cuda_stream->cuda_stream);
            } else {
                double2float_GPU(mb, mb, A, mb, A_s, mb, cuda_stream->cuda_stream);
            }
            A_use = A_s;
        }
        else if(DENSE_HP == decisionA) {
            half2float_GPU(mb, mb, A, mb, A_s, mb, cuda_stream->cuda_stream);
            A_use = A_s;
        }

        float alpha = (float)lalpha;
        status = cublasStrmm( handle,
                dplasma_cublas_side(side),
                dplasma_cublas_fill(uplo),
                dplasma_cublas_op(trans),
                dplasma_cublas_diag(diag),
                mb, nb,
                &alpha, A_use, lda,
                        B_use, ldb,
                        B_use, ldb);
        PARSEC_CUDA_CHECK_ERROR( "cublasStrmm", status, {exit(PARSEC_HOOK_RETURN_ERROR);} );
    }
    /* No half-precision trmm, so do strmm */
    else if(DENSE_HP == decisionB) {
        /* Convert A to SP */
        if(DENSE_DP == decisionA) {
            if(enable_stochastic_rounding) {
                double2float_round_GPU(mb, mb, A, mb, A_s, mb, cuda_stream->cuda_stream);
            } else {
                double2float_GPU(mb, mb, A, mb, A_s, mb, cuda_stream->cuda_stream);
            }
            A_use = A_s;
        }
        else if(DENSE_HP == decisionA) {
            half2float_GPU(mb, mb, A, mb, A_s, mb, cuda_stream->cuda_stream);
            A_use = A_s;
        }  

        /* Convert B to SP */
        if(!params_tlr->accumulation_fp32) {
            half2float_GPU(mb, mb, B, mb, B_s, mb, cuda_stream->cuda_stream);
            B_use = B_s;
        }

        float alpha = (float)lalpha;
        status = cublasStrmm( handle,
                dplasma_cublas_side(side),
                dplasma_cublas_fill(uplo),
                dplasma_cublas_op(trans),
                dplasma_cublas_diag(diag),
                mb, nb,
                &alpha, A_use, lda,
                        B_use, ldb,
                        B_use, ldb);
        PARSEC_CUDA_CHECK_ERROR( "cublasStrmm in HP", status, {exit(PARSEC_HOOK_RETURN_ERROR);} );

        /* Convert to HP, as GEMM may use in HP */
        if(!params_tlr->accumulation_fp32) {
            if(enable_stochastic_rounding) {
                float2half_round_GPU(mb, mb, B_use, mb, B, mb, cuda_stream->cuda_stream);
            } else {
                float2half_GPU(mb, mb, B_use, mb, B, mb, cuda_stream->cuda_stream);
            }
        }
    }
    else {
        fprintf(stderr, "hicma_parsec_trmm_core_gemm_lln_cpu: decisionC is wrong!\n");
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

    void *A_use = A, *B_use = B;
    void *A_d, *A_s, *A_h, *B_d, *B_s, *B_h, *C_s, *C_h;
    int NT = params_tlr->NT;
    uint16_t decisionA = params_tlr->decisions[k*NT+(NT-1)-m];
    uint16_t decisionB = params_tlr->decisionsB[n*NT+k];
    uint16_t decisionC = params_tlr->decisionsB[n*NT+(NT-1)-m];
    int enable_stochastic_rounding = params_tlr->enable_stochastic_rounding;
    cublasStatus_t status;

    /* Find workspace */
    parsec_potrf_workspace_t *_ws_gpu = (parsec_potrf_workspace_t *)ws_gpu;
    parsec_potrf_stream_workspace_t *stream_found = lookup_gpu_workspace(cuda_device, cuda_stream, _ws_gpu);

    /* Get handle_cublas */
    cublasHandle_t handle = stream_found->handle_cublas;
    //cublasSetStream( handle, cuda_stream->cuda_stream );

    /* Get the temporary buffer on GPU */
    A_d = (double *)stream_found->gpu_buffer_A;
    A_s = (float *)stream_found->gpu_buffer_A;
    A_h = (void *)stream_found->gpu_buffer_A;
    B_d = (double *)stream_found->gpu_buffer_B;
    B_s = (float *)stream_found->gpu_buffer_B;
    B_h = (void *)stream_found->gpu_buffer_B;
    C_s = (void *)stream_found->gpu_buffer_C;
    C_h = (void *)stream_found->gpu_buffer_C;

    /* DP */
    if(DENSE_DP == decisionC) {
        /* Convert A to DP */
        if(DENSE_SP == decisionA) {
            float2double_GPU(mb, mb, A, mb, A_d, mb, cuda_stream->cuda_stream);
            A_use = A_d;
        }
        else if(DENSE_HP == decisionA) {
            half2double_GPU(mb, mb, A, mb, A_d, mb, cuda_stream->cuda_stream);
            A_use = A_d;
        }

        /* Convert B to DP */
        if(DENSE_SP == decisionB || (params_tlr->accumulation_fp32 && DENSE_HP == decisionB)) {
            float2double_GPU(mb, mb, B, mb, B_d, mb, cuda_stream->cuda_stream);
            B_use = B_d;
        } 
        else if(DENSE_HP == decisionB) {
            half2double_GPU(mb, mb, B, mb, B_d, mb, cuda_stream->cuda_stream); 
            B_use = B_d;
        }

        double alpha = lalpha, beta = lbeta; 
        status = hicma_parsec_dgemm_gpu( handle, dplasma_cublas_op(transA), dplasma_cublas_op(transB), //CUBLAS_OP_N,
                mb, nb, kb,
                &alpha, A_use, lda,
                         B_use, ldb,
                &beta,  C, ldc );
        PARSEC_CUDA_CHECK_ERROR( "hicma_parsec_dgemm_gpu", status, {exit(PARSEC_HOOK_RETURN_ERROR);} );

    /* SP */
    } else if(DENSE_SP == decisionC) {
        /* Convert A to SP */
        if(DENSE_DP == decisionA) {
            if(enable_stochastic_rounding) {
                double2float_round_GPU(mb, mb, A, mb, A_s, mb, cuda_stream->cuda_stream);
            } else {
                double2float_GPU(mb, mb, A, mb, A_s, mb, cuda_stream->cuda_stream);
            }
            A_use = A_s;
        }
        else if(DENSE_HP == decisionA) {
            half2float_GPU(mb, mb, A, mb, A_s, mb, cuda_stream->cuda_stream);
            A_use = A_s;
        }

        /* Convert B to SP */
        if(DENSE_DP == decisionB) {
            if(enable_stochastic_rounding) {
                double2float_round_GPU(mb, mb, B, mb, B_s, mb, cuda_stream->cuda_stream);
            } else {
                double2float_GPU(mb, mb, B, mb, B_s, mb, cuda_stream->cuda_stream);
            }
            B_use = B_s;
        }
        else if(DENSE_HP == decisionB && !params_tlr->accumulation_fp32) {
            half2float_GPU(mb, mb, B, mb, B_s, mb, cuda_stream->cuda_stream);
            B_use = B_s;
        }

        //printf("SGEMM GPU %d %d %d\n", m, n, k);

        float alpha = (float)lalpha, beta = (float)lbeta;
        status = cublasGemmEx(handle, dplasma_cublas_op(transA), dplasma_cublas_op(transB), //CUBLAS_OP_N,
                mb, nb, kb,
                &alpha, A_use, CUDA_R_32F, lda,
                        B_use, CUDA_R_32F, ldb,
                &beta,  C, CUDA_R_32F, ldc,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT );
        PARSEC_CUDA_CHECK_ERROR( "hicma_parsec_sgemm_gpu", status, {exit(PARSEC_HOOK_RETURN_ERROR);} );
    }
    /* HP */
    else if(DENSE_HP == decisionC) {
        /* Convert A to HP */
        if(DENSE_DP == decisionA) {
            if(enable_stochastic_rounding) {
                double2half_round_GPU(mb, mb, A, mb, A_h, mb, cuda_stream->cuda_stream);
            } else {
                double2half_GPU(mb, mb, A, mb, A_h, mb, cuda_stream->cuda_stream);
            }
            A_use = A_h;
        }
        else if(DENSE_SP == decisionA) {
            if(enable_stochastic_rounding) {
                float2half_round_GPU(mb, mb, A, mb, A_h, mb, cuda_stream->cuda_stream);
            } else {
                float2half_GPU(mb, mb, A, mb, A_h, mb, cuda_stream->cuda_stream);
            }
            A_use = A_h;
        }

        /* Convert B to HP */
        if(DENSE_DP == decisionB) {
            if(enable_stochastic_rounding) {
                double2half_round_GPU(mb, mb, B, mb, B_h, mb, cuda_stream->cuda_stream);
            } else {
                double2half_GPU(mb, mb, B, mb, B_h, mb, cuda_stream->cuda_stream);
            }
            B_use = B_h;
        }
        else if(DENSE_SP == decisionB || (params_tlr->accumulation_fp32 && DENSE_HP == decisionB)) {
            if(enable_stochastic_rounding) {
                float2half_round_GPU(mb, mb, B, mb, B_h, mb, cuda_stream->cuda_stream);
            } else {
                float2half_GPU(mb, mb, B, mb, B_h, mb, cuda_stream->cuda_stream);
            }
            B_use = B_h;
        }

        //printf("HGEMM %d %d %d\n", m, n, k);


        /* First local GEMM convert C from single to half */
        if(params_tlr->accumulation_fp32) {
            /* Convert datatype to FP32*/
            //half2float_GPU( mb, mb, C, mb, C_s, mb, cuda_stream->cuda_stream );
            //memcpy_float_GPU( mb, mb, C_s, C, cuda_stream->cuda_stream );

//printf("ACC_FP32 %d %d %d\n", m, n, k);

            float alpha = (float)lalpha, beta = (float)lbeta;
            status = cublasGemmEx(handle, dplasma_cublas_op(transA), dplasma_cublas_op(transB), //CUBLAS_OP_N,
                    mb, nb, kb,
                    &alpha, A_use, CUDA_R_16F, lda,
                            B_use, CUDA_R_16F, ldb,
                    &beta,  C, CUDA_R_32F, ldc,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

            /* Convert datatype back to FP16*/
            //float2half_GPU(mb, mb, C, mb, C_h, mb, cuda_stream->cuda_stream);
            //memcpy_half_GPU( mb, mb, C_h, C, cuda_stream->cuda_stream );

            PARSEC_CUDA_CHECK_ERROR( "hicma_parsec_hgemm_gpu_acc32", status, {exit(PARSEC_HOOK_RETURN_ERROR);} );
        } else {
            char alpha[16], beta[16];
            float2half_host((float)lalpha, &alpha[0]);
            float2half_host((float)lbeta, &beta[0]);
            status = cublasGemmEx(handle, dplasma_cublas_op(transA), dplasma_cublas_op(transB), //CUBLAS_OP_N,
                    mb, nb, kb,
                    &alpha[0], A_use, CUDA_R_16F, lda,
                    B_use, CUDA_R_16F, ldb,
                    &beta[0],  C, CUDA_R_16F, ldc,
                    CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
            PARSEC_CUDA_CHECK_ERROR( "hicma_parsec_hgemm_gpu_acc16", status, {exit(PARSEC_HOOK_RETURN_ERROR);} );
        }
    }
    else {
        fprintf(stderr, "hicma_parsec_trmm_core_gemm_lln_gpu: decisionC is wrong!\n");
    }

}

#endif

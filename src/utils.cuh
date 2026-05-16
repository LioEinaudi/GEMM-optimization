#pragma once
#include<stdio.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<math.h>

/*
long long seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((long long )tp.tv_sec * 1e6 + (long long )tp.tv_usec);
}
*/

void InitialData ( float * ip , const int n ) {
    for (int i = 0; i < n; i ++ ) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void Check (float *Ref1 , float *Ref2 , const int n ) {
    bool match = 1;
    for (int i = 0; i < n; i ++ ) {
        if (fabs( Ref1[i] - Ref2[i] ) > 1e-2 ){
            match = 0;
            printf("Mismatch at %d: host=%f gpu=%f diff=%f\n",
                   i, Ref1[i], Ref2[i], fabs(Ref1[i] - Ref2[i]));
            break; 
        }
    }
    if ( match )
        printf("Match !\n"); 
}

void gemmOnCPU ( float *A , float *B  , float *C , const int M , const int K , const int N ) {
    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < N; col++)
        {
            float sum = 0.0f;
            for (int i = 0; i < K; i ++  ){
                sum += A[row * K + i] * B[N * i + col]; 
            }
            C[row*N+col] = sum; 
        }
    }
}

bool LaunchCublas(cublasHandle_t handle, float *d_A, float *d_B, float *d_C,
                  const int M, const int K, const int N)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        N, M, K,
                                        &alpha,
                                        d_B, N,
                                        d_A, K,
                                        &beta,
                                        d_C, N);
    return status == CUBLAS_STATUS_SUCCESS;
}


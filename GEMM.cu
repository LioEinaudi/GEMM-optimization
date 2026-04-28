#include<stdio.h> 
#include<cuda_runtime.h>
//#include<sys/time.h> 
#define TILE_M 16 
#define TILE_K 16 
#define TILE_N 16 
#define IPAD 1 
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

__global__ void Warmup(float *A, float *B, float *C, const int M, const int K, const int N)
{
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < K; i++)
        {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void gemmnative(float *A, float *B, float *C, const int M, const int K, const int N) {
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y; 

    if ( row < M && col < N ) {
        float sum = 0.0f;
        for (int i = 0; i < K; i ++ ) {
            sum += A[row * K + i] * B[i * N + col]; 
        }
        C[row * N + col] = sum; 
    }
}

__global__ void gemmUnroll2(float *A, float *B, float *C, const int M, const int K, const int N){
    unsigned int col = threadIdx.x + 2 * blockDim.x * blockIdx.x;
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y; 

    if ( row < M &&  col + blockDim.x < N ) {
        float sum1 = 0.0f, sum2 = 0.0f;
        for (int i = 0; i < K; i ++ ){
            sum1 += A[row * K + i] * B[i * N + col];
            sum2 += A[row  * K + i] * B[i * N + col + blockDim.x ]; 
        }
        C[row * N + col] = sum1;
        C[row * N + col + blockDim.x ] = sum2; 

    }
}

__global__ void gemmUnroll4(float *A, float *B, float *C, const int M, const int K, const int N)
{
    unsigned int col = threadIdx.x + 4 * blockDim.x * blockIdx.x;
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < M && col + blockDim.x * 3  < N)
    {
        float sum1 = 0.0f, sum2 = 0.0f,sum3 =0.0f , sum4=0.0f ;
        for (int i = 0; i < K; i++)
        {
            sum1 += A[row * K + i] * B[i * N + col];
            sum2 += A[row * K + i] * B[i * N + col + blockDim.x];
            sum3 += A[row * K + i] * B[i * N + col + blockDim.x * 2 ];
            sum4 += A[row * K + i] * B[i * N + col + blockDim.x * 3 ];
        }
        C[row * N + col] = sum1;
        C[row * N + col + blockDim.x] = sum2;
        C[row * N + col + blockDim.x * 2 ] = sum3;
        C[row * N + col + blockDim.x * 3 ] = sum4; 
    }
}

__global__ void gemmSmem (float *A, float *B, float *C, const int M, const int K, const int N){
    __shared__ float SmemA[TILE_M][TILE_K];
    __shared__ float SmemB[TILE_K][TILE_N]; 
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;

    float sum = 0.0f;
    for (int tile = 0; tile < (K + TILE_K - 1) / TILE_K ; tile ++) {
        int aCol = tile * TILE_M + threadIdx.x;
        int bRow = tile * TILE_N + threadIdx.y; 

        if ( row < M && aCol < K ) {
            SmemA[threadIdx.y][threadIdx.x] = A[row * K + aCol]; 
        }
        else {
            SmemA[threadIdx.y][threadIdx.x] = 0.0f; 
        }
        if ( bRow < K && col < N ) {
            SmemB[threadIdx.y][threadIdx.x] = B[bRow * N + col] ;
        }
        else {
            SmemB[threadIdx.y][threadIdx.x] = 0.0f; 
        }
        __syncthreads();
        for (int i = 0; i < TILE_K; i ++ ) {
            sum += SmemA[threadIdx.y][i] * SmemB[i][threadIdx.x]; 
        }
        __syncthreads(); 
    }
    if ( row < M && col < N ) {
        C[row * N + col] = sum; 
    }
}

__global__ void gemmSmemPad(float *A, float *B, float *C, const int M, const int K, const int N)
{
    __shared__ float SmemA[TILE_M][TILE_K+IPAD];
    __shared__ float SmemB[TILE_K][TILE_N+IPAD]; 
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;

    float sum = 0.0f;
    for (int tile = 0; tile < (K + TILE_K - 1) / TILE_K; tile++)
    {
        int aCol = tile * TILE_K + threadIdx.x;
        int bRow = tile * TILE_K + threadIdx.y;

        if (row < M && aCol < K)
        {
            SmemA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        }
        else
        {
            SmemA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (bRow < K && col < N)
        {
            SmemB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        }
        else
        {
            SmemB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for (int i = 0; i < TILE_K; i++)
        {
            sum += SmemA[threadIdx.y][i] * SmemB[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

__global__ void gemmSmemUnroll2(float *A, float *B, float *C, const int M, const int K, const int N){
    __shared__ float SmemA[TILE_M][TILE_K] ;
    __shared__ float SmemB[TILE_K][TILE_N * 2];

    unsigned int col = threadIdx.x + 2 * TILE_N * blockIdx.x;
    unsigned int row = threadIdx.y + TILE_M* blockIdx.y; 

    float sum1 = 0.0f, sum2 = 0.0f;
    for (int tile = 0; tile < (K + TILE_K - 1) / TILE_K; tile ++ ) {
        int aCol = tile * TILE_K + threadIdx.x;
        int bRow = tile * TILE_K + threadIdx.y; 

        if ( aCol < K && row < M ) {
            SmemA[threadIdx.y][threadIdx.x] = A[row*K + aCol]; 
        }
        else {
            SmemA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ( bRow < K && col + TILE_N < N ) {
            SmemB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
            SmemB[threadIdx.y][threadIdx.x + TILE_N] = B[bRow * N + col + TILE_N]; 
        }
        else {
            SmemB[threadIdx.y][threadIdx.x] = 0.0f;
            SmemB[threadIdx.y][threadIdx.x + TILE_N] = 0.0f; 
        }
        __syncthreads();
        for (int i = 0; i < TILE_K; i ++ ){
            sum1 += SmemA[threadIdx.y][i] * SmemB[i][threadIdx.x];
            sum2 += SmemA[threadIdx.y][i] * SmemB[i][threadIdx.x + TILE_N]; 
        }
        __syncthreads(); 
    }
    if ( col + TILE_N < N && row < M ){
            C[row * N + col] = sum1;
            C[row * N + col + TILE_N] = sum2; 
        }
}

    int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("device %d : %s \n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // C = A * B
    // A: M x K
    // B: K x N
    // C: M x N
    int M = 1 << 12;
    int N = 1 << 12;
    int K = 1 << 12;

    int blockx = 16;
    int blocky = 16;

    if ( argc > 1 ) 
        blockx = atoi (argv[1]) ;
    if ( argc > 2 )
        blocky = atoi(argv[2]); 
    
    dim3 block(blockx, blocky);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    size_t nBytesA = M * K * sizeof(float);
    size_t nBytesB = K * N * sizeof(float);
    size_t nBytesC = M * N * sizeof(float);

    float *h_A, *h_B, *h_C , *gpu_Ref ;
    h_A = (float *)malloc(nBytesA);
    h_B = (float *)malloc(nBytesB);
    h_C = (float *)malloc(nBytesC);
    gpu_Ref = (float *)malloc(nBytesC); 


    InitialData(h_A, M * K);
    InitialData(h_B, K * N);

    //M , K ,N = 1 << 8 时cpu计算时间为31763us , gpu gemmnative计算时间为139us 

    /*long long iStart = seconds(); 
    gemmOnCPU(h_A, h_B, h_C, M, K, N);
    long long iElapse = seconds()-iStart ;
    printf("CPU Elpase %lld us \n", iElapse); */

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, nBytesA);
    cudaMalloc((void **)&d_B, nBytesB);
    cudaMalloc((void **)&d_C, nBytesC);

    cudaMemcpy(d_A, h_A, nBytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytesB, cudaMemcpyHostToDevice); 


    //Warmup
    Warmup<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    // iElapse = seconds() - iStart;
    // printf("gemmnative Elapse %lld us \n", iElapse);
    cudaMemcpy(h_C, d_C, nBytesC, cudaMemcpyDeviceToHost);
    

    //kernel 1
    //iStart = seconds (); 
    gemmnative<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    //iElapse = seconds() - iStart;
    //printf("gemmnative Elapse %lld us \n", iElapse);
    cudaMemcpy(gpu_Ref, d_C, nBytesC, cudaMemcpyDeviceToHost);
    Check(h_C, gpu_Ref, M * N);

    free(gpu_Ref);
    cudaFree(d_C);
    gpu_Ref = (float *)malloc(nBytesC);
    cudaMalloc((void **)&d_C, nBytesC); 

    //kernel 2
    dim3 gridUnroll2((N + block.x * 2 - 1) / (block.x * 2), (M + block.y - 1) / block.y);
    gemmUnroll2 <<<gridUnroll2 , block >>> ( d_A , d_B ,d_C ,M ,K , N ) ;
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_Ref, d_C, nBytesC, cudaMemcpyDeviceToHost);
    Check(h_C, gpu_Ref, M * N);

    free(gpu_Ref);
    cudaFree(d_C);
    gpu_Ref = (float *)malloc(nBytesC);
    cudaMalloc((void **)&d_C, nBytesC);

    //kernel3
    dim3 gridUnroll4((N + block.x * 4 - 1) / (block.x * 4), (M + block.y - 1) / block.y);
    gemmUnroll4<<<gridUnroll4, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_Ref, d_C, nBytesC, cudaMemcpyDeviceToHost);
    Check(h_C, gpu_Ref, M * N);

    free(gpu_Ref);
    cudaFree(d_C);
    gpu_Ref = (float *)malloc(nBytesC);
    cudaMalloc((void **)&d_C, nBytesC);

    //kernel 4
    gemmSmem<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_Ref, d_C, nBytesC, cudaMemcpyDeviceToHost);
    Check(h_C, gpu_Ref, M * N);

    free(gpu_Ref);
    cudaFree(d_C);
    gpu_Ref = (float *)malloc(nBytesC);
    cudaMalloc((void **)&d_C, nBytesC);

    // kernel 5
    gemmSmemPad<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_Ref, d_C, nBytesC, cudaMemcpyDeviceToHost);
    Check(h_C, gpu_Ref, M * N);

    free(gpu_Ref);
    cudaFree(d_C);
    gpu_Ref = (float *)malloc(nBytesC);
    cudaMalloc((void **)&d_C, nBytesC);


    //kernel 6 
    dim3 gridSmemUnroll2 ( ( N + TILE_N * 2 -1 ) / ( TILE_N * 2 ) , ( M + TILE_M - 1 ) / TILE_M ) ; 
    gemmSmemUnroll2<<<gridSmemUnroll2, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_Ref, d_C, nBytesC, cudaMemcpyDeviceToHost);
    Check(h_C, gpu_Ref, M * N);

    free(gpu_Ref);
    cudaFree(d_C);
    gpu_Ref = (float *)malloc(nBytesC);
    cudaMalloc((void **)&d_C, nBytesC);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(gpu_Ref); 

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
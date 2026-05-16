#include"utils.cuh"
#include"kernels.cuh"

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

    int iKernel = 0;
    int iKernelmax = 16;

    int blockx = 16;
    int blocky = 16;

    if ( argc > 1 )
        iKernel = atoi(argv[1]); 
    
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
    
    if (iKernel < 0 || iKernel > iKernelmax)
    {
        printf("Invalid kernel id %d. Use 0 to run all kernels, or 1-16 to run one kernel.\n", iKernel);
        return EXIT_FAILURE;
    }

    cublasHandle_t cublasHandle;
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
    {
        printf("Failed to create cuBLAS handle.\n");
        return EXIT_FAILURE;
    }
    cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);

    int firstKernel = (iKernel == 0) ? 1 : iKernel;
    int lastKernel = (iKernel == 0) ? iKernelmax : iKernel;
    int exitCode = EXIT_SUCCESS;

    for (int kernel = firstKernel; kernel <= lastKernel; kernel++)
    {
        printf("Kernel %d: %s\n", kernel, KernelName(kernel));
        cudaMemset(d_C, 0, nBytesC);

        bool launched = (kernel == 10)
                            ? LaunchCublas(cublasHandle, d_A, d_B, d_C, M, K, N)
                            : LaunchKernel(kernel, d_A, d_B, d_C, M, K, N, block, grid);

        if (!launched)
        {
            printf("Invalid kernel id %d\n", kernel);
            exitCode = EXIT_FAILURE;
            break;
        }

        cudaDeviceSynchronize();
        cudaMemcpy(gpu_Ref, d_C, nBytesC, cudaMemcpyDeviceToHost);
        Check(h_C, gpu_Ref, M * N);
    }

    cublasDestroy(cublasHandle);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(gpu_Ref); 

    cudaDeviceReset();
    return exitCode;
}

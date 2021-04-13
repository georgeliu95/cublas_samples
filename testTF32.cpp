#include <iostream>

#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

static const char* cublasGetErrorEnum(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "<unknown>";
    }
}

void check(cublasStatus_t ret, int line) {
    if(ret != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error " << cublasGetErrorEnum(ret) << " in line " << line << std::endl;
    }
}

void check(cudaError_t ret, int line) {
    if(ret != cudaSuccess) {
        std::cerr << "CUDA Error " << cudaGetErrorString(ret) << " in line " << line << std::endl;
    }
}

#define CHECK(_x) check((_x), __LINE__)

cublasStatus_t row_major_gemm(cublasHandle_t handle,
                              cublasOperation_t transa,
                              cublasOperation_t transb,
                              int m,
                              int n,
                              int k,
                              const void    *alpha,
                              const void     *A,
                              cudaDataType_t Atype,
                              int lda,
                              const void     *B,
                              cudaDataType_t Btype,
                              int ldb,
                              const void    *beta,
                              void           *C,
                              cudaDataType_t Ctype,
                              int ldc,
                              cublasComputeType_t computeType,
                              cublasGemmAlgo_t algo) {

    lda = (transa==CUBLAS_OP_N)?k:m;
    ldb = (transb==CUBLAS_OP_N)?n:k;
    ldc = n;

    return cublasGemmEx(handle,
                        transb,
                        transa,
                        n, m, k,
                        alpha,
                        B, Btype, ldb,
                        A, Atype, lda,
                        beta,
                        C, Ctype, ldc,
                        computeType, algo);
}



int main() {

    float *dev_a, *dev_b, *dev_c;
    int m=3,n=4,k=5;
    float alpha = 1.0f, beta = 0.f;
    cublasHandle_t handle = nullptr;

    // init array
    float* MatA = new float[m*k];
    float* MatB = new float[n*k];
    float* MatC = new float[m*n];
    for(int i=0; i<m*k; ++i) {
        MatA[i] = static_cast<float>(i);
    }
    for(int i=0; i<n*k; ++i) {
        MatB[i] = static_cast<float>(i);
    }

    CHECK(cudaMalloc((void**)&dev_a, m*k*sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_b, n*k*sizeof(float)));
    CHECK(cudaMalloc((void**)&dev_c, m*n*sizeof(float)));

    CHECK(cudaMemcpy(dev_a, MatA, m*k*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, MatB, n*k*sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cublasCreate_v2(&handle));
    CHECK(row_major_gemm(handle, 
                        CUBLAS_OP_N, 
                        CUBLAS_OP_N,
                        m,
                        n,
                        k,
                        &alpha,
                        const_cast<const void*>(reinterpret_cast<void*>(dev_a)),
                        CUDA_R_32F, m,
                        const_cast<const void*>(reinterpret_cast<void*>(dev_b)),
                        CUDA_R_32F, n,
                        &beta,
                        reinterpret_cast<void*>(dev_c),
                        CUDA_R_32F, k,
                        CUBLAS_COMPUTE_32F_FAST_TF32,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK(cublasDestroy_v2(handle));


    CHECK(cudaMemcpy(MatC, dev_c, m*n*sizeof(float), cudaMemcpyDeviceToHost));

    for(int i=0; i<m; ++i) {
        for(int j=0; j<n; ++j) {
            printf("%f ", MatC[i*n + j]);
        }
        printf("\n");
    }

    // destroy array
    delete [] MatA;
    delete [] MatB;
    delete [] MatC;

    return 0;
}
#include <stdio.h>
#include <cuda.h>

const int TM = 8;
const int TN = 8;
const int BLOCK_DIM_x = 16;
const int BLOCK_DIM_y = 16;
const int BM = TM * BLOCK_DIM_x;
const int BN = TN * BLOCK_DIM_y;
const int BK = 8;

#include <mma.h>
using namespace nvcuda;
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 8;
const int warpSize = 32;
const int warpNum = BLOCK_DIM_x * BLOCK_DIM_y / warpSize;
const int warpX = (warpNum == 1 ? 1 : 2);
const int warpY = warpNum / warpX;


template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel1st(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float SA[BM * BK];
    __shared__ float SB[BK * BN];
    int indA = TM * (blockIdx.x * blockDim.x);
    int indB = TN * (blockIdx.y * blockDim.y);
    int width = (K + BK - 1) / BK;
    float tmp[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int smem_a_m = tid / 2;
    int smem_a_k = tid % 2;
    int smem_b_k = tid / 32;
    int smem_b_n = tid % 32;

    // load A matrix from share memory
    // A matrix index is [8 * threadIdx.x, BK] in matrix with [128, BK]
    // B matrix index is [BK, 8 * threadIdx.y] in matrix with [8, 128]
    // so the neighbour threadIdx.x offset is 128 which meet bank conflict (access SA)

    // share memory is divide into 32 bank, each bank length is 4B
    // The best scene is different threads of the same wrap access different banks

    for (int ph = 0; ph < width; ph++)
    {
        (float4 &)SA[smem_a_m * BK + 4 * smem_a_k] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];
        (float4 &)SB[smem_b_k * BN + 4 * smem_b_n] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];
        for (int id = 0; id < 4; id++)
        {
            if (indA + smem_a_m >= M || ph * BK + 4 * smem_a_k + id >= K)
            {
                SA[smem_a_m * BK + 4 * smem_a_k + id] = 0.0f;
            }
            if (indB + 4 * smem_b_n + id >= N || smem_b_k + ph * BK >= K)
            {

                SB[smem_b_k * BN + 4 * smem_b_n + id] = 0.0f;
            }
        }

        __syncthreads();
        for (int index_q = 0; index_q < TM; index_q++)
        {
            for (int index_v = 0; index_v < TN; index_v++)
            {
                int reg_c_m = threadIdx.x * TM + index_q;
                int reg_c_n = threadIdx.y * TN + index_v;
                for (int index_k = 0; index_k < BK; index_k++)
                {
                    tmp[index_q * TN + index_v] += SA[reg_c_m * BK + index_k] * SB[index_k * BN + reg_c_n];
                }
            }
        }
        __syncthreads();
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            int reg_c_m = threadIdx.x * TM + index_q;
            int reg_c_n = threadIdx.y * TN + index_v;
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_q * TN + index_v];
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel2nd(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float SA[BM * BK];
    __shared__ float SB[BK * BN];
    int indA = TM * (blockIdx.x * blockDim.x);
    int indB = TN * (blockIdx.y * blockDim.y);
    int width = (K + BK - 1) / BK;
    float tmp[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int smem_a_m = tid / 2;
    int smem_a_k = tid % 2;
    int smem_b_k = tid / 32;
    int smem_b_n = tid % 32;
    float a[4];
    for (int ph = 0; ph < width; ph++)
    {
        (float4 &)a[0] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];
        for (int id = 0; id < 4; id++)
        {
            if (indA + smem_a_m >= M || ph * BK + 4 * smem_a_k + id >= K)
            {
                SA[(4 * smem_a_k + id) * BM + smem_a_m] = 0.0f;
            }
            else
            {
                SA[(4 * smem_a_k + id) * BM + smem_a_m] = a[id];
            }
        }
        (float4 &)SB[smem_b_k * BN + 4 * smem_b_n] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];
        for (int id = 0; id < 4; id++)
        {

            if (indB + 4 * smem_b_n + id >= N || smem_b_k + ph * BK >= K)
            {

                SB[smem_b_k * BN + 4 * smem_b_n + id] = 0.0f;
            }
        }

        __syncthreads();
        for (int index_q = 0; index_q < TM; index_q++)
        {
            for (int index_v = 0; index_v < TN; index_v++)
            {
                int reg_c_m = threadIdx.x * TM + index_q;
                int reg_c_n = threadIdx.y * TN + index_v;
                for (int index_k = 0; index_k < BK; index_k++)
                {
                    tmp[index_q * TN + index_v] += SA[index_k * BM + reg_c_m] * SB[index_k * BN + reg_c_n];
                }
            }
        }
        __syncthreads();
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            int reg_c_m = threadIdx.x * TM + index_q;
            int reg_c_n = threadIdx.y * TN + index_v;
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_q * TN + index_v];
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel3rd(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float SA[BM * BK];
    __shared__ float SB[BK * BN];
    int indA = TM * (blockIdx.x * blockDim.x);
    int indB = TN * (blockIdx.y * blockDim.y);
    int width = (K + BK - 1) / BK;
    float tmp[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int smem_a_m = tid / 2;
    int smem_a_k = tid % 2;
    int smem_b_k = tid / 32;
    int smem_b_n = tid % 32;
    float a[4];
    float com_a[TM];
    float com_b[TN];
    for (int ph = 0; ph < width; ph++)
    {
        (float4 &)a[0] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];
        for (int id = 0; id < 4; id++)
        {
            if (indA + smem_a_m >= M || ph * BK + 4 * smem_a_k + id >= K)
            {
                SA[(4 * smem_a_k + id) * BM + smem_a_m] = 0.0f;
            }
            else
            {
                SA[(4 * smem_a_k + id) * BM + smem_a_m] = a[id];
            }
        }
        (float4 &)SB[smem_b_k * BN + 4 * smem_b_n] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];
        for (int id = 0; id < 4; id++)
        {

            if (indB + 4 * smem_b_n + id >= N || smem_b_k + ph * BK >= K)
            {

                SB[smem_b_k * BN + 4 * smem_b_n + id] = 0.0f;
            }
        }

        __syncthreads();

        for (int index_k = 0; index_k < BK; index_k++)
        {
            (float4 &)com_a[0] = (float4 &)SA[index_k * BM + threadIdx.x * TM];
            (float4 &)com_a[4] = (float4 &)SA[index_k * BM + threadIdx.x * TM + 4];
            (float4 &)com_b[0] = (float4 &)SB[index_k * BN + threadIdx.y * TN];
            (float4 &)com_b[4] = (float4 &)SB[index_k * BN + threadIdx.y * TN + 4];
            for (int index_q = 0; index_q < TM; index_q++)
            {
                for (int index_v = 0; index_v < TN; index_v++)
                {
                    tmp[index_q * TN + index_v] += com_a[index_q] * com_b[index_v];
                }
            }
        }

        __syncthreads();
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            int reg_c_m = threadIdx.x * TM + index_q;
            int reg_c_n = threadIdx.y * TN + index_v;
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_q * TN + index_v];
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel4th(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float SA[BM * BK * 2];
    __shared__ float SB[BK * BN * 2];
    int indA = TM * (blockIdx.x * blockDim.x);
    int indB = TN * (blockIdx.y * blockDim.y);
    int width = (K + BK - 1) / BK;
    float tmp[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int smem_a_m = tid / 2;
    int smem_a_k = tid % 2;
    int smem_b_k = tid / 32;
    int smem_b_n = tid % 32;
    float a[4];
    float com_a[TM];
    float com_b[TN];
    int ph = 0;
    (float4 &)a[0] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];
    for (int id = 0; id < 4; id++)
    {
        if (indA + smem_a_m >= M || ph * BK + 4 * smem_a_k + id >= K)
        {
            SA[(4 * smem_a_k + id) * BM + smem_a_m] = 0.0f;
        }
        else
        {
            SA[(4 * smem_a_k + id) * BM + smem_a_m] = a[id];
        }
    }
    (float4 &)SB[smem_b_k * BN + 4 * smem_b_n] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];
    for (int id = 0; id < 4; id++)
    {

        if (indB + 4 * smem_b_n + id >= N || smem_b_k + ph * BK >= K)
        {

            SB[smem_b_k * BN + 4 * smem_b_n + id] = 0.0f;
        }
    }
    __syncthreads();

    for (int ph = 1; ph < width; ph++)
    {
        (float4 &)a[0] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];
        for (int id = 0; id < 4; id++)
        {
            if (indA + smem_a_m >= M || ph * BK + 4 * smem_a_k + id >= K)
            {
                SA[(4 * smem_a_k + id) * BM + smem_a_m + ph % 2 * BM * BK] = 0.0f;
            }
            else
            {
                SA[(4 * smem_a_k + id) * BM + smem_a_m + ph % 2 * BM * BK] = a[id];
            }
        }
        (float4 &)SB[smem_b_k * BN + 4 * smem_b_n + ph % 2 * BN * BK] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];
        for (int id = 0; id < 4; id++)
        {

            if (indB + 4 * smem_b_n + id >= N || smem_b_k + ph * BK >= K)
            {

                SB[smem_b_k * BN + 4 * smem_b_n + id + ph % 2 * BN * BK] = 0.0f;
            }
        }
        for (int index_k = 0; index_k < BK; index_k++)
        {
            (float4 &)com_a[0] = (float4 &)SA[index_k * BM + threadIdx.x * TM + (ph - 1) % 2 * BM * BK];
            (float4 &)com_a[4] = (float4 &)SA[index_k * BM + threadIdx.x * TM + 4 + (ph - 1) % 2 * BM * BK];
            (float4 &)com_b[0] = (float4 &)SB[index_k * BN + threadIdx.y * TN + (ph - 1) % 2 * BN * BK];
            (float4 &)com_b[4] = (float4 &)SB[index_k * BN + threadIdx.y * TN + 4 + (ph - 1) % 2 * BN * BK];
            for (int index_q = 0; index_q < TM; index_q++)
            {
                for (int index_v = 0; index_v < TN; index_v++)
                {
                    tmp[index_q * TN + index_v] += com_a[index_q] * com_b[index_v];
                }
            }
        }

        __syncthreads();
    }
    ph = width;
    for (int index_k = 0; index_k < BK; index_k++)
    {
        (float4 &)com_a[0] = (float4 &)SA[index_k * BM + threadIdx.x * TM + (ph - 1) % 2 * BM * BK];
        (float4 &)com_a[4] = (float4 &)SA[index_k * BM + threadIdx.x * TM + 4 + (ph - 1) % 2 * BM * BK];
        (float4 &)com_b[0] = (float4 &)SB[index_k * BN + threadIdx.y * TN + (ph - 1) % 2 * BN * BK];
        (float4 &)com_b[4] = (float4 &)SB[index_k * BN + threadIdx.y * TN + 4 + (ph - 1) % 2 * BN * BK];
        for (int index_q = 0; index_q < TM; index_q++)
        {
            for (int index_v = 0; index_v < TN; index_v++)
            {
                tmp[index_q * TN + index_v] += com_a[index_q] * com_b[index_v];
            }
        }
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            int reg_c_m = threadIdx.x * TM + index_q;
            int reg_c_n = threadIdx.y * TN + index_v;
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_q * TN + index_v];
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel5th(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float SA[BM * BK * 2];
    __shared__ float SB[BK * BN * 2];
    int indA = TM * (blockIdx.x * blockDim.x);
    int indB = TN * (blockIdx.y * blockDim.y);
    int width = (K + BK - 1) / BK;
    float tmp[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int smem_a_m = tid / 2;
    int smem_a_k = tid % 2;
    int smem_b_k = tid / 32;
    int smem_b_n = tid % 32;
    float a[4];
    float b[4];
    float com_a[TM];
    float com_b[TN];
    int ph = 0;
    (float4 &)a[0] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];
    SA[(4 * smem_a_k) * BM + smem_a_m + ph % 2 * BM * BK] = a[0];
    SA[(4 * smem_a_k + 1) * BM + smem_a_m + ph % 2 * BM * BK] = a[1];
    SA[(4 * smem_a_k + 2) * BM + smem_a_m + ph % 2 * BM * BK] = a[2];
    SA[(4 * smem_a_k + 3) * BM + smem_a_m + ph % 2 * BM * BK] = a[3];
    (float4 &)b[0] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];
    (float4 &)SB[smem_b_k * BN + 4 * smem_b_n] = (float4 &)b[0];

    __syncthreads();

    for (int ph = 1; ph < width; ph++)
    {
        (float4 &)a[0] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];
        (float4 &)b[0] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];

        for (int index_k = 0; index_k < BK; index_k++)
        {
            (float4 &)com_a[0] = (float4 &)SA[index_k * BM + threadIdx.x * TM + (ph - 1) % 2 * BM * BK];
            (float4 &)com_a[4] = (float4 &)SA[index_k * BM + threadIdx.x * TM + 4 + (ph - 1) % 2 * BM * BK];
            (float4 &)com_b[0] = (float4 &)SB[index_k * BN + threadIdx.y * TN + (ph - 1) % 2 * BN * BK];
            (float4 &)com_b[4] = (float4 &)SB[index_k * BN + threadIdx.y * TN + 4 + (ph - 1) % 2 * BN * BK];
            for (int index_q = 0; index_q < TM; index_q++)
            {
                for (int index_v = 0; index_v < TN; index_v++)
                {
                    tmp[index_q * TN + index_v] += com_a[index_q] * com_b[index_v];
                }
            }
        }
        SA[(4 * smem_a_k) * BM + smem_a_m + ph % 2 * BM * BK] = a[0];
        SA[(4 * smem_a_k + 1) * BM + smem_a_m + ph % 2 * BM * BK] = a[1];
        SA[(4 * smem_a_k + 2) * BM + smem_a_m + ph % 2 * BM * BK] = a[2];
        SA[(4 * smem_a_k + 3) * BM + smem_a_m + ph % 2 * BM * BK] = a[3];

        (float4 &)SB[smem_b_k * BN + 4 * smem_b_n + ph % 2 * BN * BK] = (float4 &)b[0];
        __syncthreads();
    }

    ph = width;
    for (int index_k = 0; index_k < BK; index_k++)
    {
        (float4 &)com_a[0] = (float4 &)SA[index_k * BM + threadIdx.x * TM + (ph - 1) % 2 * BM * BK];
        (float4 &)com_a[4] = (float4 &)SA[index_k * BM + threadIdx.x * TM + 4 + (ph - 1) % 2 * BM * BK];
        (float4 &)com_b[0] = (float4 &)SB[index_k * BN + threadIdx.y * TN + (ph - 1) % 2 * BN * BK];
        (float4 &)com_b[4] = (float4 &)SB[index_k * BN + threadIdx.y * TN + 4 + (ph - 1) % 2 * BN * BK];
        for (int index_q = 0; index_q < TM; index_q++)
        {
            for (int index_v = 0; index_v < TN; index_v++)
            {
                tmp[index_q * TN + index_v] += com_a[index_q] * com_b[index_v];
            }
        }
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            int reg_c_m = threadIdx.x * TM + index_q;
            int reg_c_n = threadIdx.y * TN + index_v;
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_q * TN + index_v];
            }
        }
    }
}
__global__ void row_wmma_ker(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int lda = K;
    int ldb = N;
    int ldc = N;

    int indA = blockIdx.x * warpX * WMMA_M;
    int indB = blockIdx.y * warpY * WMMA_N;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warpId = tid / warpSize;
    int warpIdx = warpId % warpX;
    int warpIdy = warpId / warpX;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> left_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> right_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);
    int aRow = indA + warpIdx * WMMA_M;
    int bCol = indB + warpIdy * WMMA_N;
    int width = (K + WMMA_K - 1) / WMMA_K;
    for (int i = 0; i < width; i++)
    {
        int aCol = i * WMMA_K;
        int bRow = i * WMMA_K;
        if (aRow < M && aCol < K && bRow < K && bCol < N)
        {
            // load the element of submatrix of A/B
            wmma::load_matrix_sync(left_frag, dA + aRow * lda + aCol, lda);
            wmma::load_matrix_sync(right_frag, dB + bRow * ldb + bCol, ldb);

            // submatrix mad
            wmma::mma_sync(c_frag, left_frag, right_frag, c_frag);
        }
    }
    int cRow = aRow;
    int cCol = bCol;
    if (cRow < M && cCol < N)
    {
        // Store the output
        wmma::store_matrix_sync(dC + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
    }
}
extern "C" void matmul_cuda_f32(void const *dA, void const *dB, void *dC, int M, int K, int N)
{
    // normal cuda call
    int num_blocks_x = (M + BM - 1) / BM;
    int num_blocks_y = (N + BN - 1) / BN;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
    matrixKernel1st<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>((float *)dA, (float *)dB, (float *)dC, M, K, N);
    // matrixKernel2nd<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>((float *)dA, (float *)dB, (float *)dC, M, K, N);
    // matrixKernel3rd<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>((float *)dA, (float *)dB, (float *)dC, M, K, N);
    // matrixKernel4th<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>((float *)dA, (float *)dB, (float *)dC, M, K, N);
    // matrixKernel5th<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>((float *)dA, (float *)dB, (float *)dC, M, K, N);

    // cudnn matrix call
    // int num_block_x = (M + WMMA_M * warpX - 1) / (WMMA_M * warpX);
    // int num_block_y = (N + WMMA_N * warpY - 1) / (WMMA_N * warpY);
    // dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    // dim3 grid_dim(num_block_x, num_block_y, 1);
    // row_wmma_ker<<<grid_dim, block_dim>>>((float *)dA, (float *)dB, (float *)dC, M, K, N);
}

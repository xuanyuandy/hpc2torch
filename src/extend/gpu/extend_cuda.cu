#include <cuda.h>
#include <cub/cub.cuh>

// template <typename T>
// __global__ void blockExtendKernel(T const *input, T *output, int num, int dimsize, int stride, int numThread)
// {
//     int otherIdx = threadIdx.x + blockIdx.y * blockDim.x; // 范围是0到(num + 1)BCD / numThread
//     int frontIdx = blockIdx.x;
//     for (int index = 0; index < numThread; index++)
//     {
//         int idx = otherIdx * numThread + index; // j(CD) + k(D) + s, j < (num + 1)B;
//         if (idx >= (num + 1) * dimsize * stride)
//         {
//             break;
//         }
//         int behindIdx = idx % stride;            // k(D) + s
//         int dimIdx = idx / ((num + 1) * stride); // j(CD) / [(num + 1) CD]
//         int outputIdx = frontIdx * (num + 1) * dimsize * stride + idx;
//         int inputIdx = frontIdx * dimsize * stride + dimIdx * stride + behindIdx;
//         output[outputIdx] = input[inputIdx];
//     }
// }
// template <typename T>
// __global__ void warpExtendKernel(T const *input, T *output, int frontsize, int num, int dimsize, int stride, int numThread)
// {
//     int otherIdx = threadIdx.y + blockIdx.y * blockDim.y;
//     int frontIdx = threadIdx.x + blockIdx.x * blockDim.x;
//     if (frontIdx >= frontsize)
//     {
//         return;
//     }
//     for (int index = 0; index < numThread; index++)
//     {
//         int idx = otherIdx * numThread + index; // j(CD) + k(D) + s, j < (num + 1)B;
//         if (idx >= (num + 1) * dimsize * stride)
//         {
//             break;
//         }
//         int behindIdx = idx % stride;            // k(D) + s
//         int dimIdx = idx / ((num + 1) * stride); // j(CD) / [(num + 1) CD]
//         int outputIdx = frontIdx * (num + 1) * dimsize * stride + idx;
//         int inputIdx = frontIdx * dimsize * stride + dimIdx * stride + behindIdx;
//         output[outputIdx] = input[inputIdx];
//     }
// }
template <typename T>
__global__ void _extend_kernel(T *in, T *out, int blockSize,
                               int blockSizeOuter, int oSize)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= oSize)
        return;

    int stride = blockDim.x * gridDim.x;
    while (index < oSize)
    {
        auto iIdx = index % blockSize + index / blockSizeOuter * blockSize;
        out[index] = in[iIdx];
        index += stride;
    }
}
template <typename T>
void extendLaunch(void const *input, void *output, int num, int frontsize, int dimsize, int stride)
{
    int oSize = frontsize * (num + 1) * dimsize * stride; // dimsize和stride都是input的
    int blockSize = dimsize * stride;
    int blockSizeOuter = (num + 1) * blockSize;

    int blocksize = 32 * 16;
    int gridsize = (oSize + blocksize - 1) / blocksize;
    _extend_kernel<T><<<gridsize, blocksize>>>(
        (T *)input, (T *)output, blockSize,
        blockSizeOuter, oSize);
    // 假设input = [A, B, C, D], axis = 1, output = [A, B(num + 1), C, D]
    // othersize =  BCD, frontsize = A
    // int othersize = dimsize * stride * (num + 1);
    // int numThread = 2;
    // int count = othersize / numThread;
    // if (count > 1024)
    // {
    //     int BLOCK_DIM = 1024;
    //     int num_block_x = frontsize;
    //     int num_block_y = (othersize + BLOCK_DIM * numThread - 1) / (BLOCK_DIM * numThread);
    //     dim3 block_dim(BLOCK_DIM, 1, 1);
    //     dim3 grid_dim(num_block_x, num_block_y, 1);
    //     blockExtendKernel<T>
    //         <<<grid_dim, block_dim>>>((T *)input, (T *)output, num, dimsize, stride, numThread);
    // }
    // else if (count > 31)
    // {
    //     int BLOCK_DIM_x = 32;
    //     int BLOCK_DIM_y = 32;
    //     int num_block_x = (frontsize + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    //     int num_block_y = (othersize + BLOCK_DIM_y * numThread - 1) / (BLOCK_DIM_y * numThread);
    //     dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    //     dim3 grid_dim(num_block_x, num_block_y, 1);

    //     warpExtendKernel<T>
    //         <<<grid_dim, block_dim>>>((T *)input, (T *)output, frontsize, num, dimsize, stride, numThread);
    // }
    // else if (count > 15)
    // {
    //     int BLOCK_DIM_x = 64;
    //     int BLOCK_DIM_y = 16;
    //     int num_block_x = (frontsize + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    //     int num_block_y = (othersize + BLOCK_DIM_y * numThread - 1) / (BLOCK_DIM_y * numThread);
    //     dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    //     dim3 grid_dim(num_block_x, num_block_y, 1);

    //     warpExtendKernel<T>
    //         <<<grid_dim, block_dim>>>((T *)input, (T *)output, frontsize, num, dimsize, stride, numThread);
    // }
    // else
    // {
    //     int BLOCK_DIM_x = 128;
    //     int BLOCK_DIM_y = 8;
    //     int num_block_x = (frontsize + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    //     int num_block_y = (othersize + BLOCK_DIM_y * numThread - 1) / (BLOCK_DIM_y * numThread);
    //     dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    //     dim3 grid_dim(num_block_x, num_block_y, 1);

    //     warpExtendKernel<T>
    //         <<<grid_dim, block_dim>>>((T *)input, (T *)output, frontsize, num, dimsize, stride, numThread);
    // }
}
extern "C" void extend_nv(void const *input, void *output, int num, int frontsize, int dimsize, int stride, int byteSize)
{
    if (byteSize == 2)
    {
        extendLaunch<half>(input, output, num, frontsize, dimsize, stride);
    }
    else if (byteSize == 4)
    {
        extendLaunch<float>(input, output, num, frontsize, dimsize, stride);
    }
}
#include "acl/acl.h"
#include "aclnnop/level2/aclnn_gemm.h"
#include "aclnnop/aclnn_matmul.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

template <typename T>
void matmulAclnnDevice(void *aData, void *bData, void *cData,
                       int *a_shape, int *b_shape, int *c_shape,
                       int aDim, int bDim, int cDim,
                       float alpha, float beta,
                       aclrtStream &stream)
{
    aclDataType dataType;
    if (sizeof(T) == 2)
    {
        dataType = aclDataType::ACL_FLOAT16;
    }
    else if (sizeof(T) == 4)
    {
        dataType = aclDataType::ACL_FLOAT;
    }
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    std::vector<int64_t> inputADim(aDim);       // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputAStride(aDim, 1); // 初始化为1
    std::vector<int64_t> inputBDim(bDim);
    std::vector<int64_t> inputBStride(bDim, 1);
    std::vector<int64_t> outputDim(cDim);
    std::vector<int64_t> outputStride(cDim, 1);

    for (int i = aDim - 1; i >= 0; i--)
    {
        inputADim[i] = int64_t(a_shape[i]);
        if (i < aDim - 1)
        {
            inputAStride[i] = inputADim[i + 1] * inputAStride[i + 1];
        }
    }
    for (int i = bDim - 1; i >= 0; i--)
    {
        inputBDim[i] = int64_t(b_shape[i]);
        if (i < bDim - 1)
        {
            inputBStride[i] = inputBDim[i + 1] * inputBStride[i + 1];
        }
    }
    for (int i = cDim - 1; i >= 0; i--)
    {
        outputDim[i] = int64_t(c_shape[i]);

        if (i < cDim - 1)
        {
            outputStride[i] = outputDim[i + 1] * outputStride[i + 1];
        }
    }
    auto inputATensor =
        aclCreateTensor(inputADim.data(), inputADim.size(), dataType,
                        inputAStride.data(), 0, format,
                        inputADim.data(), inputADim.size(), aData); // const aclTensor *inputATensor
    auto inputBTensor =
        aclCreateTensor(inputBDim.data(), inputBDim.size(), dataType,
                        inputBStride.data(), 0, format,
                        inputBDim.data(), inputBDim.size(), bData);
    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), cData);
    // 下面开始正式计算
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    aclnnStatus ret;
    int8_t cubeMathType = 1;
    aclTensor *biasTensor = NULL;
    if (aDim == 2 && bDim == 2 && cDim == 2)
    {
        biasTensor =
            aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                            outputStride.data(), 0, format,
                            outputDim.data(), outputDim.size(), cData);
        int64_t transA = 0;
        int64_t transB = 0;
        ret = aclnnGemmGetWorkspaceSize(inputATensor, inputBTensor, biasTensor,
                                        alpha, beta, transA, transB, outputTensor,
                                        cubeMathType, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnGemmGetWorkspaceSize failed. ERROR: %d\n", ret);
        }
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0)
        {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
            {
                printf("allocate workspace failed. ERROR: %d\n", ret);
            }
        }
        ret = aclnnGemm(workspaceAddr, workspaceSize, executor, stream);
    }
    else
    {
        ret = aclnnMatmulGetWorkspaceSize(
            inputATensor, inputBTensor, outputTensor,
            cubeMathType, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret);
        }
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0)
        {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
            {
                printf("allocate workspace failed. ERROR: %d\n", ret);
            }
        }
        ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    }

    // ret = aclnnMatmulGetWorkspaceSize(
    //     inputATensor, inputBTensor, outputTensor,
    //     cubeMathType, &workspaceSize, &executor);
    // if (ret != ACL_SUCCESS)
    // {
    //     printf("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret);
    // }
    // void *workspaceAddr = nullptr;
    // if (workspaceSize > 0)
    // {
    //     ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    //     if (ret != ACL_SUCCESS)
    //     {
    //         printf("allocate workspace failed. ERROR: %d\n", ret);
    //     }
    // }
    // ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS)
    {
        printf("aclnnMatmul failed. ERROR: %d\n", ret);
    }
    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    aclDestroyTensor(inputATensor);
    aclDestroyTensor(inputBTensor);
    aclDestroyTensor(outputTensor);
}
template <typename T>
void matmulAclnn(void *aData, void *bData, void *cData,
                 int *a_shape, int *b_shape, int *c_shape,
                 int aDim, int bDim, int cDim,
                 float alpha, float beta)
{
    int32_t deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS)
    {
        printf("Init acl failed. ERROR: %d\n", ret);
    }

    matmulAclnnDevice<T>(aData, bData, cData,
                         a_shape, b_shape, c_shape,
                         aDim, bDim, cDim,
                         alpha, beta, stream);
    Finalize(deviceId, stream);
}

extern "C" void matmul_aclnn(void *aData, void *bData, void *cData,
                             int *a_shape, int *b_shape, int *c_shape,
                             int aDim, int bDim, int cDim,
                             float alpha, float beta, int byteSize)
{
    if (byteSize == 2)
    {
        matmulAclnn<uint16_t>(aData, bData, cData,
                              a_shape, b_shape, c_shape,
                              aDim, bDim, cDim,
                              alpha, beta);
    }
    else if (byteSize == 4)
    {
        matmulAclnn<float>(aData, bData, cData,
                           a_shape, b_shape, c_shape,
                           aDim, bDim, cDim,
                           alpha, beta);
    }
}

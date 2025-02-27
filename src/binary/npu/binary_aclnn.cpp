#include "acl/acl.h"
#include "aclnnop/aclnn_maximum.h"
#include "aclnnop/level2/aclnn_add.h"
#include "aclnnop/level2/aclnn_div.h"
#include "aclnnop/level2/aclnn_mul.h"
#include "aclnnop/level2/aclnn_pow_tensor_tensor.h"
#include "aclnnop/level2/aclnn_sub.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"
struct BinaryMode
{
    enum Mode
    {
        // Arithmetic operations:
        Add,
        Subtract,
        Multiply,
        Divide,
        Pow,
        Max,

        Count, ///< Number of binary operation types (marker for counting purposes).
    };

    // This static constant holds the total number of defined binary operations.
    static const size_t numBinaryMode = Count;
};
template <typename T>
void binaryAclnnDevice(void *aData, void *bData, void *cData,
                       int *aShape, int *bShape, int *cShape,
                       int aDim, int bDim, int cDim,
                       BinaryMode::Mode mode, aclrtStream &stream)
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
        inputADim[i] = int64_t(aShape[i]);

        if (i < aDim - 1)
        {
            inputAStride[i] = inputADim[i + 1] * inputAStride[i + 1];
        }
    }
    for (int i = bDim - 1; i >= 0; i--)
    {
        inputBDim[i] = int64_t(bShape[i]);
        if (i < bDim - 1)
        {
            inputBStride[i] = inputBDim[i + 1] * inputBStride[i + 1];
        }
    }
    for (int i = cDim - 1; i >= 0; i--)
    {
        outputDim[i] = int64_t(cShape[i]);
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
    if (mode == BinaryMode::Pow)
    {
        ret = aclnnPowTensorTensorGetWorkspaceSize(
            inputATensor, inputBTensor, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnPowGetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor,
                                   stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnPow failed. ERROR: %d\n", ret);
        }
    }
    else if (mode == BinaryMode::Divide)
    {
        ret = aclnnDivGetWorkspaceSize(
            inputATensor, inputBTensor, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnDivGetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnDiv(workspaceAddr, workspaceSize, executor,
                       stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnDiv failed. ERROR: %d\n", ret);
        }
    }
    else if (mode == BinaryMode::Multiply)
    {
        ret = aclnnMulGetWorkspaceSize(
            inputATensor, inputBTensor, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnMul(workspaceAddr, workspaceSize, executor,
                       stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnMul failed. ERROR: %d\n", ret);
        }
    }
    else if (mode == BinaryMode::Max)
    {
        ret = aclnnMaximumGetWorkspaceSize(
            inputATensor, inputBTensor, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnMaximumGetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnMaximum(workspaceAddr, workspaceSize, executor,
                           stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnMaximum failed. ERROR: %d\n", ret);
        }
    }
    else if (mode == BinaryMode::Add)
    {
        float bAlpha = 1.0f;

        auto alpha = aclCreateScalar(&bAlpha, ACL_FLOAT);
        ret = aclnnAddGetWorkspaceSize(
            inputATensor, inputBTensor, alpha, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnAdd(workspaceAddr, workspaceSize, executor,
                       stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnAdd failed. ERROR: %d\n", ret);
        }
    }
    else if (mode == BinaryMode::Subtract)
    {
        float bAlpha = 1.0f;

        auto alpha = aclCreateScalar(&bAlpha, ACL_FLOAT);
        ret = aclnnSubGetWorkspaceSize(
            inputATensor, inputBTensor, alpha, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnSub(workspaceAddr, workspaceSize, executor,
                       stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnAdd failed. ERROR: %d\n", ret);
        }
    }

    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    aclDestroyTensor(inputATensor);
    aclDestroyTensor(inputBTensor);
    aclDestroyTensor(outputTensor);

    // aclDestroyAclOpExecutor(executor);//似乎不支持destroy，一旦destroy测试报错
}
template <typename T>
void binaryAclnn(void *aData, void *bData, void *cData,
                 int *aShape, int *bShape, int *cShape,
                 int aDim, int bDim, int cDim,
                 BinaryMode::Mode mode)
{
    // static int count = 0;
    // printf("count is %d \n", count);
    int32_t deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS)
    {
        printf("Init acl failed. ERROR: %d\n", ret);
    }

    binaryAclnnDevice<T>(aData, bData, cData,
                         aShape, bShape, cShape,
                         aDim, bDim, cDim,
                         mode, stream);
    Finalize(deviceId, stream);
}
extern "C" void max_aclnn(void *aData, void *bData, void *cData,
                          int *aShape, int *bShape, int *cShape,
                          int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        binaryAclnn<uint16_t>(aData, bData, cData,
                              aShape, bShape, cShape,
                              aDim, bDim, cDim, BinaryMode::Max);
    }
    else if (byteSize == 4)
    {
        binaryAclnn<float>(aData, bData, cData,
                           aShape, bShape, cShape,
                           aDim, bDim, cDim, BinaryMode::Max);
    }
}
extern "C" void mul_aclnn(void *aData, void *bData, void *cData,
                          int *aShape, int *bShape, int *cShape,
                          int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        binaryAclnn<uint16_t>(aData, bData, cData,
                              aShape, bShape, cShape,
                              aDim, bDim, cDim, BinaryMode::Multiply);
    }
    else if (byteSize == 4)
    {
        binaryAclnn<float>(aData, bData, cData,
                           aShape, bShape, cShape,
                           aDim, bDim, cDim, BinaryMode::Multiply);
    }
}
extern "C" void sub_aclnn(void *aData, void *bData, void *cData,
                          int *aShape, int *bShape, int *cShape,
                          int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        binaryAclnn<uint16_t>(aData, bData, cData,
                              aShape, bShape, cShape,
                              aDim, bDim, cDim, BinaryMode::Subtract);
    }
    else if (byteSize == 4)
    {
        binaryAclnn<float>(aData, bData, cData,
                           aShape, bShape, cShape,
                           aDim, bDim, cDim, BinaryMode::Subtract);
    }
}
extern "C" void add_aclnn(void *aData, void *bData, void *cData,
                          int *aShape, int *bShape, int *cShape,
                          int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        binaryAclnn<uint16_t>(aData, bData, cData,
                              aShape, bShape, cShape,
                              aDim, bDim, cDim, BinaryMode::Add);
    }
    else if (byteSize == 4)
    {
        binaryAclnn<float>(aData, bData, cData,
                           aShape, bShape, cShape,
                           aDim, bDim, cDim, BinaryMode::Add);
    }
}
extern "C" void div_aclnn(void *aData, void *bData, void *cData,
                          int *aShape, int *bShape, int *cShape,
                          int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        binaryAclnn<uint16_t>(aData, bData, cData,
                              aShape, bShape, cShape,
                              aDim, bDim, cDim, BinaryMode::Divide);
    }
    else if (byteSize == 4)
    {
        binaryAclnn<float>(aData, bData, cData,
                           aShape, bShape, cShape,
                           aDim, bDim, cDim, BinaryMode::Divide);
    }
}
extern "C" void pow_aclnn(void *aData, void *bData, void *cData,
                          int *aShape, int *bShape, int *cShape,
                          int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        binaryAclnn<uint16_t>(aData, bData, cData,
                              aShape, bShape, cShape,
                              aDim, bDim, cDim, BinaryMode::Pow);
    }
    else if (byteSize == 4)
    {
        binaryAclnn<float>(aData, bData, cData,
                           aShape, bShape, cShape,
                           aDim, bDim, cDim, BinaryMode::Pow);
    }
}

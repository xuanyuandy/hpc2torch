#include "acl/acl.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"
#include "aclnnop/aclnn_erf.h"
#include "aclnnop/level2/aclnn_abs.h"
#include "aclnnop/level2/aclnn_acos.h"
#include "aclnnop/level2/aclnn_atan.h"
#include "aclnnop/level2/aclnn_ceil.h"
#include "aclnnop/level2/aclnn_cos.h"
#include "aclnnop/level2/aclnn_exp.h"
#include "aclnnop/level2/aclnn_floor.h"
#include "aclnnop/level2/aclnn_gelu.h"
#include "aclnnop/level2/aclnn_hardswish.h"
#include "aclnnop/level2/aclnn_hardsigmoid.h"
#include "aclnnop/level2/aclnn_leaky_relu.h"
#include "aclnnop/level2/aclnn_neg.h"
#include "aclnnop/level2/aclnn_reciprocal.h"
#include "aclnnop/level2/aclnn_relu.h"
#include "aclnnop/level2/aclnn_round.h"
#include "aclnnop/level2/aclnn_sigmoid.h"
#include "aclnnop/level2/aclnn_sin.h"
#include "aclnnop/level2/aclnn_sqrt.h"
#include "aclnnop/level2/aclnn_tanh.h"
#include "aclnnop/aclnn_prelu.h"

struct UnaryMode
{
    enum Mode
    {
        Relu,
        Gelu,
        PRelu, // 这个需要特殊处理
        LeakyRelu,
        Sigmoid,
        Round,
        HardSigmoid,
        HardSwish,
        Count, ///< Number of unary operation types (marker for counting purposes).
    };

    // This static constant holds the total number of defined unary operations.
    static const size_t numUnaryMode = Count;
};
template <typename T>
void unaryAclnnDevice(void *aData, void *cData, int *aShape, int *cShape,
                      int aDim, int cDim, UnaryMode::Mode mode, float coef,
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

    std::vector<int64_t> inputDim(aDim);       // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputStride(aDim, 1); // 初始化为1

    std::vector<int64_t> outputDim(cDim);
    std::vector<int64_t> outputStride(cDim, 1);

    for (int i = aDim - 1; i >= 0; i--)
    {
        inputDim[i] = int64_t(aShape[i]);

        if (i < aDim - 1)
        {
            inputStride[i] = inputDim[i + 1] * inputStride[i + 1];
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
    auto inputTensor =
        aclCreateTensor(inputDim.data(), inputDim.size(), dataType,
                        inputStride.data(), 0, format,
                        inputDim.data(), inputDim.size(), aData); // const aclTensor *inputTensor
    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), cData);
    // 下面开始正式计算
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    aclnnStatus ret;
    if (mode == UnaryMode::Sigmoid)
    {
        ret = aclnnSigmoidGetWorkspaceSize(
            inputTensor, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnSigmoidGetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnSigmoid(workspaceAddr, workspaceSize, executor,
                           stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnSigmoid failed. ERROR: %d\n", ret);
        }
    }
    else if (mode == UnaryMode::HardSwish)
    {
        ret = aclnnHardswishGetWorkspaceSize(
            inputTensor, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnHardSwishGetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnHardswish(workspaceAddr, workspaceSize, executor,
                             stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnHardSwish failed. ERROR: %d\n", ret);
        }
    }
    else if (mode == UnaryMode::Gelu)
    {
        ret = aclnnGeluGetWorkspaceSize(
            inputTensor, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnGeluGetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnGelu(workspaceAddr, workspaceSize, executor,
                        stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnGelu failed. ERROR: %d\n", ret);
        }
    }
    else if (mode == UnaryMode::Round)
    {
        ret = aclnnRoundGetWorkspaceSize(
            inputTensor, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnRoundGetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnRound(workspaceAddr, workspaceSize, executor,
                         stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnRound failed. ERROR: %d\n", ret);
        }
    }
    else if (mode == UnaryMode::HardSigmoid)
    {
        ret = aclnnHardsigmoidGetWorkspaceSize(
            inputTensor, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnHardSigmoidGetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnHardsigmoid(workspaceAddr, workspaceSize, executor,
                               stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnHardSigmoid failed. ERROR: %d\n", ret);
        }
    }
    else if (mode == UnaryMode::Relu)
    {
        ret = aclnnReluGetWorkspaceSize(
            inputTensor, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnReluGetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnRelu(workspaceAddr, workspaceSize, executor,
                        stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnRelu failed. ERROR: %d\n", ret);
        }
    }
    else if (mode == UnaryMode::LeakyRelu)
    {
        aclScalar *negativeSlope = nullptr;
        negativeSlope =
            aclCreateScalar(&coef, aclDataType::ACL_FLOAT);
        ret = aclnnLeakyReluGetWorkspaceSize(
            inputTensor, negativeSlope, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnLeakyReluGetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnLeakyRelu(workspaceAddr, workspaceSize, executor,
                             stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnLeakyRelu failed. ERROR: %d\n", ret);
        }
        aclDestroyScalar(negativeSlope);
    }
    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }
    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);
}
template <typename T>
void pReluAclnnDevice(void *aData, void *bData, void *cData, int *aShape, int *bShape, int *cShape,
                      int aDim, int bDim, int cDim,
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

    std::vector<int64_t> inputDim(aDim);       // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputStride(aDim, 1); // 初始化为1
    std::vector<int64_t> weightDim(bDim);
    std::vector<int64_t> weightStride(bDim, 1);
    std::vector<int64_t> outputDim(cDim);
    std::vector<int64_t> outputStride(cDim, 1);

    for (int i = aDim - 1; i >= 0; i--)
    {
        inputDim[i] = int64_t(aShape[i]);

        if (i < aDim - 1)
        {
            inputStride[i] = inputDim[i + 1] * inputStride[i + 1];
        }
    }
    for (int i = bDim - 1; i >= 0; i--)
    {
        weightDim[i] = int64_t(bShape[i]);

        if (i < bDim - 1)
        {
            weightStride[i] = weightDim[i + 1] * weightStride[i + 1];
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
    auto inputTensor =
        aclCreateTensor(inputDim.data(), inputDim.size(), dataType,
                        inputStride.data(), 0, format,
                        inputDim.data(), inputDim.size(), aData); // const aclTensor *inputTensor
    auto weightTensor =
        aclCreateTensor(weightDim.data(), weightDim.size(), dataType,
                        weightStride.data(), 0, format,
                        weightDim.data(), weightDim.size(), bData);
    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), cData);
    // 下面开始正式计算
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    aclnnStatus ret;

    ret = aclnnPreluGetWorkspaceSize(
        inputTensor, weightTensor, outputTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS)
    {
        printf("aclnnPreluGetWorkspaceSize failed. ERROR: %d\n", ret);
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

    ret = aclnnPrelu(workspaceAddr, workspaceSize, executor,
                     stream);
    if (ret != ACL_SUCCESS)
    {
        printf("aclnnPrelu failed. ERROR: %d\n", ret);
    }

    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }
    aclDestroyTensor(inputTensor);
    aclDestroyTensor(weightTensor);
    aclDestroyTensor(outputTensor);
}
template <typename T>
void unaryAclnn(void *aData, void *cData, int *aShape, int *cShape,
                int aDim, int cDim, UnaryMode::Mode mode, float coef)
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

    unaryAclnnDevice<T>(aData, cData, aShape, cShape,
                        aDim, cDim,
                        mode, coef, stream);
    Finalize(deviceId, stream);
}
template <typename T>
void pReluAclnn(void *aData, void *bData, void *cData, int *aShape, int *bShape, int *cShape,
                int aDim, int bDim, int cDim)
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

    pReluAclnnDevice<T>(aData, bData, cData, aShape, bShape, cShape,
                        aDim, bDim, cDim, stream);
    Finalize(deviceId, stream);
}
extern "C" void relu_aclnn(void *aData, void *cData, int *aShape, int *cShape,
                           int aDim, int cDim,
                           int byteSize)
{
    if (byteSize == 2)
    {
        unaryAclnn<uint16_t>(aData, cData, aShape, cShape,
                             aDim, cDim,
                             UnaryMode::Relu, 0.0f);
    }
    else if (byteSize == 4)
    {
        unaryAclnn<float>(aData, cData, aShape, cShape,
                          aDim, cDim,
                          UnaryMode::Relu, 0.0f);
    }
}
extern "C" void gelu_aclnn(void *aData, void *cData, int *aShape, int *cShape,
                           int aDim, int cDim,
                           int byteSize)
{
    if (byteSize == 2)
    {
        unaryAclnn<uint16_t>(aData, cData, aShape, cShape,
                             aDim, cDim,
                             UnaryMode::Gelu, 0.0f);
    }
    else if (byteSize == 4)
    {
        unaryAclnn<float>(aData, cData, aShape, cShape,
                          aDim, cDim,
                          UnaryMode::Gelu, 0.0f);
    }
}
extern "C" void sigmoid_aclnn(void *aData, void *cData, int *aShape, int *cShape,
                              int aDim, int cDim,
                              int byteSize)
{
    if (byteSize == 2)
    {
        unaryAclnn<uint16_t>(aData, cData, aShape, cShape,
                             aDim, cDim,
                             UnaryMode::Sigmoid, 0.0f);
    }
    else if (byteSize == 4)
    {
        unaryAclnn<float>(aData, cData, aShape, cShape,
                          aDim, cDim,
                          UnaryMode::Sigmoid, 0.0f);
    }
}
extern "C" void hardswish_aclnn(void *aData, void *cData, int *aShape, int *cShape,
                                int aDim, int cDim,
                                int byteSize)
{
    if (byteSize == 2)
    {
        unaryAclnn<uint16_t>(aData, cData, aShape, cShape,
                             aDim, cDim,
                             UnaryMode::HardSwish, 0.0f);
    }
    else if (byteSize == 4)
    {
        unaryAclnn<float>(aData, cData, aShape, cShape,
                          aDim, cDim,
                          UnaryMode::HardSwish, 0.0f);
    }
}
extern "C" void hardsigmoid_aclnn(void *aData, void *cData, int *aShape, int *cShape,
                                  int aDim, int cDim,
                                  int byteSize)
{
    if (byteSize == 2)
    {
        unaryAclnn<uint16_t>(aData, cData, aShape, cShape,
                             aDim, cDim,
                             UnaryMode::HardSigmoid, 0.0f);
    }
    else if (byteSize == 4)
    {
        unaryAclnn<float>(aData, cData, aShape, cShape,
                          aDim, cDim,
                          UnaryMode::HardSigmoid, 0.0f);
    }
}
extern "C" void leakyRelu_aclnn(void *aData, void *cData, int *aShape, int *cShape,
                                int aDim, int cDim, float coef,
                                int byteSize)
{
    if (byteSize == 2)
    {
        unaryAclnn<uint16_t>(aData, cData, aShape, cShape,
                             aDim, cDim,
                             UnaryMode::LeakyRelu, coef);
    }
    else if (byteSize == 4)
    {
        unaryAclnn<float>(aData, cData, aShape, cShape,
                          aDim, cDim,
                          UnaryMode::LeakyRelu, coef);
    }
}
extern "C" void round_aclnn(void *aData, void *cData, int *aShape, int *cShape,
                            int aDim, int cDim,
                            int byteSize)
{ // 这个函数没有coef, alpha, beta, sliceDim, gamma, scale参数，这里随便填充即可
    if (byteSize == 2)
    {
        unaryAclnn<uint16_t>(aData, cData, aShape, cShape,
                             aDim, cDim,
                             UnaryMode::Round, 0.0f);
    }
    else if (byteSize == 4)
    {
        unaryAclnn<float>(aData, cData, aShape, cShape,
                          aDim, cDim,
                          UnaryMode::Round, 0.0f);
    }
}
extern "C" void pRelu_aclnn(void *aData, void *bData, void *cData, int *aShape, int *bShape, int *cShape,
                            int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        pReluAclnn<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                             aDim, bDim, cDim);
    }
    else if (byteSize == 4)
    {
        pReluAclnn<float>(aData, bData, cData, aShape, bShape, cShape,
                          aDim, bDim, cDim);
    }
}

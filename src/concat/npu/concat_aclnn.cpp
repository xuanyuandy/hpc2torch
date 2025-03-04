#include "acl/acl.h"
#include "aclnnop/aclnn_cat.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

template <typename T>
void concatAclnnDevice(void **input, void *output, int **inputShape, int *outputShape,
                       int num, int axis, int ndim, aclrtStream &stream)
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
    std::vector<int64_t> outputDim(ndim); // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> outputStride(ndim, 1);
    for (int i = ndim - 1; i >= 0; i--)
    {
        outputDim[i] = int64_t(outputShape[i]);
        if (i < ndim - 1)
        {
            outputStride[i] = outputDim[i + 1] * outputStride[i + 1];
        }
    }
    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), output); // const aclTensor *inputTensor
    std::vector<aclTensor *> inputsData{};

    for (int i = 0; i < num; ++i)
    {

        std::vector<int64_t> inputDim(ndim);
        std::vector<int64_t> inputStride(ndim, 1);
        for (int j = ndim - 1; j >= 0; j--)
        {
            inputDim[j] = int64_t(inputShape[i][j]);
            if (j < ndim - 1)
            {
                inputStride[j] = inputDim[j + 1] * inputStride[j + 1];
            }
        }

        auto tmpTensor =
            aclCreateTensor(inputDim.data(), inputDim.size(), dataType,
                            inputStride.data(), 0, format,
                            inputDim.data(), inputDim.size(), input[i]);

        inputsData.push_back(tmpTensor);
    }
    aclTensorList *tensorList =
        aclCreateTensorList(inputsData.data(), inputsData.size());
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    auto ret = aclnnCatGetWorkspaceSize(
        tensorList, int64_t(axis), outputTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS)
    {
        printf("aclnnCatGetWorkspaceSize failed. ERROR: %d\n", ret);
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

    ret = aclnnCat(workspaceAddr, workspaceSize, executor,
                   stream);
    if (ret != ACL_SUCCESS)
    {
        printf("aclnnCat failed. ERROR: %d\n", ret);
    }
    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }
    aclDestroyTensorList(tensorList);
    aclDestroyTensor(outputTensor);
    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
}
template <typename T>
void concatAclnn(void **input, void *output, int **inputShape, int *outputShape,
                 int num, int axis, int ndim)
{
    int32_t deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS)
    {
        printf("Init acl failed. ERROR: %d\n", ret);
    }

    concatAclnnDevice<T>(input, output, inputShape, outputShape,
                         num, axis, ndim, stream);
    Finalize(deviceId, stream);
}
extern "C" void concat_aclnn(void **input, void *output, int **inputShape, int *outputShape,
                             int num, int axis, int ndim, int byteSize)
{
    if (byteSize == 4)
    {
        concatAclnn<float>(input, output, inputShape, outputShape,
                           num, axis, ndim);
    }
    else if (byteSize == 2)
    {
        concatAclnn<uint16_t>(input, output, inputShape, outputShape,
                              num, axis, ndim);
    }
}

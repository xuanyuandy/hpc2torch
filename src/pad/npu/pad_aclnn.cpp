#include "acl/acl.h"
#include "aclnnop/aclnn_constant_pad_nd.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

template <typename T>
void padAclnnDevice(void *input, int *hostPad, void *hostValue, void *output,
                    int *inputShape, int *outputShape, int ndim,
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
    std::vector<int64_t> inputDim(ndim);       // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputStride(ndim, 1); // 初始化为1

    std::vector<int64_t> outputDim(ndim);
    std::vector<int64_t> outputStride(ndim, 1);

    for (int i = ndim - 1; i >= 0; i--)
    {
        inputDim[i] = int64_t(inputShape[i]);
        outputDim[i] = int64_t(outputShape[i]);
        if (i < ndim - 1)
        {
            inputStride[i] = inputDim[i + 1] * inputStride[i + 1];
            outputStride[i] = outputDim[i + 1] * outputStride[i + 1];
        }
    }
    auto inputTensor =
        aclCreateTensor(inputDim.data(), inputDim.size(), dataType,
                        inputStride.data(), 0, format,
                        inputDim.data(), inputDim.size(), input); // const aclTensor *inputTensor

    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), output);
    std::vector<int64_t> pads(2 * ndim); // 注意寒武纪的填充方式和昇腾不一样
    for (int i = 0; i < ndim; i++)
    { // 注意昇腾我编写的填充方式和InfiniTensor不一样
        pads[2 * i] = int64_t(hostPad[2 * i]);
        pads[2 * i + 1] = int64_t(hostPad[2 * i + 1]);
    }
    aclIntArray *padding = aclCreateIntArray(pads.data(), 2 * ndim);

    auto value = aclCreateScalar(hostValue, dataType);

    // 下面开始正式计算
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    auto ret = aclnnConstantPadNdGetWorkspaceSize(
        inputTensor, padding, value, outputTensor, &workspaceSize,
        &executor);
    if (ret != ACL_SUCCESS)
    {
        printf("aclnnConstantPadNdGetWorkspaceSize failed. ERROR: %d\n", ret);
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
    ret = aclnnConstantPadNd(workspaceAddr, workspaceSize, executor,
                             stream);
    if (ret != ACL_SUCCESS)
    {
        printf("aclnnConstantPadNd failed. ERROR: %d\n", ret);
    }
    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    aclDestroyTensor(inputTensor);
    aclDestroyIntArray(padding);
    aclDestroyScalar(value);
    aclDestroyTensor(outputTensor);
    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
}
template <typename T>
void padAclnn(void *input, int *hostPad, void *hostValue, void *output,
              int *inputShape, int *outputShape, int ndim)
{
    int32_t deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS)
    {
        printf("Init acl failed. ERROR: %d\n", ret);
    }

    padAclnnDevice<T>(input, hostPad, hostValue, output,
                      inputShape, outputShape, ndim, stream);
    Finalize(deviceId, stream);
}
extern "C" void pad_aclnn(void *input, int *hostPad, void *hostValue, void *output,
                          int *inputShape, int *outputShape, int ndim, int byteSize)
{
    if (byteSize == 4)
    {
        padAclnn<float>(input, hostPad, hostValue, output,
                        inputShape, outputShape, ndim);
    }
    else if (byteSize == 2)
    {
        padAclnn<uint16_t>(input, hostPad, hostValue, output,
                           inputShape, outputShape, ndim);
    }
}

#include "acl/acl.h"
#include "aclnnop/aclnn_gather_v2.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

template <typename T>
void extendAclnnDevice(void *input, void *output,
                       int *x_shape, int *y_shape,
                       int ndim, int num, int axis, aclrtStream &stream)
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
    int w_ndim = 1;
    int dimsize = x_shape[(axis + ndim) % ndim];
    int IndicesSize = dimsize * (num + 1);
    int *hostIndices = (int *)malloc(IndicesSize * sizeof(int));
    for (int i = 0; i < IndicesSize; i++)
    {
        hostIndices[i] = i % dimsize;
    }
    int *deviceIndices;
    aclrtMalloc((void **)&deviceIndices, IndicesSize * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(deviceIndices, IndicesSize * sizeof(int), hostIndices, IndicesSize * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE);

    std::vector<int64_t> inputDim(ndim);       // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputStride(ndim, 1); // 初始化为1
    std::vector<int64_t> indicesDim(w_ndim, IndicesSize);
    std::vector<int64_t> indicesStride(w_ndim, 1);
    std::vector<int64_t> outputDim(ndim);
    std::vector<int64_t> outputStride(ndim, 1);

    for (int i = ndim - 1; i >= 0; i--)
    {
        inputDim[i] = int64_t(x_shape[i]);
        if (i < ndim - 1)
        {
            inputStride[i] = inputDim[i + 1] * inputStride[i + 1];
        }
    }

    for (int i = ndim - 1; i >= 0; i--)
    {
        outputDim[i] = int64_t(y_shape[i]);

        if (i < ndim - 1)
        {
            outputStride[i] = outputDim[i + 1] * outputStride[i + 1];
        }
    }
    auto inputTensor =
        aclCreateTensor(inputDim.data(), inputDim.size(), dataType,
                        inputStride.data(), 0, format,
                        inputDim.data(), inputDim.size(), input); // const aclTensor *inputTensor
    auto indicesTensor =
        aclCreateTensor(indicesDim.data(), indicesDim.size(), aclDataType::ACL_INT32,
                        indicesStride.data(), 0, format,
                        indicesDim.data(), indicesDim.size(), deviceIndices);
    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), output);
    // 下面开始正式计算
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    auto ret = aclnnGatherV2GetWorkspaceSize(inputTensor, int64_t(axis), indicesTensor,
                                             outputTensor, &workspaceSize, &executor);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnGatherV2GetWorkspaceSize failed. ERROR: %d\n", ret);
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

    ret = aclnnGatherV2(workspaceAddr, workspaceSize, executor,
                        stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnGatherV2 failed. ERROR: %d\n", ret);
    }
    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    aclDestroyTensor(inputTensor);
    aclDestroyTensor(indicesTensor);
    aclDestroyTensor(outputTensor);
    free(hostIndices);
    aclrtFree(deviceIndices);
    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
    // aclDestroyAclOpExecutor(executor);//似乎不支持destroy，一旦destroy测试报错
}
template <typename T>
void extendAclnn(void *input, void *output,
                 int *x_shape, int *y_shape,
                 int ndim, int num, int axis)
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

    extendAclnnDevice<T>(input, output,
                         x_shape, y_shape,
                         ndim, num, axis, stream);
    Finalize(deviceId, stream);
}
extern "C" void extend_aclnn(void *input, void *output,
                             int *x_shape, int *y_shape,
                             int ndim, int num, int axis, int byteSize)
{
    if (byteSize == 4)
    {
        extendAclnn<float>(input, output,
                           x_shape, y_shape,
                           ndim, num, axis);
    }
    else if (byteSize == 2)
    {
        extendAclnn<uint16_t>(input, output,
                              x_shape, y_shape,
                              ndim, num, axis);
    }
}

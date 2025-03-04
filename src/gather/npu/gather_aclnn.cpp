#include "acl/acl.h"
#include "aclnnop/aclnn_gather_v2.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

template <typename T>
void gatherAclnnDevice(void *input, void *indices, void *output,
                       int *x_shape, int *w_shape, int *y_shape,
                       int x_ndim, int w_ndim, int y_ndim, int axis, aclrtStream &stream)
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

    std::vector<int64_t> inputDim(x_ndim);       // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputStride(x_ndim, 1); // 初始化为1
    std::vector<int64_t> indicesDim(w_ndim);
    std::vector<int64_t> indicesStride(w_ndim, 1);
    std::vector<int64_t> outputDim(y_ndim);
    std::vector<int64_t> outputStride(y_ndim, 1);

    for (int i = x_ndim - 1; i >= 0; i--)
    {
        inputDim[i] = int64_t(x_shape[i]);
        if (i < x_ndim - 1)
        {
            inputStride[i] = inputDim[i + 1] * inputStride[i + 1];
        }
    }
    for (int i = w_ndim - 1; i >= 0; i--)
    {
        indicesDim[i] = int64_t(w_shape[i]);
        if (i < w_ndim - 1)
        {
            indicesStride[i] = indicesDim[i + 1] * indicesStride[i + 1];
        }
    }
    for (int i = y_ndim - 1; i >= 0; i--)
    {
        outputDim[i] = int64_t(y_shape[i]);

        if (i < y_ndim - 1)
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
                        indicesDim.data(), indicesDim.size(), indices);
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
    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
    // aclDestroyAclOpExecutor(executor);//似乎不支持destroy，一旦destroy测试报错
}
template <typename T>
void gatherAclnn(void *input, void *indices, void *output,
                 int *x_shape, int *w_shape, int *y_shape,
                 int x_ndim, int w_ndim, int y_ndim, int axis)
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

    gatherAclnnDevice<T>(input, indices, output,
                         x_shape, w_shape, y_shape,
                         x_ndim, w_ndim, y_ndim, axis, stream);
    Finalize(deviceId, stream);
}
extern "C" void gather_aclnn(void *input, void *indices, void *output,
                             int *x_shape, int *w_shape, int *y_shape,
                             int x_ndim, int w_ndim, int y_ndim, int axis, int byteSize)
{
    if (byteSize == 4)
    {
        gatherAclnn<float>(input, indices, output,
                           x_shape, w_shape, y_shape,
                           x_ndim, w_ndim, y_ndim, axis);
    }
    else if (byteSize == 2)
    {
        gatherAclnn<uint16_t>(input, indices, output,
                              x_shape, w_shape, y_shape,
                              x_ndim, w_ndim, y_ndim, axis);
    }
}

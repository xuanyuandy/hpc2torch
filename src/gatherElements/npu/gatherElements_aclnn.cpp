#include "acl/acl.h"
#include "aclnnop/aclnn_gather.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

template <typename T, typename Tind>
void gatherElementsAclnnDevice(void *input, void *indices, void *output,
                               int *x_shape, int *w_shape, int *y_shape,
                               int ndim, int axis, aclrtStream &stream)
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
    std::vector<int64_t> indicesDim(ndim);
    std::vector<int64_t> indicesStride(ndim, 1);
    std::vector<int64_t> outputDim(ndim);
    std::vector<int64_t> outputStride(ndim, 1);

    for (int i = ndim - 1; i >= 0; i--)
    {
        inputDim[i] = int64_t(x_shape[i]);
        indicesDim[i] = int64_t(w_shape[i]);
        outputDim[i] = int64_t(y_shape[i]);
        if (i < ndim - 1)
        {
            inputStride[i] = inputDim[i + 1] * inputStride[i + 1];
            indicesStride[i] = indicesDim[i + 1] * indicesStride[i + 1];
            outputStride[i] = outputDim[i + 1] * outputStride[i + 1];
        }
    }

    auto inputTensor =
        aclCreateTensor(inputDim.data(), inputDim.size(), dataType,
                        inputStride.data(), 0, format,
                        inputDim.data(), inputDim.size(), input); // const aclTensor *inputTensor
    const aclTensor *indicesTensor;
    if (sizeof(Tind) == 4)
    {
        indicesTensor =
            aclCreateTensor(indicesDim.data(), indicesDim.size(), aclDataType::ACL_INT32,
                            indicesStride.data(), 0, format,
                            indicesDim.data(), indicesDim.size(), indices);
    }
    else if (sizeof(Tind) == 8)
    {
        indicesTensor =
            aclCreateTensor(indicesDim.data(), indicesDim.size(), aclDataType::ACL_INT64,
                            indicesStride.data(), 0, format,
                            indicesDim.data(), indicesDim.size(), indices);
    }

    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), output);
    // 下面开始正式计算
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    auto ret = aclnnGatherGetWorkspaceSize(inputTensor, int64_t(axis), indicesTensor,
                                           outputTensor, &workspaceSize, &executor);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnGatherGetWorkspaceSize failed. ERROR: %d\n", ret);
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

    ret = aclnnGather(workspaceAddr, workspaceSize, executor,
                      stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnGather failed. ERROR: %d\n", ret);
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
template <typename T, typename Tind>
void gatherElementsAclnn(void *input, void *indices, void *output,
                         int *x_shape, int *w_shape, int *y_shape,
                         int ndim, int axis)
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

    gatherElementsAclnnDevice<T, Tind>(input, indices, output,
                                       x_shape, w_shape, y_shape,
                                       ndim, axis, stream);
    Finalize(deviceId, stream);
}
extern "C" void gatherElements_aclnn(void *input, void *indices, void *output,
                                     int *x_shape, int *w_shape, int *y_shape,
                                     int ndim, int axis, int byteSize)
{
    if (byteSize == 4)
    { // 暂时假定indices的数据类型是torch.int32
        gatherElementsAclnn<float, int32_t>(input, indices, output,
                                            x_shape, w_shape, y_shape,
                                            ndim, axis);
    }
    else if (byteSize == 2)
    {
        gatherElementsAclnn<uint16_t, int32_t>(input, indices, output,
                                               x_shape, w_shape, y_shape,
                                               ndim, axis);
    }
}

#include "acl/acl.h"
#include "aclnnop/aclnn_slice_v2.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

template <typename T>
void sliceAclnnDevice(void *input, void *output,
                      int *inputShape, int *outputShape, int *begin, int *end, int *stride,
                      int iDim, int oDim, int ndim,
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
    std::vector<int64_t> inputDim(iDim);       // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputStride(iDim, 1); // 初始化为1

    std::vector<int64_t> outputDim(oDim);
    std::vector<int64_t> outputStride(oDim, 1);

    for (int i = iDim - 1; i >= 0; i--)
    {
        inputDim[i] = int64_t(inputShape[i]);
        if (i < iDim - 1)
        {
            inputStride[i] = inputDim[i + 1] * inputStride[i + 1];
        }
    }
    for (int i = oDim - 1; i >= 0; i--)
    {
        outputDim[i] = int64_t(outputShape[i]);
        if (i < oDim - 1)
        {
            outputStride[i] = outputDim[i + 1] * outputStride[i + 1];
        }
    }
    std::vector<int64_t> begins(ndim);
    std::vector<int64_t> ends(ndim);
    std::vector<int64_t> strides(ndim);
    std::vector<int64_t> axes_64(ndim);
    for (int i = 0; i < ndim; i++)
    {
        begins[i] = int64_t(begin[i]);
        ends[i] = int64_t(end[i]);
        strides[i] = int64_t(stride[i]);
        axes_64[i] = int64_t(i);
    }
    auto inputTensor =
        aclCreateTensor(inputDim.data(), inputDim.size(), dataType,
                        inputStride.data(), 0, format,
                        inputDim.data(), inputDim.size(), input); // const aclTensor *inputTensor

    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), output);
    aclIntArray *beginsArray =
        aclCreateIntArray(begins.data(), begins.size());
    aclIntArray *endsArray = aclCreateIntArray(ends.data(), ends.size());
    aclIntArray *stridesArray =
        aclCreateIntArray(strides.data(), strides.size());
    aclIntArray *axes = aclCreateIntArray(axes_64.data(), axes_64.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto ret =
        aclnnSliceV2GetWorkspaceSize(inputTensor, beginsArray, endsArray, axes, stridesArray,
                                     outputTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS)
    {
        printf("aclnnSliceV2GetWorkspaceSize failed. ERROR: %d\n", ret);
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

    ret = aclnnSliceV2(workspaceAddr, workspaceSize, executor,
                       stream);
    if (ret != ACL_SUCCESS)
    {
        printf("aclnnSliceV2 failed. ERROR: %d\n", ret);
    }
    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    aclDestroyTensor(inputTensor);
    aclDestroyIntArray(beginsArray);
    aclDestroyIntArray(endsArray);
    aclDestroyIntArray(axes);
    aclDestroyIntArray(stridesArray);
    aclDestroyTensor(outputTensor);
    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
}
template <typename T>
void sliceAclnn(void *input, void *output,
                int *inputShape, int *outputShape, int *begin, int *end, int *stride,
                int iDim, int oDim, int ndim)
{
    int32_t deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS)
    {
        printf("Init acl failed. ERROR: %d\n", ret);
    }

    sliceAclnnDevice<T>(input, output,
                        inputShape, outputShape, begin, end, stride,
                        iDim, oDim, ndim, stream);
    Finalize(deviceId, stream);
}
extern "C" void slice_aclnn(void *input, void *output,
                            int *inputShape, int *outputShape, int *begin, int *end, int *stride,
                            int iDim, int oDim, int ndim, int byteSize)
{
    if (byteSize == 4)
    {
        sliceAclnn<float>(input, output,
                          inputShape, outputShape, begin, end, stride,
                          iDim, oDim, ndim);
    }
    else if (byteSize == 2)
    {
        sliceAclnn<uint16_t>(input, output,
                             inputShape, outputShape, begin, end, stride,
                             iDim, oDim, ndim);
    }
}

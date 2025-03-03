#include "acl/acl.h"
#include "aclnnop/aclnn_split_tensor.h"
#include "aclnnop/aclnn_split_with_size.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

template <typename T>
void splitAclnnDevice(void *input, void **output, int *inputShape, int **outputShape,
                      int num, int axis, int ndim,
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
    std::vector<int64_t> inputDim(ndim); // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputStride(ndim, 1);
    for (int i = ndim - 1; i >= 0; i--)
    {
        inputDim[i] = int64_t(inputShape[i]);
        if (i < ndim - 1)
        {
            inputStride[i] = inputDim[i + 1] * inputStride[i + 1];
        }
    }
    auto inputTensor =
        aclCreateTensor(inputDim.data(), inputDim.size(), dataType,
                        inputStride.data(), 0, format,
                        inputDim.data(), inputDim.size(), input); // const aclTensor *inputTensor

    std::vector<aclTensor *> outputsData{};
    for (int i = 0; i < num; ++i)
    {
        std::vector<int64_t> cDim(ndim);
        std::vector<int64_t> cStride(ndim, 1);

        for (int j = ndim - 1; j >= 0; j--)
        {
            cDim[j] = int64_t(outputShape[i][j]);
            if (j < ndim - 1)
            {
                cStride[j] = cDim[j + 1] * cStride[j + 1];
            }
            // printf("%ld ", cDim[j]);
        }
        // printf("\n");

        aclTensor *tmpTensor = aclCreateTensor(
            cDim.data(), cDim.size(), dataType, cStride.data(), 0,
            format, cDim.data(), cDim.size(), output[i]);

        outputsData.push_back(tmpTensor);
    }
    aclTensorList *tensorList =
        aclCreateTensorList(outputsData.data(), outputsData.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    uint64_t splitSections = inputShape[(axis + ndim) % ndim] / num;

    if (inputShape[(axis + ndim) % ndim] % num > 0)
    {
        splitSections += 1;
    }
    std::vector<int64_t> nonList(num, splitSections);

    if (inputShape[(axis + ndim) % ndim] % num > 0)
    {
        nonList[num - 1] = inputShape[(axis + ndim) % ndim] - num * splitSections;
    }
    bool paraList = false; // 判断切分到底是按照list还是按照均匀

    for (int i = 0; i < num; i++)
    {
        if (nonList[i] != int64_t(outputShape[i][axis]))
        {
            paraList = true; // 说明切片是按照list
            break;
        }
    }
    aclIntArray *splitSize = nullptr;
    aclnnStatus ret;
    if (paraList)
    {
        std::vector<int64_t> splitList(num);
        for (int i = 0; i < num; i++)
        {
            splitList[i] = int64_t(outputShape[i][axis]);
        }
        splitSize = aclCreateIntArray(splitList.data(), num);
        ret = aclnnSplitWithSizeGetWorkspaceSize(
            inputTensor, splitSize, int64_t(axis), tensorList, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnSplitWithSizeGetWorkspaceSize failed. ERROR: %d\n", ret);
        }
    }
    else
    {
        ret = aclnnSplitTensorGetWorkspaceSize(
            inputTensor, splitSections, int64_t(axis), tensorList, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnSplitTensorGetWorkspaceSize failed. ERROR: %d\n", ret);
        }
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
    if (paraList)
    {
        ret = aclnnSplitWithSize(workspaceAddr, workspaceSize, executor,
                                 stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnSplitWithSize failed. ERROR: %d\n", ret);
        }
    }
    else
    {
        ret = aclnnSplitTensor(workspaceAddr, workspaceSize, executor,
                               stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnSplitTensor failed. ERROR: %d\n", ret);
        }
    }

    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    aclDestroyTensor(inputTensor);
    aclDestroyTensorList(tensorList);
    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
    if (paraList)
    {
        aclDestroyIntArray(splitSize);
    }
}
template <typename T>
void splitAclnn(void *input, void **output, int *inputShape, int **outputShape, int num, int axis, int ndim)
{
    int32_t deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS)
    {
        printf("Init acl failed. ERROR: %d\n", ret);
    }

    splitAclnnDevice<T>(input, output, inputShape, outputShape,
                        num, axis, ndim, stream);
    Finalize(deviceId, stream);
}
extern "C" void split_aclnn(void *input, void **output, int *inputShape, int **outputShape, int num, int axis, int ndim, int byteSize)
{
    if (byteSize == 4)
    {
        splitAclnn<float>(input, output, inputShape, outputShape, num, axis, ndim);
    }
    else if (byteSize == 2)
    {
        splitAclnn<uint16_t>(input, output, inputShape, outputShape, num, axis, ndim);
    }
}

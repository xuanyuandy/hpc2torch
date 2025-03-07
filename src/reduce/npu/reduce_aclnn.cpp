#include "acl/acl.h"
#include "aclnnop/aclnn_prod.h"
#include "aclnnop/aclnn_min_dim.h"
#include "aclnnop/aclnn_max_v2.h"
#include "aclnnop/aclnn_mean.h"
#include "aclnnop/aclnn_reduce_sum.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

struct ReduceMode
{
    enum Mode
    {
        // Arithmetic operations:
        Max,
        Mean,
        Min,
        Prod,
        Sum,

        Count, ///< Number of reduce operation types (marker for counting purposes).
    };

    // This static constant holds the total number of defined reduce operations.
    static const size_t numReduceMode = Count;
};
template <typename T>
void reduceAclnnDevice(void *aData, int *axes, void *cData,
                       int *aShape, int *cShape,
                       int ndim, int axesDim,
                       ReduceMode::Mode mode,
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
    std::vector<int64_t> inputDim(ndim);
    std::vector<int64_t> inputStride(ndim, 1);
    std::vector<int64_t> outputDim(ndim);
    std::vector<int64_t> outputStride(ndim, 1);
    std::vector<int64_t> axes_64(axesDim);
    int64_t iSize = 1;
    for (int i = ndim - 1; i >= 0; i--)
    {
        inputDim[i] = int64_t(aShape[i]);
        outputDim[i] = int64_t(cShape[i]);
        iSize *= inputDim[i];
        if (i < ndim - 1)
        {
            inputStride[i] = inputDim[i + 1] * inputStride[i + 1];
            outputStride[i] = outputDim[i + 1] * outputStride[i + 1];
        }
    }
    for (int i = axesDim - 1; i >= 0; i--)
    {
        axes_64[i] = int64_t(axes[i]);
    }
    auto inputTensor =
        aclCreateTensor(inputDim.data(), inputDim.size(), dataType,
                        inputStride.data(), 0, format,
                        inputDim.data(), inputDim.size(), aData); // const aclTensor *inputTensor
    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), cData);
    aclIntArray *dim = aclCreateIntArray(axes_64.data(), axes_64.size());
    // 下面开始正式计算
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    bool KeepDim = true;
    aclnnStatus ret;
    if (mode == ReduceMode::Max)
    {
        ret = aclnnMaxV2GetWorkspaceSize(
            inputTensor, dim, KeepDim, true, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnMaxV2GetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnMaxV2(workspaceAddr, workspaceSize, executor,
                         stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnMaxV2 failed. ERROR: %d\n", ret);
        }
    }
    else if (mode == ReduceMode::Mean)
    {
        ret = aclnnMeanV2GetWorkspaceSize(
            inputTensor, dim, KeepDim, true, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnMeanV2GetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnMeanV2(workspaceAddr, workspaceSize, executor,
                          stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnMeanV2 failed. ERROR: %d\n", ret);
        }
    }
    else if (mode == ReduceMode::Min)
    { // 似乎只支持针对某固定维度求min

        std::vector<int64_t> tmpInDim = inputDim;
        std::vector<int64_t> tmpInStride = inputStride;
        std::vector<int64_t> tmpOutDim = inputDim;
        std::vector<int64_t> tmpOutStride = inputStride;
        void *tmpInData;
        void *tmpIndicesData;
        void *tmpOutData;
        ret = aclrtMalloc((void **)&tmpInData, iSize * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclrtMalloc((void **)&tmpIndicesData, iSize * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclrtMalloc((void **)&tmpOutData, iSize * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
        {
            printf("aclrtMalloc failed. ERROR: %d\n", ret);
        }
        ret = aclrtMemcpy(tmpInData, iSize * sizeof(T), aData, iSize * sizeof(T), ACL_MEMCPY_DEVICE_TO_DEVICE);
        ret = aclrtMemcpy(tmpOutData, iSize * sizeof(T), aData, iSize * sizeof(T), ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (ret != ACL_SUCCESS)
        {
            printf("aclrtMemcpy failed. ERROR: %d\n", ret);
        }
        int64_t oSize = iSize;
        aclTensor *tmpInTensor = nullptr;
        aclTensor *tmpIndicesTensor = nullptr;
        aclTensor *tmpOutTensor = nullptr;
        for (int i = 0; i < axesDim; i++)
        {
            tmpOutDim[axes[i]] = 1;
            oSize = oSize / inputDim[axes[i]];
            for (int j = ndim - 2; j >= 0; j--)
            {
                tmpOutStride[j] = tmpOutStride[j + 1] * tmpOutDim[j + 1];
            }
            tmpInTensor =
                aclCreateTensor(tmpInDim.data(), tmpInDim.size(), dataType,
                                tmpInStride.data(), 0, format,
                                tmpInDim.data(), tmpInDim.size(), tmpInData);
            tmpIndicesTensor =
                aclCreateTensor(tmpOutDim.data(), tmpOutDim.size(), aclDataType::ACL_INT32,
                                tmpOutStride.data(), 0, format,
                                tmpOutDim.data(), tmpOutDim.size(), tmpIndicesData);

            tmpOutTensor =
                aclCreateTensor(tmpOutDim.data(), tmpOutDim.size(), dataType,
                                tmpOutStride.data(), 0, format,
                                tmpOutDim.data(), tmpOutDim.size(), tmpOutData);
            ret = aclnnMinDimGetWorkspaceSize(
                tmpInTensor, int64_t(axes[i]), KeepDim, tmpOutTensor, tmpIndicesTensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS)
            {
                printf("aclnnMinDimGetWorkspaceSize failed. ERROR: %d\n", ret);
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
            ret = aclnnMinDim(workspaceAddr, workspaceSize, executor,
                              stream);
            if (ret != ACL_SUCCESS)
            {
                printf("aclnnMinDim failed. ERROR: %d\n", ret);
            }
            ret = aclrtSynchronizeStream(stream);

            if (ret != ACL_SUCCESS)
            {
                printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            }
            ret = aclrtMemcpy(tmpInData, oSize * sizeof(T), tmpOutData, oSize * sizeof(T), ACL_MEMCPY_DEVICE_TO_DEVICE);
            for (int j = ndim - 1; j >= 0; j--)
            {
                tmpInDim[j] = tmpOutDim[j];
            }
            for (int j = ndim - 2; j >= 0; j--)
            {
                tmpInStride[j] = tmpInStride[j + 1] * tmpInDim[j + 1];
            }
        }
        ret = aclrtMemcpy(cData, oSize * sizeof(T), tmpOutData, oSize * sizeof(T), ACL_MEMCPY_DEVICE_TO_DEVICE);
        aclDestroyTensor(tmpInTensor);
        aclDestroyTensor(tmpIndicesTensor);
        aclDestroyTensor(tmpOutTensor);
        aclrtFree(tmpInData);
        aclrtFree(tmpIndicesData);
        aclrtFree(tmpOutData);
    }
    else if (mode == ReduceMode::Prod)
    {
        std::vector<int64_t> tmpInDim = inputDim;
        std::vector<int64_t> tmpInStride = inputStride;
        std::vector<int64_t> tmpOutDim = inputDim;
        std::vector<int64_t> tmpOutStride = inputStride;
        void *tmpInData;
        void *tmpOutData;
        ret = aclrtMalloc((void **)&tmpInData, iSize * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclrtMalloc((void **)&tmpOutData, iSize * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
        {
            printf("aclrtMalloc failed. ERROR: %d\n", ret);
        }
        ret = aclrtMemcpy(tmpInData, iSize * sizeof(T), aData, iSize * sizeof(T), ACL_MEMCPY_DEVICE_TO_DEVICE);
        ret = aclrtMemcpy(tmpOutData, iSize * sizeof(T), aData, iSize * sizeof(T), ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (ret != ACL_SUCCESS)
        {
            printf("aclrtMemcpy failed. ERROR: %d\n", ret);
        }
        int64_t oSize = iSize;
        aclTensor *tmpInTensor = nullptr;
        aclTensor *tmpOutTensor = nullptr;
        for (int i = 0; i < axesDim; i++)
        {
            tmpOutDim[axes[i]] = 1;
            oSize = oSize / inputDim[axes[i]];
            for (int j = ndim - 2; j >= 0; j--)
            {
                tmpOutStride[j] = tmpOutStride[j + 1] * tmpOutDim[j + 1];
            }
            tmpInTensor =
                aclCreateTensor(tmpInDim.data(), tmpInDim.size(), dataType,
                                tmpInStride.data(), 0, format,
                                tmpInDim.data(), tmpInDim.size(), tmpInData);

            tmpOutTensor =
                aclCreateTensor(tmpOutDim.data(), tmpOutDim.size(), dataType,
                                tmpOutStride.data(), 0, format,
                                tmpOutDim.data(), tmpOutDim.size(), tmpOutData);
            ret = aclnnProdDimGetWorkspaceSize(
                tmpInTensor, int64_t(axes[i]), KeepDim, dataType, tmpOutTensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS)
            {
                printf("aclnnProdDimGetWorkspaceSize failed. ERROR: %d\n", ret);
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
            ret = aclnnProdDim(workspaceAddr, workspaceSize, executor,
                               stream);
            if (ret != ACL_SUCCESS)
            {
                printf("aclnnProdDim failed. ERROR: %d\n", ret);
            }
            ret = aclrtSynchronizeStream(stream);

            if (ret != ACL_SUCCESS)
            {
                printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            }
            ret = aclrtMemcpy(tmpInData, oSize * sizeof(T), tmpOutData, oSize * sizeof(T), ACL_MEMCPY_DEVICE_TO_DEVICE);
            for (int j = ndim - 1; j >= 0; j--)
            {
                tmpInDim[j] = tmpOutDim[j];
            }
            for (int j = ndim - 2; j >= 0; j--)
            {
                tmpInStride[j] = tmpInStride[j + 1] * tmpInDim[j + 1];
            }
        }
        ret = aclrtMemcpy(cData, oSize * sizeof(T), tmpOutData, oSize * sizeof(T), ACL_MEMCPY_DEVICE_TO_DEVICE);
        aclDestroyTensor(tmpInTensor);
        aclDestroyTensor(tmpOutTensor);
        aclrtFree(tmpInData);
        aclrtFree(tmpOutData);
    }
    else if (mode == ReduceMode::Sum)
    {
        // 这个地方似乎不支持使用dataType
        ret = aclnnReduceSumGetWorkspaceSize(
            inputTensor, dim, KeepDim, dataType, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnReduceSumGetWorkspaceSize failed. ERROR: %d\n", ret);
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

        ret = aclnnReduceSum(workspaceAddr, workspaceSize, executor,
                             stream);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnReduceSum failed. ERROR: %d\n", ret);
        }
    }

    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);
    aclDestroyIntArray(dim);
}

template <typename T>
void reduceAclnn(void *aData, int *axes, void *cData,
                 int *aShape, int *cShape,
                 int ndim, int axesDim,
                 ReduceMode::Mode mode)
{
    int32_t deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS)
    {
        printf("Init acl failed. ERROR: %d\n", ret);
    }

    if (axesDim == 0)
    {
        int *newAxes = (int *)malloc(ndim * sizeof(int));
        for (int i = 0; i < ndim; i++)
        {
            newAxes[i] = i;
        }
        reduceAclnnDevice<T>(aData, newAxes, cData,
                             aShape, cShape,
                             ndim, ndim,
                             mode, stream);
        free(newAxes);
    }
    else
    {
        reduceAclnnDevice<T>(aData, axes, cData,
                             aShape, cShape,
                             ndim, axesDim,
                             mode, stream);
    }
    Finalize(deviceId, stream);
}
extern "C" void maxReduce_aclnn(void *aData, int *axes, void *cData,
                                int *aShape, int *cShape,
                                int ndim, int axesDim, int byteSize)
{
    if (byteSize == 2)
    {
        reduceAclnn<uint16_t>(aData, axes, cData,
                              aShape, cShape,
                              ndim, axesDim,
                              ReduceMode::Max);
    }
    else if (byteSize == 4)
    {
        reduceAclnn<float>(aData, axes, cData,
                           aShape, cShape,
                           ndim, axesDim,
                           ReduceMode::Max);
    }
}
extern "C" void meanReduce_aclnn(void *aData, int *axes, void *cData,
                                 int *aShape, int *cShape,
                                 int ndim, int axesDim, int byteSize)
{
    if (byteSize == 2)
    {
        reduceAclnn<uint16_t>(aData, axes, cData,
                              aShape, cShape,
                              ndim, axesDim,
                              ReduceMode::Mean);
    }
    else if (byteSize == 4)
    {
        reduceAclnn<float>(aData, axes, cData,
                           aShape, cShape,
                           ndim, axesDim,
                           ReduceMode::Mean);
    }
}
extern "C" void minReduce_aclnn(void *aData, int *axes, void *cData,
                                int *aShape, int *cShape,
                                int ndim, int axesDim, int byteSize)
{
    if (byteSize == 2)
    {
        reduceAclnn<uint16_t>(aData, axes, cData,
                              aShape, cShape,
                              ndim, axesDim,
                              ReduceMode::Min);
    }
    else if (byteSize == 4)
    {
        reduceAclnn<float>(aData, axes, cData,
                           aShape, cShape,
                           ndim, axesDim,
                           ReduceMode::Min);
    }
}
extern "C" void prodReduce_aclnn(void *aData, int *axes, void *cData,
                                 int *aShape, int *cShape,
                                 int ndim, int axesDim, int byteSize)
{
    if (byteSize == 2)
    {
        reduceAclnn<uint16_t>(aData, axes, cData,
                              aShape, cShape,
                              ndim, axesDim,
                              ReduceMode::Prod);
    }
    else if (byteSize == 4)
    {
        reduceAclnn<float>(aData, axes, cData,
                           aShape, cShape,
                           ndim, axesDim,
                           ReduceMode::Prod);
    }
}
extern "C" void sumReduce_aclnn(void *aData, int *axes, void *cData,
                                int *aShape, int *cShape,
                                int ndim, int axesDim, int byteSize)
{
    if (byteSize == 2)
    {
        reduceAclnn<uint16_t>(aData, axes, cData,
                              aShape, cShape,
                              ndim, axesDim,
                              ReduceMode::Sum);
    }
    else if (byteSize == 4)
    {
        reduceAclnn<float>(aData, axes, cData,
                           aShape, cShape,
                           ndim, axesDim,
                           ReduceMode::Sum);
    }
}

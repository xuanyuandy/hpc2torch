#include "acl/acl.h"
#include "aclnnop/aclnn_s_where.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

template <typename T>
void whereAclnnDevice(void *aData, void *bData, void *cData, void *dData,
                      int *aShape, int *bShape, int *cShape, int *dShape,
                      int aDim, int bDim, int cDim, int dDim,
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
    std::vector<int64_t> inputADim(aDim); // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputAStride(aDim);
    if (aDim == 0)
    {
        inputADim.push_back(1);
        inputAStride.push_back(1);
    }
    else
    {
        inputAStride[aDim - 1] = 1;
        for (int i = aDim - 1; i >= 0; i--)
        {
            inputADim[i] = int64_t(aShape[i]);
            if (i < aDim - 1)
            {
                inputAStride[i] = inputADim[i + 1] * inputAStride[i + 1];
            }
        }
    }
    std::vector<int64_t> inputBDim(bDim); // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputBStride(bDim);
    if (bDim == 0)
    {
        inputBDim.push_back(1);
        inputBStride.push_back(1);
    }
    else
    {
        inputBStride[bDim - 1] = 1;
        for (int i = bDim - 1; i >= 0; i--)
        {
            inputBDim[i] = int64_t(bShape[i]);
            if (i < bDim - 1)
            {
                inputBStride[i] = inputBDim[i + 1] * inputBStride[i + 1];
            }
        }
    }
    std::vector<int64_t> inputCDim(cDim); // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputCStride(cDim);
    if (cDim == 0)
    {
        inputCDim.push_back(1);
        inputCStride.push_back(1);
    }
    else
    {
        inputCStride[cDim - 1] = 1;
        for (int i = cDim - 1; i >= 0; i--)
        {
            inputCDim[i] = int64_t(cShape[i]);
            if (i < cDim - 1)
            {
                inputCStride[i] = inputCDim[i + 1] * inputCStride[i + 1];
            }
        }
    }
    std::vector<int64_t> outputDim(dDim); // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> outputStride(dDim);
    if (dDim == 0)
    {
        outputDim.push_back(1);
        outputStride.push_back(1);
    }
    else
    {
        outputStride[dDim - 1] = 1;
        for (int i = dDim - 1; i >= 0; i--)
        {
            outputDim[i] = int64_t(dShape[i]);
            if (i < dDim - 1)
            {
                outputStride[i] = outputDim[i + 1] * outputStride[i + 1];
            }
        }
    }
    auto inputATensor =
        aclCreateTensor(inputADim.data(), inputADim.size(), dataType,
                        inputAStride.data(), 0, format,
                        inputADim.data(), inputADim.size(), aData);
    auto inputBTensor =
        aclCreateTensor(inputBDim.data(), inputBDim.size(), dataType,
                        inputBStride.data(), 0, format,
                        inputBDim.data(), inputBDim.size(), bData);
    auto inputCTensor =
        aclCreateTensor(inputCDim.data(), inputCDim.size(), aclDataType::ACL_BOOL,
                        inputCStride.data(), 0, format,
                        inputCDim.data(), inputCDim.size(), cData);
    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), dData);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    auto ret = aclnnSWhereGetWorkspaceSize(inputCTensor, inputATensor, inputBTensor, outputTensor, &workspaceSize, &executor);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnSWhereGetWorkspaceSize failed. ERROR: %d\n", ret);
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

    ret = aclnnSWhere(workspaceAddr, workspaceSize, executor,
                      stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnSWhere failed. ERROR: %d\n", ret);
    }
    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }
    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
    aclDestroyTensor(inputATensor);
    aclDestroyTensor(inputBTensor);
    aclDestroyTensor(inputCTensor);
    aclDestroyTensor(outputTensor);
}
template <typename T>
void whereAclnn(void *aData, void *bData, void *cData, void *dData,
                int *aShape, int *bShape, int *cShape, int *dShape,
                int aDim, int bDim, int cDim, int dDim)
{
    int32_t deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS)
    {
        printf("Init acl failed. ERROR: %d\n", ret);
    }

    whereAclnnDevice<T>(aData, bData, cData, dData,
                        aShape, bShape, cShape, dShape,
                        aDim, bDim, cDim, dDim, stream);
    Finalize(deviceId, stream);
}
extern "C" void where_aclnn(void *aData, void *bData, void *cData, void *dData,
                            int *aShape, int *bShape, int *cShape, int *dShape,
                            int aDim, int bDim, int cDim, int dDim, int byteSize)
{
    if (byteSize == 4)
    {
        whereAclnn<float>(aData, bData, cData, dData,
                          aShape, bShape, cShape, dShape,
                          aDim, bDim, cDim, dDim);
    }
    else if (byteSize == 2)
    {
        whereAclnn<uint16_t>(aData, bData, cData, dData,
                             aShape, bShape, cShape, dShape,
                             aDim, bDim, cDim, dDim);
    }
}

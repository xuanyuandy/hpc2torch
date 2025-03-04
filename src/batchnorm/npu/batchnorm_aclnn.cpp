#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm.h"
#include "npu/common_npu.h"

template <typename T>
void batchnormAclnnDevice(void *input, void *scale,
                          void *bias, void *mean, void *var,
                          void *output, int *shape, int nDim, float eps,
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
    aclFormat format;
    if (nDim == 2)
    {
        format = aclFormat::ACL_FORMAT_NC;
    }
    else if (nDim == 3)
    {
        format = aclFormat::ACL_FORMAT_NCL;
    }
    else if (nDim == 4)
    {
        format = aclFormat::ACL_FORMAT_NCHW;
    }
    else if (nDim == 5)
    {
        format = aclFormat::ACL_FORMAT_NCDHW;
    }
    else if (nDim == 6 || nDim == 7 || nDim == 8)
    {
        format = aclFormat::ACL_FORMAT_ND;
    }
    std::vector<int64_t> inputDim(nDim);       // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputStride(nDim, 1); // 初始化为1
    std::vector<int64_t> paraDim(1);
    std::vector<int64_t> paraStride(1, 1);
    std::vector<int64_t> outputDim(nDim);
    std::vector<int64_t> outputStride(nDim, 1);

    for (int i = nDim - 1; i >= 0; i--)
    {
        inputDim[i] = int64_t(shape[i]);
        outputDim[i] = int64_t(shape[i]);

        if (i < nDim - 1)
        {
            inputStride[i] = inputDim[i + 1] * inputStride[i + 1];
            outputStride[i] = outputDim[i + 1] * outputStride[i + 1];
        }
    }
    paraDim[0] = int64_t(shape[1]);

    auto inputTensor =
        aclCreateTensor(inputDim.data(), inputDim.size(), dataType,
                        inputStride.data(), 0, format,
                        inputDim.data(), inputDim.size(), input); // const aclTensor *inputTensor
    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), output);
    // 下面这些数据的dataType昇腾要求必须和input保持一致
    auto meanTensor = aclCreateTensor(
        paraDim.data(), paraDim.size(), dataType, paraStride.data(), 0,
        aclFormat::ACL_FORMAT_ND, paraDim.data(), paraDim.size(), mean);
    auto varTensor = aclCreateTensor(
        paraDim.data(), paraDim.size(), dataType, paraStride.data(), 0,
        aclFormat::ACL_FORMAT_ND, paraDim.data(), paraDim.size(), var);
    auto scaleTensor =
        aclCreateTensor(paraDim.data(), paraDim.size(), dataType,
                        paraStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                        paraDim.data(), paraDim.size(), scale);
    auto biasTensor = aclCreateTensor(
        paraDim.data(), paraDim.size(), dataType, paraStride.data(), 0,
        aclFormat::ACL_FORMAT_ND, paraDim.data(), paraDim.size(), bias);
    // 下面两个参数是保存的mean,保存的var
    // 上面的meanTensor,varTensor是训练期间计算的mean,var,区别不明
    // 按照InfiniTensor来填充数据
    auto savemeanTensor =
        aclCreateTensor(paraDim.data(), paraDim.size(), dataType,
                        paraStride.data(), 0, aclFormat::ACL_FORMAT_ND,
                        paraDim.data(), paraDim.size(), scale);
    auto saveinvstdTensor = aclCreateTensor(
        paraDim.data(), paraDim.size(), dataType, paraStride.data(), 0,
        aclFormat::ACL_FORMAT_ND, paraDim.data(), paraDim.size(), bias);
    // 下面开始正式计算
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    float momentum = 0.1; // pytorch默认是0.1,但是onnx默认是0.9

    auto ret = aclnnBatchNormGetWorkspaceSize(
        inputTensor, scaleTensor, biasTensor, meanTensor, varTensor, false,
        momentum, eps, outputTensor, savemeanTensor,
        saveinvstdTensor, &workspaceSize, &executor);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnBatchNormGetWorkspaceSize. ERROR: %d\n", ret);
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
    ret = aclnnBatchNorm(workspaceAddr, workspaceSize, executor,
                         stream);
    if (ret != ACL_SUCCESS)
    {
        printf("aclnnBatchNorm failed. ERROR: %d\n", ret);
    }
    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);
    aclDestroyTensor(meanTensor);
    aclDestroyTensor(varTensor);
    aclDestroyTensor(scaleTensor);
    aclDestroyTensor(biasTensor);
    aclDestroyTensor(savemeanTensor);
    aclDestroyTensor(saveinvstdTensor);
    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
    // aclDestroyAclOpExecutor(executor);//似乎不支持destroy，一旦destroy测试报错
}
template <typename T>
void batchnormAclnn(void *input, void *scale,
                    void *bias, void *mean, void *var,
                    void *output, int *shape, int nDim, float eps)
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

    batchnormAclnnDevice<T>(input, scale,
                            bias, mean, var,
                            output, shape, nDim, eps, stream);
    Finalize(deviceId, stream);
}
extern "C" void batchnorm_aclnn(void *input, void *scale,
                                void *bias, void *mean, void *var,
                                void *output, int *shape, int nDim, float eps, int byteSize)
{
    if (byteSize == 4)
    {
        batchnormAclnn<float>(input, scale,
                              bias, mean, var,
                              output, shape, nDim, eps);
    }
    else if (byteSize == 2)
    {
        batchnormAclnn<uint16_t>(input, scale,
                                 bias, mean, var,
                                 output, shape, nDim, eps);
    }
}

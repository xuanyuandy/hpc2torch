#include "cnnl.h"
#include "cnnl_extra.h"
#include <vector>
struct UnaryMode
{
    enum Mode
    {
        Relu,
        Gelu,
        PRelu, // 这个需要特殊处理
        LeakyRelu,
        Sigmoid,
        Round,
        HardSigmoid,
        HardSwish,
        Count, ///< Number of unary operation types (marker for counting purposes).
    };

    // This static constant holds the total number of defined unary operations.
    static const size_t numUnaryMode = Count;
};
template <typename T>
void unaryCnnlDevice(void const *aData, void *cData, int *aShape, int *cShape,
                     int aDim, int cDim,
                     UnaryMode::Mode mode, float coef, float alpha, float beta, float sliceDim, float gamma, float scale,
                     cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    cnnlDataType_t dataType;
    if (sizeof(T) == 2)
    {
        dataType = CNNL_DTYPE_HALF;
    }
    else if (sizeof(T) == 4)
    {
        dataType = CNNL_DTYPE_FLOAT;
    }
    cnnlTensorDescriptor_t aDesc, cDesc;
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlSetTensorDescriptor(
        aDesc, CNNL_LAYOUT_ARRAY, dataType,
        aDim, aShape);

    cnnlCreateTensorDescriptor(&cDesc);
    cnnlSetTensorDescriptor(
        cDesc, CNNL_LAYOUT_ARRAY, dataType,
        cDim, cShape);

    cnnlStatus_t stat;
    if (mode == UnaryMode::Relu)
    {
        cnnlActivationDescriptor_t opDesc;
        cnnlCreateActivationDescriptor(&opDesc);
        cnnlSetActivationDescriptor_v5(
            opDesc, CNNL_ACTIVATION_RELU, CNNL_ACTIVATION_HIGH_PRECISION,
            CNNL_NOT_PROPAGATE_NAN, coef, sliceDim, gamma,
            scale, true);

        stat =
            cnnlActivationForward(handle, opDesc, &alpha, aDesc,
                                  aData, &beta, cDesc, cData);
        cnnlDestroyActivationDescriptor(opDesc);
    }
    else if (mode == UnaryMode::Gelu)
    {
        cnnlActivationDescriptor_t opDesc;
        cnnlCreateActivationDescriptor(&opDesc);
        cnnlSetActivationDescriptor_v5(
            opDesc, CNNL_ACTIVATION_GELU, CNNL_ACTIVATION_HIGH_PRECISION,
            CNNL_NOT_PROPAGATE_NAN, coef, sliceDim, gamma,
            scale, true);

        stat =
            cnnlActivationForward(handle, opDesc, &alpha, aDesc,
                                  aData, &beta, cDesc, cData);
        cnnlDestroyActivationDescriptor(opDesc);
    }
    else if (mode == UnaryMode::Sigmoid)
    {
        cnnlActivationDescriptor_t opDesc;
        cnnlCreateActivationDescriptor(&opDesc);
        cnnlSetActivationDescriptor_v5(
            opDesc, CNNL_ACTIVATION_SIGMOID, CNNL_ACTIVATION_HIGH_PRECISION,
            CNNL_NOT_PROPAGATE_NAN, coef, sliceDim, gamma,
            scale, true);

        stat =
            cnnlActivationForward(handle, opDesc, &alpha, aDesc,
                                  aData, &beta, cDesc, cData);
        cnnlDestroyActivationDescriptor(opDesc);
    }
    else if (mode == UnaryMode::HardSwish)
    {
        cnnlActivationDescriptor_t opDesc;
        cnnlCreateActivationDescriptor(&opDesc);
        cnnlSetActivationDescriptor_v5(
            opDesc, CNNL_ACTIVATION_HARDSWISH, CNNL_ACTIVATION_HIGH_PRECISION,
            CNNL_NOT_PROPAGATE_NAN, coef, sliceDim, gamma,
            scale, true);

        stat =
            cnnlActivationForward(handle, opDesc, &alpha, aDesc,
                                  aData, &beta, cDesc, cData);
        cnnlDestroyActivationDescriptor(opDesc);
    }
    else if (mode == UnaryMode::HardSigmoid)
    {
        cnnlActivationDescriptor_t opDesc;
        cnnlCreateActivationDescriptor(&opDesc);
        cnnlSetActivationDescriptor_v5(
            opDesc, CNNL_ACTIVATION_HARDSIGMOID, CNNL_ACTIVATION_HIGH_PRECISION,
            CNNL_NOT_PROPAGATE_NAN, coef, sliceDim, gamma,
            scale, true);

        stat =
            cnnlActivationForward(handle, opDesc, &alpha, aDesc,
                                  aData, &beta, cDesc, cData);
        cnnlDestroyActivationDescriptor(opDesc);
    }
    else if (mode == UnaryMode::LeakyRelu)
    {
        cnnlActivationDescriptor_t opDesc;
        cnnlCreateActivationDescriptor(&opDesc);
        cnnlSetActivationDescriptor_v5(
            opDesc, CNNL_ACTIVATION_LEAKYRELU, CNNL_ACTIVATION_HIGH_PRECISION,
            CNNL_NOT_PROPAGATE_NAN, coef, sliceDim, gamma,
            scale, true);

        stat =
            cnnlActivationForward(handle, opDesc, &alpha, aDesc,
                                  aData, &beta, cDesc, cData);
        cnnlDestroyActivationDescriptor(opDesc);
    }
    else if (mode == UnaryMode::Round)
    {
        stat =
            cnnlRound(handle, aDesc, aData, cDesc, cData);
    }
    if (stat != CNNL_STATUS_SUCCESS)
        return;
    cnnlDestroyTensorDescriptor(aDesc);
    cnnlDestroyTensorDescriptor(cDesc);
}
// PRelu需要特殊处理
template <typename T>
void pReluCnnlDevice(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                     int aDim, int bDim, int cDim,
                     cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    cnnlDataType_t dataType;
    if (sizeof(T) == 2)
    {
        dataType = CNNL_DTYPE_HALF;
    }
    else if (sizeof(T) == 4)
    {
        dataType = CNNL_DTYPE_FLOAT;
    }
    cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlSetTensorDescriptor(
        aDesc, CNNL_LAYOUT_ARRAY, dataType,
        aDim, aShape);

    cnnlCreateTensorDescriptor(&bDesc);
    if (aDim > bDim)
    {
        std::vector<int> new_bDim(aDim, 1);
        new_bDim[1] = bShape[0];
        cnnlSetTensorDescriptor(
            bDesc, CNNL_LAYOUT_ARRAY, dataType,
            new_bDim.size(), new_bDim.data());
    }
    else
    {
        cnnlSetTensorDescriptor(
            bDesc, CNNL_LAYOUT_ARRAY, dataType,
            bDim, bShape);
    }

    cnnlCreateTensorDescriptor(&cDesc);
    cnnlSetTensorDescriptor(
        cDesc, CNNL_LAYOUT_ARRAY, dataType,
        cDim, cShape);

    cnnlStatus_t stat;
    stat = cnnlPrelu(handle, aDesc, aData,
                     bDesc, bData, cDesc, cData);
    if (stat != CNNL_STATUS_SUCCESS)
        return;

    cnnlDestroyTensorDescriptor(aDesc);
    cnnlDestroyTensorDescriptor(bDesc);
    cnnlDestroyTensorDescriptor(cDesc);
}
template <typename T>
void unaryCnnl(void const *aData, void *cData, int *aShape, int *cShape,
               int aDim, int cDim,
               UnaryMode::Mode mode, float coef, float alpha, float beta, float sliceDim, float gamma, float scale)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    unaryCnnlDevice<T>(aData, cData, aShape, cShape,
                       aDim, cDim,
                       mode, coef, alpha, beta, sliceDim, gamma, scale, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
template <typename T>
void pReluCnnl(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
               int aDim, int bDim, int cDim)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    pReluCnnlDevice<T>(aData, bData, cData, aShape, bShape, cShape,
                       aDim, bDim, cDim, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void relu_cnnl(void const *aData, void *cData, int *aShape, int *cShape,
                          int aDim, int cDim,
                          int byteSize)
{
    if (byteSize == 2)
    {
        unaryCnnl<uint16_t>(aData, cData, aShape, cShape,
                            aDim, cDim,
                            UnaryMode::Relu, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        unaryCnnl<float>(aData, cData, aShape, cShape,
                         aDim, cDim,
                         UnaryMode::Relu, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
}
extern "C" void gelu_cnnl(void const *aData, void *cData, int *aShape, int *cShape,
                          int aDim, int cDim,
                          int byteSize)
{
    if (byteSize == 2)
    {
        unaryCnnl<uint16_t>(aData, cData, aShape, cShape,
                            aDim, cDim,
                            UnaryMode::Gelu, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        unaryCnnl<float>(aData, cData, aShape, cShape,
                         aDim, cDim,
                         UnaryMode::Gelu, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
}
extern "C" void sigmoid_cnnl(void const *aData, void *cData, int *aShape, int *cShape,
                             int aDim, int cDim,
                             int byteSize)
{
    if (byteSize == 2)
    {
        unaryCnnl<uint16_t>(aData, cData, aShape, cShape,
                            aDim, cDim,
                            UnaryMode::Sigmoid, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        unaryCnnl<float>(aData, cData, aShape, cShape,
                         aDim, cDim,
                         UnaryMode::Sigmoid, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
}
extern "C" void hardswish_cnnl(void const *aData, void *cData, int *aShape, int *cShape,
                               int aDim, int cDim,
                               int byteSize)
{
    if (byteSize == 2)
    {
        unaryCnnl<uint16_t>(aData, cData, aShape, cShape,
                            aDim, cDim,
                            UnaryMode::HardSwish, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        unaryCnnl<float>(aData, cData, aShape, cShape,
                         aDim, cDim,
                         UnaryMode::HardSwish, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
}
extern "C" void hardsigmoid_cnnl(void const *aData, void *cData, int *aShape, int *cShape,
                                 int aDim, int cDim,
                                 int byteSize)
{
    if (byteSize == 2)
    {
        unaryCnnl<uint16_t>(aData, cData, aShape, cShape,
                            aDim, cDim,
                            UnaryMode::HardSigmoid, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f / 6.0f, 0.5f);
    }
    else if (byteSize == 4)
    {
        unaryCnnl<float>(aData, cData, aShape, cShape,
                         aDim, cDim,
                         UnaryMode::HardSigmoid, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f / 6.0f, 0.5f);
    }
}
extern "C" void leakyRelu_cnnl(void const *aData, void *cData, int *aShape, int *cShape,
                               int aDim, int cDim, float coef,
                               int byteSize)
{
    if (byteSize == 2)
    {
        unaryCnnl<uint16_t>(aData, cData, aShape, cShape,
                            aDim, cDim,
                            UnaryMode::LeakyRelu, coef, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        unaryCnnl<float>(aData, cData, aShape, cShape,
                         aDim, cDim,
                         UnaryMode::LeakyRelu, coef, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
}
extern "C" void round_cnnl(void const *aData, void *cData, int *aShape, int *cShape,
                           int aDim, int cDim,
                           int byteSize)
{ // 这个函数没有coef, alpha, beta, sliceDim, gamma, scale参数，这里随便填充即可
    if (byteSize == 2)
    {
        unaryCnnl<uint16_t>(aData, cData, aShape, cShape,
                            aDim, cDim,
                            UnaryMode::Round, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        unaryCnnl<float>(aData, cData, aShape, cShape,
                         aDim, cDim,
                         UnaryMode::Round, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
}
extern "C" void pRelu_cnnl(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                           int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        pReluCnnl<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                            aDim, bDim, cDim);
    }
    else if (byteSize == 4)
    {
        pReluCnnl<float>(aData, bData, cData, aShape, bShape, cShape,
                         aDim, bDim, cDim);
    }
}

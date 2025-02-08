#include "cnnl.h"
#include "cnnl_extra.h"
#include <vector>
struct BinaryMode
{
    enum Mode
    {
        // Arithmetic operations:
        Add,
        Subtract,
        Multiply,
        Divide,
        Pow,
        Mod,
        Max,
        Min,
        BitwiseAnd,
        BitwiseOr,
        BitwiseXor,
        BitwiseNot,
        // Logical operations:
        // **TODO Not currently supported**
        // Requires Boolean data type
        And,
        Or,
        Xor,
        Less,
        LessOrEqual,
        Equal,
        Greater,
        GreaterOrEqual,

        Count, ///< Number of binary operation types (marker for counting purposes).
    };

    // This static constant holds the total number of defined binary operations.
    static const size_t numBinaryMode = Count;
};
template <typename T>
void elementWiseCnnlDevice(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                           int aDim, int bDim, int cDim,
                           BinaryMode::Mode mode, float aAlpha, float bAlpha, float beta,
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
    cnnlSetTensorDescriptor(
        bDesc, CNNL_LAYOUT_ARRAY, dataType,
        bDim, bShape);

    cnnlCreateTensorDescriptor(&cDesc);
    cnnlSetTensorDescriptor(
        cDesc, CNNL_LAYOUT_ARRAY, dataType,
        cDim, cShape);

    size_t wsSize;
    void *wsData;
    cnnlStatus_t stat;
    if (mode == BinaryMode::Add)
    {
        cnnlOpTensorDescriptor_t opDesc;
        cnnlCreateOpTensorDescriptor(&opDesc);
        cnnlSetOpTensorDescriptor(
            opDesc, CNNL_OP_TENSOR_ADD, dataType,
            CNNL_NOT_PROPAGATE_NAN);
        cnnlGetOpTensorWorkspaceSize_v2(handle, opDesc, &aAlpha,
                                        aDesc, aData, &bAlpha, bDesc, bData,
                                        &beta, cDesc, cData, &wsSize);
        cnrtMalloc(&wsData, wsSize);

        stat = cnnlOpTensor(handle, opDesc, &aAlpha,
                            aDesc, aData, &bAlpha, bDesc, bData,
                            wsData, wsSize, &beta, cDesc, cData);
        cnnlDestroyOpTensorDescriptor(opDesc);
    }
    else if (mode == BinaryMode::Multiply)
    {
        cnnlOpTensorDescriptor_t opDesc;
        cnnlCreateOpTensorDescriptor(&opDesc);
        cnnlSetOpTensorDescriptor(
            opDesc, CNNL_OP_TENSOR_MUL, dataType,
            CNNL_NOT_PROPAGATE_NAN);
        cnnlGetOpTensorWorkspaceSize_v2(handle, opDesc, &aAlpha,
                                        aDesc, aData, &bAlpha, bDesc, bData,
                                        &beta, cDesc, cData, &wsSize);
        cnrtMalloc(&wsData, wsSize);

        stat = cnnlOpTensor(handle, opDesc, &aAlpha,
                            aDesc, aData, &bAlpha, bDesc, bData,
                            wsData, wsSize, &beta, cDesc, cData);
        cnnlDestroyOpTensorDescriptor(opDesc);
    }
    else if (mode == BinaryMode::Divide)
    {
        cnnlGetDivWorkspaceSize(handle, aDesc, bDesc, cDesc,
                                &wsSize);
        cnrtMalloc(&wsData, wsSize);

        stat = cnnlDiv_v2(
            handle, CNNL_COMPUTATION_HIGH_PRECISION, aDesc,
            aData, bDesc, bData, wsData, wsSize, cDesc, cData);
    }
    else if (mode == BinaryMode::Max)
    {
        cnnlGetMaximumWorkspaceSize(handle, cDesc, &wsSize);

        cnrtMalloc(&wsData, wsSize);
        stat =
            cnnlMaximum(handle, aDesc, aData, bDesc, bData,
                        cDesc, cData, wsData, wsSize);
    }
    else if (mode == BinaryMode::Min)
    {
        cnnlGetMinimumWorkspaceSize(handle, cDesc, &wsSize);

        cnrtMalloc(&wsData, wsSize);
        stat =
            cnnlMinimum(handle, aDesc, aData, bDesc, bData,
                        cDesc, cData, wsData, wsSize);
    }
    else if (mode == BinaryMode::Pow)
    {
        cnnlGetPowWorkspaceSize(handle, aDesc, bDesc, cDesc,
                                &wsSize);
        cnrtMalloc(&wsData, wsSize);

        stat =
            cnnlPow(handle, CNNL_COMPUTATION_HIGH_PRECISION,
                    aDesc, aData, bDesc, bData, wsData, wsSize, cDesc, cData);
    }
    else if (mode == BinaryMode::Mod)
    {
        cnnlGetFloorModWorkspaceSize(handle, aDesc, bDesc, cDesc,
                                     &wsSize);
        cnrtMalloc(&wsData, wsSize);

        stat =
            cnnlFloorMod(handle, aDesc, aData, bDesc, bData,
                         cDesc, cData, wsData, wsSize);
    }

    if (stat != CNNL_STATUS_SUCCESS)
        return;

    cnnlDestroyTensorDescriptor(aDesc);
    cnnlDestroyTensorDescriptor(bDesc);
    cnnlDestroyTensorDescriptor(cDesc);

    cnrtFree(wsData);
}
template <typename T>
void bitCnnlDevice(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                   int aDim, int bDim, int cDim,
                   BinaryMode::Mode mode,
                   cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    cnnlDataType_t dataType;
    if (sizeof(T) == 2)
    {
        dataType = CNNL_DTYPE_INT16;
    }
    else if (sizeof(T) == 4)
    {
        dataType = CNNL_DTYPE_INT32;
    }
    cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlSetTensorDescriptor(
        aDesc, CNNL_LAYOUT_ARRAY, dataType,
        aDim, aShape);

    cnnlCreateTensorDescriptor(&bDesc);
    cnnlSetTensorDescriptor(
        bDesc, CNNL_LAYOUT_ARRAY, dataType,
        bDim, bShape);

    cnnlCreateTensorDescriptor(&cDesc);
    cnnlSetTensorDescriptor(
        cDesc, CNNL_LAYOUT_ARRAY, dataType,
        cDim, cShape);

    size_t wsSize;
    void *wsData;
    cnnlStatus_t stat;
    if (mode == BinaryMode::BitwiseAnd)
    {
        cnnlGetBitComputeWorkspaceSize(handle, aDesc, bDesc,
                                       cDesc, &wsSize);
        cnrtMalloc(&wsData, wsSize);

        stat = cnnlBitCompute_v2(handle, CNNL_CYCLE_BAND_OP, aDesc, aData,
                                 bDesc, bData, cDesc, cData, wsData, wsSize);
    }
    else if (mode == BinaryMode::BitwiseOr)
    {
        cnnlGetBitComputeWorkspaceSize(handle, aDesc, bDesc,
                                       cDesc, &wsSize);
        cnrtMalloc(&wsData, wsSize);

        stat = cnnlBitCompute_v2(handle, CNNL_CYCLE_BOR_OP, aDesc, aData,
                                 bDesc, bData, cDesc, cData, wsData, wsSize);
    }
    else if (mode == BinaryMode::BitwiseXor)
    {
        cnnlGetBitComputeWorkspaceSize(handle, aDesc, bDesc,
                                       cDesc, &wsSize);
        cnrtMalloc(&wsData, wsSize);

        stat = cnnlBitCompute_v2(handle, CNNL_CYCLE_BXOR_OP, aDesc, aData,
                                 bDesc, bData, cDesc, cData, wsData, wsSize);
    }
    else if (mode == BinaryMode::BitwiseNot)
    {
        cnnlGetBitComputeWorkspaceSize(handle, aDesc, bDesc,
                                       cDesc, &wsSize);
        cnrtMalloc(&wsData, wsSize);

        stat = cnnlBitCompute_v2(handle, CNNL_BNOT_OP, aDesc, aData,
                                 bDesc, bData, cDesc, cData, wsData, wsSize);
    }

    if (stat != CNNL_STATUS_SUCCESS)
        return;

    cnnlDestroyTensorDescriptor(aDesc);
    cnnlDestroyTensorDescriptor(bDesc);
    cnnlDestroyTensorDescriptor(cDesc);

    cnrtFree(wsData);
}
template <typename T>
void elementWiseCnnl(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                     int aDim, int bDim, int cDim,
                     BinaryMode::Mode mode, float aAlpha, float bAlpha, float beta)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    elementWiseCnnlDevice<T>(aData, bData, cData, aShape, bShape, cShape,
                             aDim, bDim, cDim,
                             mode, aAlpha, bAlpha, beta, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void add_cnnl(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                         int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        elementWiseCnnl<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                                  aDim, bDim, cDim,
                                  BinaryMode::Add, 1.0f, 1.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        elementWiseCnnl<float>(aData, bData, cData, aShape, bShape, cShape,
                               aDim, bDim, cDim,
                               BinaryMode::Add, 1.0f, 1.0f, 0.0f);
    }
}
extern "C" void mul_cnnl(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                         int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        elementWiseCnnl<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                                  aDim, bDim, cDim,
                                  BinaryMode::Multiply, 1.0f, 1.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        elementWiseCnnl<float>(aData, bData, cData, aShape, bShape, cShape,
                               aDim, bDim, cDim,
                               BinaryMode::Multiply, 1.0f, 1.0f, 0.0f);
    }
}
extern "C" void div_cnnl(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                         int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        elementWiseCnnl<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                                  aDim, bDim, cDim,
                                  BinaryMode::Divide, 1.0f, 1.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        elementWiseCnnl<float>(aData, bData, cData, aShape, bShape, cShape,
                               aDim, bDim, cDim,
                               BinaryMode::Divide, 1.0f, 1.0f, 0.0f);
    }
}
extern "C" void max_cnnl(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                         int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        elementWiseCnnl<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                                  aDim, bDim, cDim,
                                  BinaryMode::Max, 1.0f, 1.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        elementWiseCnnl<float>(aData, bData, cData, aShape, bShape, cShape,
                               aDim, bDim, cDim,
                               BinaryMode::Max, 1.0f, 1.0f, 0.0f);
    }
}
extern "C" void min_cnnl(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                         int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        elementWiseCnnl<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                                  aDim, bDim, cDim,
                                  BinaryMode::Min, 1.0f, 1.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        elementWiseCnnl<float>(aData, bData, cData, aShape, bShape, cShape,
                               aDim, bDim, cDim,
                               BinaryMode::Min, 1.0f, 1.0f, 0.0f);
    }
}
extern "C" void pow_cnnl(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                         int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        elementWiseCnnl<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                                  aDim, bDim, cDim,
                                  BinaryMode::Pow, 1.0f, 1.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        elementWiseCnnl<float>(aData, bData, cData, aShape, bShape, cShape,
                               aDim, bDim, cDim,
                               BinaryMode::Pow, 1.0f, 1.0f, 0.0f);
    }
}
extern "C" void mod_cnnl(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                         int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        elementWiseCnnl<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                                  aDim, bDim, cDim,
                                  BinaryMode::Mod, 1.0f, 1.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        elementWiseCnnl<float>(aData, bData, cData, aShape, bShape, cShape,
                               aDim, bDim, cDim,
                               BinaryMode::Mod, 1.0f, 1.0f, 0.0f);
    }
}
template <typename T>
void bitCnnl(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
             int aDim, int bDim, int cDim,
             BinaryMode::Mode mode)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    bitCnnlDevice<T>(aData, bData, cData, aShape, bShape, cShape,
                     aDim, bDim, cDim,
                     mode, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
// bit相关函数似乎不支持浮点数运算
extern "C" void bitwiseAnd_cnnl(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                                int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        bitCnnl<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                          aDim, bDim, cDim,
                          BinaryMode::BitwiseAnd);
    }
    else if (byteSize == 4)
    {
        bitCnnl<uint32_t>(aData, bData, cData, aShape, bShape, cShape,
                          aDim, bDim, cDim,
                          BinaryMode::BitwiseAnd);
    }
}
extern "C" void bitwiseOr_cnnl(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                               int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        bitCnnl<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                          aDim, bDim, cDim,
                          BinaryMode::BitwiseOr);
    }
    else if (byteSize == 4)
    {
        bitCnnl<uint32_t>(aData, bData, cData, aShape, bShape, cShape,
                          aDim, bDim, cDim,
                          BinaryMode::BitwiseOr);
    }
}
extern "C" void bitwiseXor_cnnl(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                                int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        bitCnnl<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                          aDim, bDim, cDim,
                          BinaryMode::BitwiseXor);
    }
    else if (byteSize == 4)
    {
        bitCnnl<uint32_t>(aData, bData, cData, aShape, bShape, cShape,
                          aDim, bDim, cDim,
                          BinaryMode::BitwiseXor);
    }
}
extern "C" void bitwiseNot_cnnl(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                                int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        bitCnnl<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                          aDim, bDim, cDim,
                          BinaryMode::BitwiseNot);
    }
    else if (byteSize == 4)
    {
        bitCnnl<uint32_t>(aData, bData, cData, aShape, bShape, cShape,
                          aDim, bDim, cDim,
                          BinaryMode::BitwiseNot);
    }
}

#include "cnnl.h"
#include <vector>

template <typename T>
void splitCnnlDevice(void const *input, void **output, int *inputShape, int **outputShape, int num, int axis, int nDim, cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dataType;
    if (sizeof(T) == 2)
    {
        dataType = CNNL_DTYPE_HALF;
    }
    else if (sizeof(T) == 4)
    {
        dataType = CNNL_DTYPE_FLOAT;
    }
    cnnlTensorDescriptor_t desc;
    cnnlCreateTensorDescriptor(&desc);
    cnnlSetTensorDescriptor(
        desc, layout, dataType,
        nDim, inputShape);

    cnnlTensorDescriptor_t descArray[num];
    for (int i = 0; i < num; ++i) {
        cnnlCreateTensorDescriptor(&descArray[i]);
        
        cnnlSetTensorDescriptor(descArray[i], layout,
                                dataType,
                                nDim,
                                outputShape[i]);
    }
    size_t wsSize;
    cnnlGetSplitWorkspaceSize(handle, num, &wsSize);
    void *wsData;
    cnrtMalloc(&wsData, wsSize);

    cnnlStatus_t stat =
        cnnlSplit(handle, num, axis, desc, input, wsData,
                      wsSize, descArray, output);
    if (stat != CNNL_STATUS_SUCCESS)
        return;

    for (int i = 0; i < num; ++i) {
        cnnlDestroyTensorDescriptor(descArray[i]);
    }
    cnnlDestroyTensorDescriptor(desc);
}
template <typename T>
void splitCnnl(void const *input, void **output, int *inputShape, int **outputShape, int num, int axis, int nDim)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    splitCnnlDevice<T>(input, output, inputShape, outputShape, num, axis, nDim, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void split_cnnl(void const *input, void **output, int *inputShape, int **outputShape, int num, int axis, int nDim, int byteSize)
{
    if (byteSize == 4)
    {
        splitCnnl<float>(input, output, inputShape, outputShape, num, axis, nDim);
    }
    else if (byteSize == 2)
    {
        splitCnnl<uint16_t>(input, output, inputShape, outputShape, num, axis, nDim);
    }
}

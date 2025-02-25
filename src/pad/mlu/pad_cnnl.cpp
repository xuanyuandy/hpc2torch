#include "cnnl.h"
#include <vector>

template <typename T>
void padCnnlDevice(void const *input, int const *hostPad, void const *hostValue, void *output, 
    int *inputShape, int *outputShape, int nDim, 
    cnnlHandle_t &handle, cnrtQueue_t &queue)
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
    cnnlTensorDescriptor_t aDesc, cDesc;
    // input
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlSetTensorDescriptor(
        aDesc, layout, dataType,
        nDim, inputShape);
    // output
    cnnlCreateTensorDescriptor(&cDesc);
    cnnlSetTensorDescriptor(
        cDesc, layout, dataType,
        nDim, outputShape);
    
    int *paddings = (int *)malloc(2 * nDim * sizeof(int));
    for(int i = 0; i < nDim; i++){
        paddings[2 * i] = hostPad[2 * (nDim - i - 1)];
        paddings[2 * i + 1] = hostPad[2 * (nDim - i - 1) + 1];
    }
    cnnlStatus_t stat = cnnlPad(handle, aDesc, input,
                                paddings, hostValue, cDesc, output);
    if (stat != CNNL_STATUS_SUCCESS)
        return;
    CNRT_CHECK(cnrtQueueSync(queue)); 
    cnnlDestroyTensorDescriptor(aDesc);
    cnnlDestroyTensorDescriptor(cDesc);
    
    free(paddings);
}
template <typename T>
void padCnnl(void const *input, int const *hostPad, void const *hostValue, void *output, 
    int *inputShape, int *outputShape, int nDim)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    padCnnlDevice<T>(input, hostPad, hostValue, output, 
    inputShape, outputShape, nDim, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void pad_cnnl(void const *input, int const *hostPad, void const *hostValue, void *output, 
    int *inputShape, int *outputShape, int nDim, int byteSize)
{
    if (byteSize == 4)
    {
        padCnnl<float>(input, hostPad, hostValue, output, 
    inputShape, outputShape, nDim);
    }
    else if (byteSize == 2)
    {
        padCnnl<uint16_t>(input, hostPad, hostValue, output, 
    inputShape, outputShape, nDim);
    }
}

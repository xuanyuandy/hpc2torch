#include "cnnl.h"
#include <vector>

template <typename T>
void sliceCnnlDevice(void const *input, void *output, 
    int *inputShape, int *outputShape, int *begin, int *end, int *stride, 
    int iDim, int oDim, int ndim,
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
    
    char *tmp = (char *)malloc(3 * ndim * sizeof(int64_t));
    int64_t *begins = (int64_t *)tmp;
    int64_t *ends = begins + ndim;
    int64_t *strides = ends + ndim;
    for(int i = 0; i < ndim; i++){
        begins[i] = static_cast<int64_t>(begin[i]);
        ends[i] = static_cast<int64_t>(end[i]);
        strides[i] = static_cast<int64_t>(stride[i]);
    }
    
    cnnlTensorDescriptor_t aDesc, cDesc;
    // input
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlSetTensorDescriptor(
        aDesc, layout, dataType,
        iDim, inputShape);
    // output
    cnnlCreateTensorDescriptor(&cDesc);
    cnnlSetTensorDescriptor(
        cDesc, layout, dataType,
        oDim, outputShape);
    
    cnnlStatus_t stat =
        cnnlStridedSlice_v2(handle, aDesc, input, begins,
                            ends, strides, cDesc, output);
    
    if (stat != CNNL_STATUS_SUCCESS)
        return;
    CNRT_CHECK(cnrtQueueSync(queue)); 
    cnnlDestroyTensorDescriptor(aDesc);
    cnnlDestroyTensorDescriptor(cDesc);
    free(tmp);
}
template <typename T>
void sliceCnnl(void const *input, void *output, 
    int *inputShape, int *outputShape, int *begin, int *end, int *stride, 
    int iDim, int oDim, int ndim)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    sliceCnnlDevice<T>(input, output, 
    inputShape, outputShape, begin, end, stride, 
    iDim, oDim, ndim, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void slice_cnnl(void const *input, void *output, 
    int *inputShape, int *outputShape, int *begin, int *end, int *stride, 
    int iDim, int oDim, int ndim, int byteSize)
{
    if (byteSize == 4)
    {
        sliceCnnl<float>(input, output, 
    inputShape, outputShape, begin, end, stride, 
    iDim, oDim, ndim);
    }
    else if (byteSize == 2)
    {
        sliceCnnl<uint16_t>(input, output, 
    inputShape, outputShape, begin, end, stride, 
    iDim, oDim, ndim);
    }
}

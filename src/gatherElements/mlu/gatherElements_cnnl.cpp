#include "cnnl.h"
#include "cnnl_extra.h"
#include <vector>

template <typename T, typename Tind>
void gatherElementsCnnlDevice(void const *input, void const *indices, void *output,
                      int *x_shape, int *w_shape, int *y_shape,
                      int ndim, int axis, cnnlHandle_t &handle, cnrtQueue_t &queue)
{

    cnnlTensorDescriptor_t input_desc, indices_desc, output_desc;
    cnnlCreateTensorDescriptor(&input_desc);
    cnnlCreateTensorDescriptor(&indices_desc);
    cnnlCreateTensorDescriptor(&output_desc);

    cnnlDataType_t dataType;
    if (sizeof(T) == 2)
    {
        dataType = CNNL_DTYPE_HALF;
    }
    else if (sizeof(T) == 4)
    {
        dataType = CNNL_DTYPE_FLOAT;
    }

    cnnlSetTensorDescriptor(
        input_desc, CNNL_LAYOUT_ARRAY, dataType,
        ndim, x_shape);
    cnnlSetTensorDescriptor(
        output_desc, CNNL_LAYOUT_ARRAY, dataType,
        ndim, y_shape);
    if (sizeof(Tind) == 4)
    {
        cnnlSetTensorDescriptor(
        indices_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32,
        ndim, w_shape);
    }
    else if (sizeof(Tind) == 8)
    {
        cnnlSetTensorDescriptor(
        indices_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT64,
        ndim, w_shape);
    }
    
    cnnlGather(handle, axis, input_desc, input,
                    indices_desc,
                    indices, output_desc, output);

    CNRT_CHECK(cnrtQueueSync(queue));
    cnnlDestroyTensorDescriptor(input_desc);
    cnnlDestroyTensorDescriptor(output_desc);
    cnnlDestroyTensorDescriptor(indices_desc);
}
template <typename T, typename Tind>
void gatherElementsCnnl(void const *input, void const *indices, void *output,
                int *x_shape, int *w_shape, int *y_shape,
                int ndim, int axis)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    gatherElementsCnnlDevice<T, Tind>(input, indices, output,
                              x_shape, w_shape, y_shape,
                              ndim, axis, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void gatherElements_cnnl(void const *input, void const *indices, void *output,
                                int *x_shape, int *w_shape, int *y_shape,
                                int ndim, int axis, int byteSize)
{
    if(byteSize == 2){
        gatherElementsCnnl<uint16_t, int32_t>(input, indices, output,
                               x_shape, w_shape, y_shape,
                               ndim, axis);
    }
    else if(byteSize == 4){
        gatherElementsCnnl<float, int32_t>(input, indices, output,
                               x_shape, w_shape, y_shape,
                               ndim, axis);
    }
    
}

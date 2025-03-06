#include "cnnl.h"
#include "cnnl_extra.h"
#include <vector>

template <typename T, typename Tind>
void extendCnnlDevice(void const *input, void *output,
                      int *x_shape, int *y_shape,
                      int ndim, int num, int axis, cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    int w_ndim = 1;
    int dimsize = x_shape[(axis + ndim) % ndim];
    int IndicesSize = dimsize * (num + 1);
    int *hostIndices = (int *)malloc(IndicesSize * sizeof(int));
    for(int i = 0; i < IndicesSize; i++){
        hostIndices[i] = i % dimsize;
    }
    int *deviceIndices;
    CNRT_CHECK(cnrtMalloc((void **)&deviceIndices, IndicesSize * sizeof(int)));
    CNRT_CHECK(cnrtMemcpy(deviceIndices, hostIndices, IndicesSize * sizeof(int), cnrtMemcpyHostToDev));

    int *shape_data;
    int shape_size = 1;
    for (int i = 0; i < ndim; i++)
    {
        shape_size *= x_shape[i];
    }
    CNRT_CHECK(cnrtMalloc((void **)&shape_data, shape_size * sizeof(int)));
    CNRT_CHECK(cnrtMemcpy(shape_data, x_shape, shape_size * sizeof(int), cnrtMemcpyHostToDev));

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
    cnnlSetTensorDescriptor(
        indices_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32,
        w_ndim, &IndicesSize);
    cnnlGatherV2(handle, axis, input_desc, input,
                    shape_data, indices_desc,
                    reinterpret_cast<const int *>(deviceIndices), output_desc, output);

    CNRT_CHECK(cnrtQueueSync(queue));
    cnrtFree(shape_data);
    cnrtFree(deviceIndices);
    cnnlDestroyTensorDescriptor(input_desc);
    cnnlDestroyTensorDescriptor(output_desc);
    cnnlDestroyTensorDescriptor(indices_desc);
}
template <typename T, typename Tind>
void extendCnnl(void const *input, void *output,
                      int *x_shape, int *y_shape,
                      int ndim, int num, int axis)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    extendCnnlDevice<T, Tind>(input, output,
                      x_shape, y_shape,
                      ndim, num, axis, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void extend_cnnl(void const *input, void *output,
                      int *x_shape, int *y_shape,
                      int ndim, int num, int axis, int byteSize)
{
    if (byteSize == 2)
    {
        extendCnnl<uint16_t, int32_t>(input, output,
                      x_shape, y_shape,
                      ndim, num, axis);
    }
    else if (byteSize == 4)
    {
        extendCnnl<float, int32_t>(input, output,
                      x_shape, y_shape,
                      ndim, num, axis);
    }
}

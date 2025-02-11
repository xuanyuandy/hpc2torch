#include "cnnl.h"
#include <vector>
struct PoolingMode
{
    enum Mode
    {
        Avg,
        Max,
        Count, ///< Number of unary operation types (marker for counting purposes).
    };

    static const size_t numPoolingMode = Count;
};
template <typename T>
void poolingCnnlDevice(void const *input, void *output,
                       int *windows, int *pads, int *strides, int *dilations,
                       int *x_shape, int *y_shape,
                       PoolingMode::Mode mode,
                       bool ceil_mode, int nDim,
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
    cnnlTensorDescriptor_t inDesc, outDesc;
    cnnlCreateTensorDescriptor(&inDesc);
    cnnlCreateTensorDescriptor(&outDesc);
    if (nDim == 4)
    {
        cnnlSetTensorDescriptor(
            inDesc, CNNL_LAYOUT_NCHW, dataType, nDim,
            x_shape);
        cnnlSetTensorDescriptor(
            outDesc, CNNL_LAYOUT_NCHW, dataType, nDim,
            y_shape);
    }
    else if (nDim == 5)
    { // 对于nDim = 5,cnnl不支持NCDHW
        cnnlSetTensorDescriptor(
            inDesc, CNNL_LAYOUT_NDHWC, dataType, nDim,
            x_shape);
        cnnlSetTensorDescriptor(
            outDesc, CNNL_LAYOUT_NDHWC, dataType, nDim,
            y_shape);
    }
    cnnlPoolingDescriptor_t poolingDesc;
    cnnlCreatePoolingDescriptor(&poolingDesc);
    // 如果nDim = 4，windows,strides, dilations都是长度为2的向量，但是pads长度为4
    // 如果nDim = 5，windows,strides, dilations都是长度为3的向量，但是pads长度为6，表示front, back, top, bottom, left, and right
    cnnlPoolingMode_t computeMode;
    if (mode == PoolingMode::Avg)
    {
        computeMode = CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    }
    else if (mode == PoolingMode::Max)
    {
        computeMode = CNNL_POOLING_MAX;
    }
    int *padding;
    if (nDim == 4)
    {
        padding = (int *)malloc(4 * sizeof(int));
        for (int i = 0; i < 2; i++)
        {
            padding[2 * i] = pads[i];
            padding[2 * i + 1] = pads[i];
        }
    }
    else if (nDim == 5)
    {
        padding = (int *)malloc(6 * sizeof(int));
        for (int i = 0; i < 3; i++)
        {
            padding[2 * i] = pads[i];
            padding[2 * i + 1] = pads[i];
        }
    }
    // printf("%d\n", nDim);
    // cnnlSetPoolingNdDescriptor_v2(poolingDesc,
    //                               computeMode, CNNL_NOT_PROPAGATE_NAN, nDim,
    //                               windows, padding, strides, dilations, ceil_mode);
    cnnlSetPooling2dDescriptor_v2(poolingDesc,
                                  computeMode, CNNL_NOT_PROPAGATE_NAN,
                                  windows[0], windows[1],
                                  padding[0], padding[1], padding[2], padding[3],
                                  strides[0], strides[1], dilations[0], dilations[1], ceil_mode);
    int out_w_size = y_shape[3];
    int out_h_size = y_shape[2];
    size_t wsSize;
    cnnlGetPoolingWorkspaceSize(handle, computeMode,
                                out_w_size, out_h_size, &wsSize); // 这个函数只支持2D计算
    void *wsData;
    cnrtMalloc(&wsData, wsSize);
    float alpha = 1.f, beta = 0.f;
    cnnlPoolingForward(handle, poolingDesc,
                       &alpha, inDesc, input, &beta,
                       outDesc, output, wsData, wsSize);
    cnnlDestroyTensorDescriptor(inDesc);
    cnnlDestroyTensorDescriptor(outDesc);
    cnnlDestroyPoolingDescriptor(poolingDesc);
    free(padding);
}
template <typename T>
void poolingCnnl(void const *input, void *output,
                 int *windows, int *pads, int *strides, int *dilations,
                 int *x_shape, int *y_shape,
                 PoolingMode::Mode mode,
                 bool ceil_mode, int nDim)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。
    // windows其实就是torch.nn.MaxPool2d里面的kernel_shape
    if (nDim == 3)
    {
        int new_ndim = 4;
        int *new_windows = (int *)malloc(2 * sizeof(int));
        int *new_pads = (int *)malloc(2 * sizeof(int));
        int *new_strides = (int *)malloc(2 * sizeof(int));
        int *new_dilations = (int *)malloc(2 * sizeof(int));
        int *new_x_shape = (int *)malloc(new_ndim * sizeof(int));
        int *new_w_shape = (int *)malloc(new_ndim * sizeof(int));
        int *new_y_shape = (int *)malloc(new_ndim * sizeof(int));
        for (int i = 0; i < 2; i++)
        {
            new_windows[i] = (i < 1 ? windows[i] : 1);
            new_pads[i] = (i < 1 ? pads[i] : 0);
            new_strides[i] = (i < 1 ? strides[i] : 1);
            new_dilations[i] = (i < 1 ? dilations[i] : 1);
        }
        for (int i = 0; i < new_ndim; i++)
        {
            new_x_shape[i] = (i < nDim ? x_shape[i] : 1);
            new_y_shape[i] = (i < nDim ? y_shape[i] : 1);
        }
        poolingCnnlDevice<T>(input, output,
                             new_windows, new_pads, new_strides, new_dilations,
                             new_x_shape, new_y_shape,
                             mode,
                             ceil_mode, new_ndim, handle, queue);
        free(new_windows);
        free(new_pads);
        free(new_strides);
        free(new_dilations);
        free(new_x_shape);
        free(new_y_shape);
    }
    else
    {
        poolingCnnlDevice<T>(input, output,
                             windows, pads, strides, dilations,
                             x_shape, y_shape,
                             mode,
                             ceil_mode, nDim, handle, queue);
    }

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void MaxPooling_cnnl(void const *input, void *output,
                                int *windows, int *pads, int *strides, int *dilations,
                                int *x_shape, int *y_shape,
                                int nDim,
                                int byteSize)
{
    bool ceil_mode = false;
    if (byteSize == 2)
    {
        poolingCnnl<uint16_t>(input, output,
                              windows, pads, strides, dilations,
                              x_shape, y_shape,
                              PoolingMode::Max,
                              ceil_mode, nDim);
    }
    else if (byteSize == 4)
    {
        poolingCnnl<float>(input, output,
                           windows, pads, strides, dilations,
                           x_shape, y_shape,
                           PoolingMode::Max,
                           ceil_mode, nDim);
    }
}
extern "C" void AvgPooling_cnnl(void const *input, void *output,
                                int *windows, int *pads, int *strides, int *dilations,
                                int *x_shape, int *y_shape,
                                int nDim,
                                int byteSize)
{
    bool ceil_mode = true;
    if (byteSize == 2)
    {
        poolingCnnl<uint16_t>(input, output,
                              windows, pads, strides, dilations,
                              x_shape, y_shape,
                              PoolingMode::Avg,
                              ceil_mode, nDim);
    }
    else if (byteSize == 4)
    {
        poolingCnnl<float>(input, output,
                           windows, pads, strides, dilations,
                           x_shape, y_shape,
                           PoolingMode::Avg,
                           ceil_mode, nDim);
    }
}

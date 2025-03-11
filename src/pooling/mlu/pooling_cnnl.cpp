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
    cnnlSetTensorDescriptor(
        inDesc, CNNL_LAYOUT_NCHW, dataType, nDim,
        x_shape);
    cnnlSetTensorDescriptor(
        outDesc, CNNL_LAYOUT_NCHW, dataType, nDim,
        y_shape);

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
    cnrtFree(wsData);
    free(padding);
}
template <typename T>
void poolingCnnlDeviceDim_5(void const *input, void *output,
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
    std::vector<int> permuteI(nDim); // 从ncdhw做转置到ndhwc
    std::vector<int> permuteO(nDim); // 从ndhwc转置回ncdhw
    for (int i = 0; i < nDim; i++)
    {
        permuteI[i] = i;
        permuteO[i] = i;
    }
    for (int i = 0; i < nDim; i++)
    {
        if (i >= 1)
        {
            permuteI[i] = i + 1;
        }
        if (i >= 2)
        {
            permuteO[i] = i - 1;
        }
    }
    permuteI[nDim - 1] = 1;
    permuteO[1] = nDim - 1;
    std::vector<int> x_tranDim(nDim); // tmpGdramI的形状
    std::vector<int> y_tranDim(nDim); // tmpGdramO的形状
    int x_size = 1;                   // 表示input的size
    int y_size = 1;                   // 表示output的size
    for (int i = 0; i < nDim; i++)
    {
        x_tranDim[i] = x_shape[permuteI[i]];
        y_tranDim[i] = y_shape[permuteI[i]];
        x_size *= x_shape[i];
        y_size *= y_shape[i];
    }
    T *tmpGdramI, *tmpGdramO;
    CNRT_CHECK(cnrtMalloc((void **)&tmpGdramI, x_size * sizeof(T)));
    CNRT_CHECK(cnrtMalloc((void **)&tmpGdramO, y_size * sizeof(T)));

    cnnlTensorDescriptor_t x_desc, y_desc, inDesc, outDesc;
    cnnlCreateTensorDescriptor(&x_desc);
    cnnlCreateTensorDescriptor(&y_desc);
    cnnlCreateTensorDescriptor(&inDesc);
    cnnlCreateTensorDescriptor(&outDesc);

    // 对于nDim = 5,cnnl不支持NCDHW
    cnnlSetTensorDescriptor(
        x_desc, CNNL_LAYOUT_NCDHW, dataType, nDim,
        x_shape);
    cnnlSetTensorDescriptor(
        inDesc, CNNL_LAYOUT_NDHWC, dataType, x_tranDim.size(),
        x_tranDim.data());
    cnnlSetTensorDescriptor(
        y_desc, CNNL_LAYOUT_NCDHW, dataType, nDim,
        y_shape);
    cnnlSetTensorDescriptor(
        outDesc, CNNL_LAYOUT_NDHWC, dataType, y_tranDim.size(),
        y_tranDim.data());
    // 下面开始针对input做转置
    cnnlTransposeDescriptor_t desc;
    cnnlCreateTransposeDescriptor(&desc);
    cnnlSetTransposeDescriptor(desc, nDim, permuteI.data());
    // 然后针对input做转置nchw2nhwc
    size_t tSizeI;
    cnnlGetTransposeWorkspaceSize(handle, x_desc, desc, &tSizeI);
    void *workspaceI;
    cnrtMalloc(&workspaceI, tSizeI);

    cnnlTranspose_v2(handle, desc, x_desc, input, inDesc,
                     tmpGdramI, workspaceI, tSizeI);
    CNRT_CHECK(cnrtQueueSync(queue));
    // 下面开始做pool
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

    padding = (int *)malloc(6 * sizeof(int));
    for (int i = 0; i < 3; i++)
    {
        padding[2 * i] = pads[i];
        padding[2 * i + 1] = pads[i];
    }

    cnnlSetPoolingNdDescriptor_v2(poolingDesc,
                                  computeMode, CNNL_NOT_PROPAGATE_NAN, nDim,
                                  windows, padding, strides, dilations, ceil_mode);

    int out_w_size = y_shape[4];
    int out_h_size = y_shape[3];
    size_t wsSize;
    cnnlGetPoolingWorkspaceSize(handle, computeMode,
                                out_w_size, out_h_size, &wsSize); // 这个函数只支持2D计算
    void *wsData;
    cnrtMalloc(&wsData, wsSize);
    float alpha = 1.f, beta = 0.f;
    cnnlPoolingForward(handle, poolingDesc,
                       &alpha, inDesc, tmpGdramI, &beta,
                       outDesc, tmpGdramO, wsData, wsSize);
    // 下面开始针对tmpGdramO转置得到NCDHW
    size_t tSizeO;
    cnnlGetTransposeWorkspaceSize(handle, outDesc, desc, &tSizeO);
    void *workspaceO;
    cnrtMalloc(&workspaceO, tSizeO);
    cnnlSetTransposeDescriptor(desc, nDim, permuteO.data());
    cnnlTranspose_v2(handle, desc, outDesc, tmpGdramO, y_desc,
                     output, workspaceO, tSizeO);
    CNRT_CHECK(cnrtQueueSync(queue));

    cnrtFree(tmpGdramI);
    cnrtFree(tmpGdramO);

    cnrtFree(workspaceI);
    cnrtFree(workspaceO);

    cnnlDestroyTensorDescriptor(x_desc);
    cnnlDestroyTensorDescriptor(y_desc);

    cnnlDestroyTensorDescriptor(inDesc);
    cnnlDestroyTensorDescriptor(outDesc);
    cnnlDestroyPoolingDescriptor(poolingDesc);
    cnrtFree(wsData);

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
    else if (nDim == 4)
    {
        poolingCnnlDevice<T>(input, output,
                             windows, pads, strides, dilations,
                             x_shape, y_shape,
                             mode,
                             ceil_mode, nDim, handle, queue);
    }
    else if (nDim == 5)
    {
        poolingCnnlDeviceDim_5<T>(input, output,
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

#include "cnnl.h"
#include <vector>
struct ReduceMode
{
    enum Mode
    {
        // Arithmetic operations:
        Max,
        Mean,
        Min,
        Prod,
        Sum,

        Count, ///< Number of reduce operation types (marker for counting purposes).
    };

    // This static constant holds the total number of defined reduce operations.
    static const size_t numReduceMode = Count;
};
template <typename T>
void reduceCnnlDevice(void const *aData, int *axes, void *cData, 
                           int *aShape, int *cShape,
                           int ndim, int axesDim,
                           ReduceMode::Mode mode,
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
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlTensorDescriptor_t inDesc, outDesc;
    cnnlCreateTensorDescriptor(&inDesc);
    cnnlCreateTensorDescriptor(&outDesc);
    cnnlSetTensorDescriptor(
        inDesc, layout, dataType,
        ndim, aShape);
    cnnlSetTensorDescriptor(
        outDesc, layout, dataType,
        ndim, cShape);
    // get reduce descriptor
    cnnlReduceDescriptor_t reduceDesc;
    cnnlCreateReduceDescriptor(&reduceDesc);

    cnnlReduceOp_t reduceOp;
    if (mode == ReduceMode::Max){
        reduceOp = CNNL_REDUCE_MAX;
    }
    else if (mode == ReduceMode::Mean){
        reduceOp = CNNL_REDUCE_AVG;
    }
    else if (mode == ReduceMode::Min){
        reduceOp = CNNL_REDUCE_MIN;
    }
    else if (mode == ReduceMode::Prod){
        reduceOp = CNNL_REDUCE_MUL;
    }
    else if (mode == ReduceMode::Sum){
        reduceOp = CNNL_REDUCE_ADD;
    }
    cnnlSetReduceDescriptor_v2(
        reduceDesc, axes, axesDim, reduceOp,
        dataType, CNNL_NOT_PROPAGATE_NAN,
        CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES, 0.0);

    size_t wsSize;
    void *wsData;
    cnnlGetReduceOpWorkspaceSize(handle,
                                inDesc, outDesc, reduceDesc,
                                &wsSize);
    int indicesSize = axesDim * sizeof(int);
    cnrtMalloc(&wsData, wsSize + indicesSize);
    char *indices = (char *)wsData + wsSize;
    int *indicesData = (int *)indices;
    cnrtMemcpy(indicesData, axes, indicesSize, cnrtMemcpyHostToDev);
    float alpha = 1.f, beta = 0.f;
    cnnlReduce(
            handle, reduceDesc, wsData, wsSize, &alpha,
            inDesc, aData, indicesSize, indicesData, &beta, outDesc, cData);
    
    cnnlDestroyTensorDescriptor(inDesc);
    cnnlDestroyTensorDescriptor(outDesc);
    cnnlDestroyReduceDescriptor(reduceDesc);
    cnrtFree(wsData);
}

template <typename T>
void reduceCnnl(void const *aData, int *axes, void *cData, 
                           int *aShape, int *cShape,
                           int ndim, int axesDim,
                           ReduceMode::Mode mode)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。
    if (axesDim == 0){
        int *newAxes = (int *)malloc(ndim * sizeof(int));
        for(int i = 0; i < ndim; i++){
            newAxes[i] = i;
        }
        reduceCnnlDevice<T>(aData, newAxes, cData, 
                           aShape, cShape,
                           ndim, ndim,
                           mode, handle, queue);
        free(newAxes);                   
    }
    else{
        reduceCnnlDevice<T>(aData, axes, cData, 
                           aShape, cShape,
                           ndim, axesDim,
                           mode, handle, queue);
    }
    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void maxReduce_cnnl(void const *aData, int *axes, void *cData, 
                           int *aShape, int *cShape,
                           int ndim, int axesDim, int byteSize)
{
    if (byteSize == 2)
    {
        reduceCnnl<uint16_t>(aData, axes, cData, 
                           aShape, cShape,
                           ndim, axesDim,
                           ReduceMode::Max);
    }
    else if (byteSize == 4)
    {
        reduceCnnl<float>(aData, axes, cData, 
                           aShape, cShape,
                           ndim, axesDim,
                           ReduceMode::Max);
    }
}
extern "C" void meanReduce_cnnl(void const *aData, int *axes, void *cData, 
                           int *aShape, int *cShape,
                           int ndim, int axesDim, int byteSize)
{
    if (byteSize == 2)
    {
        reduceCnnl<uint16_t>(aData, axes, cData, 
                           aShape, cShape,
                           ndim, axesDim,
                           ReduceMode::Mean);
    }
    else if (byteSize == 4)
    {
        reduceCnnl<float>(aData, axes, cData, 
                           aShape, cShape,
                           ndim, axesDim,
                           ReduceMode::Mean);
    }
}
extern "C" void minReduce_cnnl(void const *aData, int *axes, void *cData, 
                           int *aShape, int *cShape,
                           int ndim, int axesDim, int byteSize)
{
    if (byteSize == 2)
    {
        reduceCnnl<uint16_t>(aData, axes, cData, 
                           aShape, cShape,
                           ndim, axesDim,
                           ReduceMode::Min);
    }
    else if (byteSize == 4)
    {
        reduceCnnl<float>(aData, axes, cData, 
                           aShape, cShape,
                           ndim, axesDim,
                           ReduceMode::Min);
    }
}
extern "C" void prodReduce_cnnl(void const *aData, int *axes, void *cData, 
                           int *aShape, int *cShape,
                           int ndim, int axesDim, int byteSize)
{
    if (byteSize == 2)
    {
        reduceCnnl<uint16_t>(aData, axes, cData, 
                           aShape, cShape,
                           ndim, axesDim,
                           ReduceMode::Prod);
    }
    else if (byteSize == 4)
    {
        reduceCnnl<float>(aData, axes, cData, 
                           aShape, cShape,
                           ndim, axesDim,
                           ReduceMode::Prod);
    }
}
extern "C" void sumReduce_cnnl(void const *aData, int *axes, void *cData, 
                           int *aShape, int *cShape,
                           int ndim, int axesDim, int byteSize)
{
    if (byteSize == 2)
    {
        reduceCnnl<uint16_t>(aData, axes, cData, 
                           aShape, cShape,
                           ndim, axesDim,
                           ReduceMode::Sum);
    }
    else if (byteSize == 4)
    {
        reduceCnnl<float>(aData, axes, cData, 
                           aShape, cShape,
                           ndim, axesDim,
                           ReduceMode::Sum);
    }
}

#include "cnnl.h"
#include "cnnl.h"
#include <vector>

template <typename T>
void whereCnnlDevice(void const *aData, void const *bData, void const *cData, void *dData, 
    int *aShape, int *bShape, int *cShape, int *dShape, 
    int aDim, int bDim, int cDim, int dDim, 
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
    cnnlTensorDescriptor_t aDesc, bDesc, cDesc, dDesc;
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlCreateTensorDescriptor(&bDesc);
    cnnlCreateTensorDescriptor(&cDesc);
    cnnlCreateTensorDescriptor(&dDesc);

    std::vector<int> aVector(aDim);
    std::vector<int> bVector(bDim);
    std::vector<int> cVector(cDim);
    std::vector<int> dVector(dDim);
    if (aDim == 0){
        aVector.push_back(1);
    }
    else{
        for(int i = 0; i < aDim; i++){
            aVector[i] = aShape[i];
        }
    }
    if (bDim == 0){
        bVector.push_back(1);
    }
    else{
        for(int i = 0; i < bDim; i++){
            bVector[i] = bShape[i];
        }
    }
    if (cDim == 0){
        cVector.push_back(1);
    }
    else{
        for(int i = 0; i < cDim; i++){
            cVector[i] = cShape[i];
        }
    }
    if (dDim == 0){
        dVector.push_back(1);
    }
    else{
        for(int i = 0; i < dDim; i++){
            dVector[i] = dShape[i];
        }
    }

    cnnlSetTensorDescriptor(
        aDesc, layout, dataType,
        aVector.size(), aVector.data());//如果aDim = 0，需要特殊处理，这里暂时不考虑
    cnnlSetTensorDescriptor(
        bDesc, layout, dataType,
        bVector.size(), bVector.data());
    cnnlSetTensorDescriptor(
        cDesc, layout, CNNL_DTYPE_BOOL,
        cVector.size(), cVector.data());
    cnnlSetTensorDescriptor(
        dDesc, layout, dataType,
        dVector.size(), dVector.data());
    size_t wsSize;
    cnnlGetSelectV2WorkspaceSize(handle, cDesc, aDesc, bDesc,
                                    &wsSize);
    void *wsData;
    cnrtMalloc(&wsData, wsSize);
    cnnlStatus_t stat =
        cnnlSelectV2(handle, cDesc, cData, aDesc, aData,
                        bDesc, bData, wsData, wsSize, dDesc, dData);
    if (stat != CNNL_STATUS_SUCCESS)
        return;
    CNRT_CHECK(cnrtQueueSync(queue));

    cnnlDestroyTensorDescriptor(aDesc);
    cnnlDestroyTensorDescriptor(bDesc);
    cnnlDestroyTensorDescriptor(cDesc);
    cnnlDestroyTensorDescriptor(dDesc);
    cnrtFree(wsData);
}
template <typename T>
void whereCnnl(void const *aData, void const *bData, void const *cData, void *dData, 
    int *aShape, int *bShape, int *cShape, int *dShape, 
    int aDim, int bDim, int cDim, int dDim)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    whereCnnlDevice<T>(aData, bData, cData, dData, 
        aShape, bShape, cShape, dShape, 
        aDim, bDim, cDim, dDim, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void where_cnnl(void const *aData, void const *bData, void const *cData, void *dData, 
    int *aShape, int *bShape, int *cShape, int *dShape, 
    int aDim, int bDim, int cDim, int dDim, int byteSize)
{
    if (byteSize == 4)
    {
        whereCnnl<float>(aData, bData, cData, dData, 
        aShape, bShape, cShape, dShape, 
        aDim, bDim, cDim, dDim);
    }
    else if (byteSize == 2)
    {
        whereCnnl<uint16_t>(aData, bData, cData, dData, 
        aShape, bShape, cShape, dShape, 
        aDim, bDim, cDim, dDim);
    }
}

#include "cnnl_extra.h"
#include <vector>

template<typename T>
void causal_softmaxCnnlDevice(void *destination, int *shape, int ndim, cnnlHandle_t &handle, cnrtQueue_t &queue){
    int ndim_ = std::max(ndim, 4);
    std::vector<int> dims(ndim_, 1);
    for (uint64_t i = 0; i < ndim; i++) {
        dims[ndim_ - 1 - i] = static_cast<int>(shape[ndim - i - 1]);
    }

    cnnlTensorDescriptor_t yDesc, maskDesc;
    cnnlCreateTensorDescriptor(&yDesc);
    cnnlCreateTensorDescriptor(&maskDesc);
    cnnlDataType_t dataType;
    if(sizeof(T) == 2){
        dataType = CNNL_DTYPE_HALF;
    }
    else if(sizeof(T) == 4){
        dataType = CNNL_DTYPE_FLOAT;
    }
    cnnlSetTensorDescriptor(yDesc, CNNL_LAYOUT_ARRAY, dataType,
                            dims.size(), dims.data());
    cnnlSetTensorDescriptor(maskDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL,
                            dims.size(), dims.data());
    bool mask_matrix[dims[0]][dims[1]][dims[2]][dims[3]];

    // 填充上三角矩阵（右上角为 false）
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            for (int m = 0; m < dims[2]; ++m) {
                for (int n = 0; n < dims[3]; ++n) {
                    if (n - m > dims[3] - dims[2]) {
                        mask_matrix[i][j][m][n] = true;
                    } else {
                        mask_matrix[i][j][m][n] = false;
                    }
                }
            }
        }
    }
    void *workspace;
    int workspace_size = sizeof(bool) * dims[0] * dims[1] * dims[2] * dims[3];
    cnrtMalloc(&workspace, workspace_size);
    cnrtMemcpy(workspace, mask_matrix, workspace_size, cnrtMemcpyHostToDev);
    cnnlMaskedSoftmax(handle, CNNL_MASKED_SOFTMAX_MASKED_FILL,
                                   -1, 1.0, yDesc, destination, maskDesc, workspace,
                                   yDesc, destination);
    cnrtFree(workspace);
    cnnlDestroyTensorDescriptor(yDesc);
    cnnlDestroyTensorDescriptor(maskDesc);                        

}
template<typename T>
void causal_softmaxCnnl(void *destination, int *shape, int ndim)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。
    
    causal_softmaxCnnlDevice<T>(destination, shape, ndim, handle, queue);
    
    
    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));

    
}
extern "C" void causal_softmax_cnnl_f32(void *destination, int *shape, int ndim){
    causal_softmaxCnnl<float>(destination, shape, ndim);
}
extern "C" void causal_softmax_cnnl_f16(void *destination, int *shape, int ndim){
    causal_softmaxCnnl<uint16_t>(destination, shape, ndim);
}

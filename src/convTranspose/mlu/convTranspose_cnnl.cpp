#include "cnnl.h"
#include <vector>

template<typename T>
void convTransposeCnnlDeviceDim_4(void const *input, void const *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim, cnnlHandle_t &handle, cnrtQueue_t &queue){
    //nDim = len(w_shape) = len(x_shape) = len(y_shape)
    
    std::vector<int> inDim(nDim);//原始input的形状为[n,c,h,w]
    std::vector<int> wDim(nDim);
    std::vector<int> outDim(nDim);
    
    for (int i = 0; i < nDim; i++) {
        inDim[i] = x_shape[i];
        outDim[i] = y_shape[i];
        wDim[i] = w_shape[i];
        
    }
    
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NCHW;//只支持nDim=4,5
    
    cnnlDataType_t dataType;
    if(sizeof(T) == 2){
        dataType = CNNL_DTYPE_HALF;
    }
    else if(sizeof(T) == 4){
        dataType = CNNL_DTYPE_FLOAT;
    }
    cnnlTensorDescriptor_t x_desc, w_desc, y_desc;
    cnnlCreateTensorDescriptor(&x_desc);
    cnnlCreateTensorDescriptor(&w_desc);
    cnnlCreateTensorDescriptor(&y_desc);
    
    cnnlSetTensorDescriptor(
        x_desc, layout, dataType,
        inDim.size(), inDim.data());//原始input,ncdhw
    
    cnnlSetTensorDescriptor(
        w_desc, layout, dataType,
        wDim.size(), wDim.data());//原始scale, ncdhw
    
    cnnlSetTensorDescriptor(
        y_desc, layout, dataType,
        outDim.size(), outDim.data());
    int *pad;
    pad = (int *)malloc(4 * sizeof(int));
    for(int i = 0; i < 2; i++){
        pad[2 * i] = pads[i];
        pad[2 * i + 1] = pads[i];
    }
    
    
    
    cnnlConvolutionDescriptor_t convDesc;
    cnnlCreateConvolutionDescriptor(&convDesc);
    cnnlSetConvolutionDescriptor(convDesc, nDim, pad, strides, dilations, 1,
                                         dataType);
    cnnlConvolutionBwdDataAlgo_t algo;
    cnnlGetConvolutionBackwardDataAlgorithm(
        handle, x_desc, w_desc, convDesc, y_desc,
        CNNL_CONVOLUTION_BWD_DATA_FASTEST, &algo);                                  
    size_t convSize;                                     
    cnnlGetConvolutionBackwardDataWorkspaceSize(handle,
                                       w_desc,
                                       x_desc,
                                       convDesc,
                                       y_desc,
                                       algo,
                                       &convSize);   
    void *workspaceConv;
    cnrtMalloc(&workspaceConv, convSize);  
    cnnlConvolutionBackwardData(
            handle, NULL, w_desc, scale, x_desc, input, convDesc,
            algo, workspaceConv, convSize, NULL, y_desc, output);                                                                 
    
    CNRT_CHECK(cnrtQueueSync(queue));  
    free(pad);
    
    cnrtFree(workspaceConv);

    cnnlDestroyTensorDescriptor(x_desc);
    cnnlDestroyTensorDescriptor(w_desc);
    cnnlDestroyTensorDescriptor(y_desc);
    cnnlDestroyConvolutionDescriptor(convDesc);
}
template<typename T>
void convTransposeCnnlDeviceDim_5(void const *input, void const *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim, cnnlHandle_t &handle, cnrtQueue_t &queue){
    //nDim = len(w_shape) = len(x_shape) = len(y_shape)
    std::vector<int> permuteI(nDim);//从ncdhw做转置到ndhwc
    std::vector<int> permuteO(nDim);//从ndhwc转置回ncdhw
    for (int i = 0; i < nDim; i++) {
        permuteI[i] = i;
        permuteO[i] = i;
    }
    for (int i = 0; i < nDim; i++) {
        if(i >= 1){
            permuteI[i] = i + 1;
        }
        if(i >= 2){
            permuteO[i] = i - 1;
        }
    }
    permuteI[nDim - 1] = 1;
    permuteO[1] = nDim - 1;

    std::vector<int> inDim(nDim);//原始input的形状为[n,c,h,w]
    std::vector<int> wDim(nDim);
    std::vector<int> outDim(nDim);
    int x_size = 1;//表示input的size
    int w_size = 1;//表示scale的size
    int y_size = 1;//表示output的size
    for (int i = 0; i < nDim; i++) {
        inDim[i] = x_shape[i];
        outDim[i] = y_shape[i];
        wDim[i] = w_shape[i];
        x_size *= x_shape[i];
        w_size *= w_shape[i];
        y_size *= y_shape[i];
        
    }
    std::vector<int> x_tranDim(nDim);//tmpGdramI的形状
    std::vector<int> w_tranDim(nDim);//tmpGdramS的形状
    std::vector<int> y_tranDim(nDim);//tmpGdramO的形状
    for(int i = 0; i < nDim; i++){
        x_tranDim[i] = x_shape[permuteI[i]];
        w_tranDim[i] = w_shape[permuteI[i]];
        y_tranDim[i] = y_shape[permuteI[i]];
    }
    cnnlTensorLayout_t layoutI;//cnnlConv只支持nDim=4,5
    cnnlTensorLayout_t layoutO;
    
    layoutI = CNNL_LAYOUT_NCDHW;
    layoutO = CNNL_LAYOUT_NDHWC;
    
    cnnlDataType_t dataType;
    if(sizeof(T) == 2){
        dataType = CNNL_DTYPE_HALF;
    }
    else if(sizeof(T) == 4){
        dataType = CNNL_DTYPE_FLOAT;
    }
    //由于cnnl支持的操作是ndhwc，所以需要提前对数据做permute
    T *tmpGdramI, *tmpGdramS, *tmpGdramO;//conv库函数只能处理[n,h,w,c],tmpGdramI作为转置来变换input
    CNRT_CHECK(cnrtMalloc((void **)&tmpGdramI, x_size * sizeof(T)));
    CNRT_CHECK(cnrtMalloc((void **)&tmpGdramS, w_size * sizeof(T)));
    CNRT_CHECK(cnrtMalloc((void **)&tmpGdramO, y_size * sizeof(T)));

    cnnlTensorDescriptor_t x_desc, w_desc, y_desc, IDesc, SDesc, ODesc;
    cnnlCreateTensorDescriptor(&x_desc);
    cnnlCreateTensorDescriptor(&w_desc);
    cnnlCreateTensorDescriptor(&y_desc);
    cnnlCreateTensorDescriptor(&IDesc);
    cnnlCreateTensorDescriptor(&SDesc);
    cnnlCreateTensorDescriptor(&ODesc);
    
    cnnlSetTensorDescriptor(
        x_desc, layoutI, dataType,
        inDim.size(), inDim.data());//原始input,ncdhw
    cnnlSetTensorDescriptor(
        IDesc, layoutO, dataType,
        x_tranDim.size(), x_tranDim.data());//转置以后的input,ndhwc
    cnnlSetTensorDescriptor(
        w_desc, layoutI, dataType,
        wDim.size(), wDim.data());//原始scale, ncdhw
    cnnlSetTensorDescriptor(
        SDesc, layoutO, dataType,
        w_tranDim.size(), w_tranDim.data());//转置以后的scale,ndhwc
    cnnlSetTensorDescriptor(
        y_desc, layoutI, dataType,
        outDim.size(), outDim.data());
    cnnlSetTensorDescriptor(
        ODesc, layoutO, dataType,
        y_tranDim.size(), y_tranDim.data());
    cnnlTransposeDescriptor_t desc;
    cnnlCreateTransposeDescriptor(&desc);
    cnnlSetTransposeDescriptor(desc, nDim, permuteI.data());
    //然后针对input做转置ncdhw2ndhwc
    size_t tSizeI;
    cnnlGetTransposeWorkspaceSize(handle, x_desc, desc, &tSizeI);
    void *workspaceI;
    cnrtMalloc(&workspaceI, tSizeI);
    
    cnnlTranspose_v2(handle, desc, x_desc, input, IDesc,
                            tmpGdramI, workspaceI, tSizeI);
    CNRT_CHECK(cnrtQueueSync(queue));  
    //然后针对scale做转置ncdhw2ndhwc
    
    size_t tSizeS;
    cnnlGetTransposeWorkspaceSize(handle, w_desc, desc, &tSizeS);
    void *workspaceS;
    cnrtMalloc(&workspaceS, tSizeS);
    
    cnnlTranspose_v2(handle, desc, w_desc, scale, SDesc,
                            tmpGdramS, workspaceS, tSizeS);
    CNRT_CHECK(cnrtQueueSync(queue));  
    //------------------------------------------------------------               
    //上面成功对input, scale做好了ncdhw2ndhwc，下面开始正式计算conv
    int *pad;
    pad = (int *)malloc(6 * sizeof(int));
    for(int i = 0; i < 3; i++){
        pad[2 * i] = pads[i];
        pad[2 * i + 1] = pads[i];
    }
    
    
    
    cnnlConvolutionDescriptor_t convDesc;
    cnnlCreateConvolutionDescriptor(&convDesc);
    cnnlSetConvolutionDescriptor(convDesc, nDim, pad, strides, dilations, 1,
                                         dataType);
    cnnlConvolutionBwdDataAlgo_t algo;
    cnnlGetConvolutionBackwardDataAlgorithm(
        handle, IDesc, SDesc, convDesc, ODesc,
        CNNL_CONVOLUTION_BWD_DATA_FASTEST, &algo);                                  
    size_t convSize;                         
    
    cnnlGetConvolutionBackwardDataWorkspaceSize(handle,
                                       SDesc,
                                       IDesc,
                                       convDesc,
                                       ODesc,
                                       algo,
                                       &convSize);   
    void *workspaceConv;
    cnrtMalloc(&workspaceConv, convSize);  
    cnnlConvolutionBackwardData(
            handle, NULL, SDesc, tmpGdramS, IDesc, tmpGdramI, convDesc,
            algo, workspaceConv, convSize, NULL, ODesc, tmpGdramO);   
    CNRT_CHECK(cnrtQueueSync(queue));                                                                       
    //------------------------------------------------------------ 
    //下面开始提前对output做转置：ndhwc2ncdhw，此时需要重新设置aDesc和cDesc,desc
    
    size_t tSizeO;
    cnnlGetTransposeWorkspaceSize(handle, ODesc, desc, &tSizeO);
    void *workspaceO;
    cnrtMalloc(&workspaceO, tSizeO);
    cnnlSetTransposeDescriptor(desc, nDim, permuteO.data());
    cnnlTranspose_v2(handle, desc, ODesc, tmpGdramO, y_desc,
                            output, workspaceO, tSizeO);
    CNRT_CHECK(cnrtQueueSync(queue));  
    free(pad);
    cnrtFree(tmpGdramI);
    cnrtFree(tmpGdramS);
    cnrtFree(tmpGdramO);

    cnrtFree(workspaceI);
    cnrtFree(workspaceConv);
    cnrtFree(workspaceS);
    cnrtFree(workspaceO);

    cnnlDestroyTensorDescriptor(IDesc);
    cnnlDestroyTensorDescriptor(SDesc);
    cnnlDestroyTensorDescriptor(ODesc);
    cnnlDestroyTransposeDescriptor(desc);

    cnnlDestroyTensorDescriptor(x_desc);
    cnnlDestroyTensorDescriptor(w_desc);
    cnnlDestroyTensorDescriptor(y_desc);
    cnnlDestroyConvolutionDescriptor(convDesc);
}
template<typename T>
void convTransposeCnnl(void const *input, void const *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。
    if(nDim == 3){
        int new_ndim = 4;
        int *new_pads = (int *)malloc(2 * sizeof(int));
        int *new_strides = (int *)malloc(2 * sizeof(int));
        int *new_dilations = (int *)malloc(2 * sizeof(int));
        int *new_x_shape = (int *)malloc(new_ndim * sizeof(int));
        int *new_w_shape = (int *)malloc(new_ndim * sizeof(int));
        int *new_y_shape = (int *)malloc(new_ndim * sizeof(int));
        for(int i = 0; i < 2; i++){
            new_pads[i] = (i < 1 ? pads[i] : 0);
            new_strides[i] = (i < 1 ? strides[i] : 1);
            new_dilations[i] = (i < 1 ? dilations[i] : 1);
        }
        for(int i = 0; i < new_ndim; i++){
            new_x_shape[i] = (i < nDim ? x_shape[i] : 1);
            new_w_shape[i] = (i < nDim ? w_shape[i] : 1);
            new_y_shape[i] = (i < nDim ? y_shape[i] : 1);
        }
        convTransposeCnnlDeviceDim_4<T>(input, scale, output, new_pads, new_strides, new_dilations, new_x_shape, new_w_shape, new_y_shape, new_ndim, handle, queue);
        free(new_pads);
        free(new_strides);
        free(new_dilations);
        free(new_x_shape);
        free(new_w_shape);
        free(new_y_shape);
    }
    else if(nDim == 4){
        convTransposeCnnlDeviceDim_4<T>(input, scale, output, pads, strides, dilations, x_shape, w_shape, y_shape, nDim, handle, queue);
    }
    else if(nDim == 5){
        convTransposeCnnlDeviceDim_5<T>(input, scale, output, pads, strides, dilations, x_shape, w_shape, y_shape, nDim, handle, queue);
    }
    
    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));

    
}
extern "C" void convTranspose_cnnl(void const *input, void const *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim, int byteSize){
    if(byteSize == 4){
        convTransposeCnnl<float>(input, scale, output, pads, strides, dilations, x_shape, w_shape, y_shape, nDim);
    }
    else if(byteSize == 2){
        convTransposeCnnl<uint16_t>(input, scale, output, pads, strides, dilations, x_shape, w_shape, y_shape, nDim);
    }
    
}


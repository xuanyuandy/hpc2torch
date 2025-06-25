import torch
import ctypes
import torch.nn.functional as F
import argparse
import numpy as np
import performance
import sys
import os
from precision_compare import data_compare

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def matmul(_c, beta, _a, _b, alpha):
    a = _a.clone()
    b = _b.clone()
    c = _c.clone()
    input_dtype = c.dtype
    ans = (
        alpha * torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(input_dtype)
        + beta * c
    )
    return ans
    
def test_cuda(M, K, N, test_dtype):
    device = "cuda"
    print(
        f"Testing Matmul on {device} with M-K-N:{M, K, N} , dtype:{test_dtype}"
    )
    A = torch.randn([M, K], device=device, dtype=torch.float32, requires_grad=False) 
    B = torch.randn([K, N], device=device, dtype=torch.float32, requires_grad=False)
    C = torch.zeros([M, N], device=device, dtype=torch.float32, requires_grad=False)
    
    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    B_ptr = ctypes.cast(B.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    C_ptr = ctypes.cast(C.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    if device == "cuda":
        torch_matmul_time = performance.CudaProfile((torch.matmul, (A, B)))
        
        lib.matmul_cuda_f32.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_matmul_time = performance.CudaProfile((
            lib.matmul_cuda_f32,
            (A_ptr, B_ptr, C_ptr, M, K, N)
        ))
        # cudnn still be compiled
        '''
        lib.matmul_cudnn_f32.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
        ]
        custom_matmul_time = performance.CudaProfile((
            lib.matmul_cudnn_f32,
            (A_ptr, B_ptr, C_ptr, M, K, N)
        ))
        '''
    performance.logBenchmark(torch_matmul_time, custom_matmul_time)
    tmpa = torch.matmul(A, B).to('cpu').numpy().flatten()
    tmpb = C.to('cpu').numpy().flatten()

    data_compare(tmpb, tmpa)
    atol = max(abs(tmpa - tmpb))
    rtol = atol / max(abs(tmpb) + 1e-8)

    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
    
parser = argparse.ArgumentParser(description="Test matmul on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda'], required=True, help="Device to run the tests on.")
args = parser.parse_args()

if args.device == "cuda":
    test_cases = [
        # M, K, N, test_dtype, device
        (1024, 128, 1024, torch.float32),
        (1024, 256, 1024, torch.float32),
        (1024, 512, 1024, torch.float32),
        (1024, 1024, 1024, torch.float32),
    ] 
    for M, K, N, test_dtype in test_cases:
        test_cuda(M, K, N, test_dtype)
    

import torch
import ctypes
import numpy as np
from functools import partial
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def random_sample(data, random_val, topp, topk, voc, temperature, torch_device):
    indices = torch.zeros([topk], dtype = torch.int64)
    dataNp = data.clone()
                
    sorted_indices = torch.argsort(dataNp, descending=True)
    indices = sorted_indices[:topk] 
    
    dataNp = dataNp[sorted_indices]
    
    globalM = dataNp[0]
    dataNp = (dataNp - globalM) / temperature
    dataNp = torch.softmax(dataNp.float(), dim = 0)
    
    for i in range(1, topk):
        dataNp[i] = dataNp[i] + dataNp[i - 1]
    
    for end in range(topk):
        if(dataNp[end] >= topp):
            break
    if(end < topk - 1):
        end += 1
    else:
        end = topk
    
    random_val *= dataNp[end - 1]
    
    for i in range(end):
        if(random_val < dataNp[i]):
            return indices[i]

def random_sample_0(data):
    return torch.argmax(data)
def random_sample_torch(data, random_val, topp, topk, voc, temperature, torch_device):
    if(topp > 0 and topk > 1):
        ans = random_sample(data, random_val, topp, topk, voc, temperature, torch_device)
    else:
        ans = random_sample_0(data)
    return ans

def test(torch_device, voc, random_val, topp, topk, temperature):
    byteSize = 2
    x_dtype = torch.float16
    if byteSize == 4:
        x_dtype = torch.float32
    print(
        f"Testing RandomSample on {torch_device} with voc:{voc} , topk:{topk}, topp:{topp}, dtype:{x_dtype}"
    )
    print(1)
    data = torch.arange(voc).float() * 0.0001
    print(2)
    _perm = torch.randperm(voc)
    print(3)
    data = data[_perm].to(x_dtype).to(torch_device)
    print(4)
    if(torch_device == 'mlu' or torch_device == 'npu'):
        
        indices = torch.zeros([1], dtype = torch.int64, device = torch_device)
    else:
        print(5)
        indices = torch.zeros([1], dtype = torch.uint64, device = torch_device)
    print(voc)
    ans = random_sample_torch(data, random_val, topp, topk, voc, temperature, torch_device)
    print(topk)
    probs_ptr = ctypes.cast(data.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    result_ptr = ctypes.cast(indices.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    if torch_device == "sdaa":
        torch_randomSample_time = performance.TecoProfile((random_sample_torch, (data, random_val, topp, topk, voc, temperature, torch_device)))
        lib.randomSample_teco.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int
        ]
        custom_randomSample_time = \
        performance.TecoProfile((lib.randomSample_teco, (result_ptr, 
                                probs_ptr, 
                                random_val,
                                topp,
                                voc,
                                topk,
                                temperature, byteSize)))
        
    performance.logBenchmark(torch_randomSample_time, custom_randomSample_time)
    
    if torch_device == "sdaa":
        torch.sdaa.synchronize()
    #print(indices[0], ans , data[ans], data[indices[0]])
    assert indices[0].to(ans.dtype) == ans or data[ans] == data[indices[0]]

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test randomSample on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', "sdaa"], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # voc, random_val, topp, topk, temperature
        (512, 0.8, 0.8, 3, 0.5),
        (4096, 0.05, 0.9, 5, 1.0),
        (16384, 0.15, 0.85, 10, 2.0),
        (512, 0.08, 0, 3, 0.5),
        (4096, 0.5, 0.9, 1, 1.0),
        (16384, 0.15, 0, 1, 2.0),
        (16384, 0.15, 0, 1, 2.0),
        (32000, 0.08, 0.8, 50, 1.0),
        (32000, 0.08, 1.0, 25, 1.0),
        # (119696, 0.01, 1.0, 100, 1.0),
    ]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'sdaa':
    import torch_sdaa
# 执行过滤后的测试用例
for voc, random_val, topp, topk, temperature in test_cases:
    test(args.device, voc, random_val, topp, topk, temperature)

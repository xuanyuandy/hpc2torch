import torch
import ctypes
import numpy as np
import torch.nn as nn
import math
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def quantize(x, scale, zero, maxq):
    if scale.shape[1] == 1:
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)
    else:
        group_size = x.shape[1] // scale.shape[1]
        y = torch.zeros_like(x)
        for j in range(scale.shape[1]):
            q = torch.clamp(
                torch.round(
                    x[:, j * group_size : (j + 1) * group_size] / scale[:, j : j + 1]
                )
                + zero[:, j : j + 1],
                0,
                maxq,
            )
            y[:, j * group_size : (j + 1) * group_size] = scale[:, j : j + 1] * (
                q - zero[:, j : j + 1]
            )
        return y


class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits=4,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
    ):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            # self.scale = self.scale.unsqueeze(1)
            # self.zero = self.zero.unsqueeze(1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


class GPTQ:

    def __init__(self, weight):
        self.weight = weight
        self.dev = self.weight.device

        self.rows = self.weight.shape[0]
        self.columns = self.weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=0.01, group_size=-1):
        W = self.weight.clone()

        W = W.float()

        # tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        num_groups = self.columns // group_size
        if group_size == -1:
            scale = self.quantizer.scale.clone()
            zero = self.quantizer.zero.clone()
        else:
            scale = torch.zeros(self.rows, num_groups)
            zero = torch.zeros(self.rows, num_groups)
        for index in range(self.columns // blocksize):
            i1 = index * blocksize
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size != -1:
                    if (i1 + i) % group_size == 0:
                        self.quantizer.find_params(
                            W[:, (i1 + i) : (i1 + i + group_size)], weight=True
                        )
                        ind = index * blocksize // group_size + i // group_size

                        scale[:, ind : ind + 1] = self.quantizer.scale
                        zero[:, ind : ind + 1] = self.quantizer.zero

                q = quantize(
                    w.unsqueeze(1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        print('error', torch.sum(Losses).item())

        self.weight = Q.reshape(self.weight.shape).to(self.weight.dtype)
        self.scale = scale.to(self.weight.dtype)
        self.zero = zero.to(self.weight.dtype)


def get_scale_zero(b, a, c, group_size):
    weight = b.clone()
    inp = a.clone()
    out = c.clone()
    gptq = GPTQ(weight)
    gptq.quantizer = Quantizer()
    gptq.quantizer.configure(perchannel=True, sym=False, mse=False)
    gptq.add_batch(inp, out)
    gptq.fasterquant(group_size=group_size)

    return (
        gptq.weight.to(weight.device),
        gptq.scale.to(weight.device),
        gptq.zero.to(weight.device),
    )

def mat(a, b):
    dtype = b.dtype
    ans = torch.matmul(b.to(torch.float32), a.t().to(torch.float32)).to(dtype)
    return ans

def test(torch_device,
    M,
    K,
    N):
    dtype = torch.float16
    print(
        f"Testing MatmulGptq on {torch_device}" f" M:{M}, K:{K}, N:{N}, dtype:{dtype}"
    )
    torch.manual_seed(12)
    # Initialize tensors
    a = 1e0 * torch.randn([M, K], dtype=dtype).to(torch_device)
    layer = nn.Linear(K, N)
    b = 1e-3 * layer.weight.data.to(dtype).to(torch_device)
    c = torch.zeros([M, N], dtype=dtype).to(torch_device).t()

    group_size = -1
    num_groups = 1
    if group_size == -1:
        num_groups = 1
    else:
        num_groups = a.shape[1] / group_size

    b_ref, s, z = get_scale_zero(b, a, c, group_size)

    
    output_ptr = ctypes.cast(c.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    inputA_ptr = ctypes.cast(a.t().data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    inputB_ptr = ctypes.cast(b_ref.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    scale_ptr = ctypes.cast(s.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    zero_ptr = ctypes.cast(z.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    
    if torch_device == "cpu":
        torch_gptq_time = performance.CpuProfile((mat, (a, b)))  # 可以替换为mul, div
        lib.gptq_cpu.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int

        ]
        custom_gptq_time = \
        performance.CpuProfile((lib.gptq_cpu, (output_ptr, inputA_ptr, inputB_ptr, scale_ptr, zero_ptr, M, K, N, group_size)))
    
    performance.logBenchmark(torch_gptq_time, custom_gptq_time)
    tmpa = mat(a, b).to('cpu').numpy().flatten()
    tmpb = c.to('cpu').numpy().flatten()
    torch.allclose(c, mat(a, b), atol=1e-3, rtol=1e-3)
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
parser = argparse.ArgumentParser(description="Test gptq on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    
test_cases = [
        # 
        (1, 1024, 4),
        (16, 1024, 4)
         
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for M, K, N in test_cases:
    test(args.device, M, K, N)
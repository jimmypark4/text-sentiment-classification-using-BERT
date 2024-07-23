import torch

def to_gpu(x, on_cpu=False, gpu_id=None):
    if torch.cuda.is_available() and not on_cpu:
        x=x.cuda(gpu_id)
    
    return x

import torch
import pickle
from collections import defaultdict

def to_gpu(x, on_cpu=False, gpu_id=None):
    if torch.cuda.is_available() and not on_cpu:
        x=x.cuda(gpu_id)
    
    return x
def return_unk():
    word2id = defaultdict(lambda: len(word2id))

    return word2id['']

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
import torch
import torch.nn as nn
from torch.nn import functional as F 






def embedding():
    embedding = nn.Embedding(10, 3)
    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])

    print(input)
    print(embedding(input))



def tensor_view():
    t = torch.rand(4, 4)
    print(t)
    b = t.view(2,8)
    print(b)
    print(t.storage().data_ptr() == b.storage().data_ptr())

 

def tril():
    torch.manual_seed(1337)
    B,T,C = 4,8,32
    x = torch.randn(B,T,C)

    head_size = 6
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    k = key(x)
    q = query(x)
    wei = q @ k.transpose(-2, -1)


    tril = torch.tril(torch.ones(T, T))
    # wei = torch.zeros((T,T))
    wei = wei.masked_fill(tril==0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    out = wei @ x
    print(out.shape)
    print(wei[0])

def transpose():    
    x = torch.randn(2, 3)
    print(x)
    print(x.ndim)
    print(torch.transpose(x, -2, 1))


def t():
    B,T,C = 4,8,32
    head_size = 6
    k = torch.randn(B,T,head_size)
    q = torch.randn(B,T,head_size)
    wei = q @ k.transpose(-2, -1)  * head_size**-0.5
    print(k.var())
    print(q.var())
    print(wei.var())
    print(torch.softmax(torch.tensor([0.1,-0.2,0.3,-0.2,0.5]), dim =-1))
t()
import torch
from torch import nn

def flatten():
    x = torch.randn(10,2,2,3)
    print(x)
    flatten = nn.Flatten()
    y = flatten(x)
    print(y)
    print(y.shape)

def softmax():
    m = nn.Softmax(dim=1)
    input = torch.randn(2,3)
    print(input)
    output = m(input)
    print(output)


def relu():
    m = nn.ReLU()
    input = torch.randn(2)
    print(input)
    output = m(input)
    print(output)

def relu2():
    m = nn.ReLU()
    input = torch.randn(2).unsqueeze(0)
    print(input)
    output = torch.cat((m(input), m(-input)))
    print(output)

# 张量维度比较绕
def unsqueeze():
    x = torch.tensor([1,2,3,4])
    print(x)
    x1 = torch.unsqueeze(x,0)
    print(x1)
    x2 = torch.unsqueeze(x,1)
    print(x2)
    # 这里会出错
    # x3 = torch.unsqueeze(x,2)
    # print(x3)

def crossEntropyLoss():
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3,5, requires_grad=True)
    print(input)
    target = torch.empty(3, dtype = torch.long).random_(5)
    print(target)
    output = loss(input, target)
    print(output)
    b = output.backward()
    print(b)

def crossEntropyLoss2():
    # 创建一个 CrossEntropyLoss 对象
    criterion = nn.CrossEntropyLoss()
    # 假设我们有一个二分类问题，batch_size=3
    outputs = torch.randn(3, 2)  # 模型的输出
    targets = torch.tensor([0, 1, 0])  # 目标类别 
    # 计算损失
    print(outputs)
    print(targets)
    loss = criterion(outputs, targets) 
    print(loss)
        
crossEntropyLoss2()    
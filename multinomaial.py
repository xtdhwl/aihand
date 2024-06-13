import torch

weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
print(weights)

print(torch.multinomial(weights, 2))
print(torch.multinomial(weights, 4))
print(torch.multinomial(weights, 4, replacement=True))
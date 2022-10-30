



import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()

x = torch.tensor([
    [100, 20, 1],
    [30, 100, 88]
]).float()

l = torch.tensor([0, 1])

print(loss(x, l))
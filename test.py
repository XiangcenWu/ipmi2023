import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, coeff: float = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.coeff, None

class GRL(nn.Module):
    def __init__(self, l=1.):
        self.l = l
        super(GRL, self).__init__()

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.l)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.para = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        return x*self.para


model_0 = Model()
model_1 = GRL(3)
model_2 = Model()
x = torch.tensor(3.)
# o = model_2(model_0(x))
o = model_2(model_1(model_0(x)))
o = o*2
o.backward()

print(model_0.para.grad, model_2.para.grad)
print("dfsafsad")


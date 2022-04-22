import torch
import torch.nn as nn
nan = float("nan")
t = torch.Tensor([[2.0, 2.0], [0, 0], [.5, .5],])
t_o = torch.where(t >1, 2*t, 5*t)

l = nn.Linear(2, 2)
tf = l(t_o)
tf1=torch.where(tf>2.0,tf*3,tf*3)
(10 - tf1.sum()).backward()
print(l.weight.grad)
print(l.bias.grad)
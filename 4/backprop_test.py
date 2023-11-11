import torch

x = torch.ones(5, requires_grad=True)
print(x)
co = torch.tensor([0.9294, 0.4346, 0.5625, 0.3701, 0.9861])
y = x * co
y.retain_grad()
print(y)
# z = y.mean()
z = torch.tensor([0.2]*5)
print(z)
# z.backward()
y.backward(z)
print(x.grad)
print(y.grad)


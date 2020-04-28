import torch
import torch.autograd as ag
import torch.nn.functional as F
from torch.nn import Linear

# m = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.float32)
# m = (m.t() @ m).unsqueeze(-1)
# print(m)

m = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.float32)
m = m @ m.t()
e = m - torch.eye(m.shape[0]) * m
print(e)
q = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
r = (e * F.relu((q.unsqueeze(0) - q.unsqueeze(1)).norm(dim=2) - 1)).sum()
print(r)
# w = Linear(3, 2)
# q1 = q.unsqueeze(0)
# print(q1)
# q2 = q.unsqueeze(1)
# print(q2)
# p = q1 - q2
# print(p)
# n = w(p)
# print(n)
# r = m * n
# print(r)
# res = r.sum(1)
# print(res)
# print(torch.cat([q, res], dim=1))

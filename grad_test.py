# import torch
# import torch.autograd as ag
# import torch.nn.functional as F
# from torch.nn import Linear
# from sklearn.metrics import roc_auc_score
#
# import numpy as np
# a1 = np.array([1, 2, 3])
# a2 = np.array([0, 1, 1])
# a = a1[a2 == 1]
# print(a)
# a = a.sum(axis=1)
# print(a)
# m = np.argsort(a)[-2:]
# print(m)

# m = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.float32)
# m = (m.t() @ m).unsqueeze(-1)
# print(m)

# m = torch.tensor([[1], [0], [2]], dtype=torch.float32)
# print(torch.diag(m.view([-1])))

# m = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.float32)
# m = m @ m.t()
# e = m - torch.eye(m.shape[0]) * m
# print(e)
# q = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
# r = (e * F.relu((q.unsqueeze(0) - q.unsqueeze(1)).norm(dim=2) - 1)).sum()
# print(r)
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

import torch
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence

a = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.float32)
b = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
c = torch.tensor([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=torch.float32)
nfs = torch.cat([a, b, c])
print(nfs)
lstm = LSTM(2, 6, 1)
mnm = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1]],
                   dtype=torch.int64)

seqs = [nfs[n == 1, :] for n in mnm]
print(seqs)
lengths = [s.shape[0] for s in seqs]
print(lengths)
m = pad_sequence(seqs)
print(m)
output, (hn, cn) = lstm(m)
print(output)
ret = torch.cat([output[:lengths[i], i, :] for i in range(len(lengths))])
print(ret[:, :3], ret[:, 3:])


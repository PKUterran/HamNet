# import numpy as np
# import pandas as pd
import torch
# from visualize.trajectory import plt_trajectory
#
# a = np.array([[1, 2], [1, 3], [2, 3], [8, 3], [5, 3]])
# b = np.array([[3, 4], [5, 6], [2, 6], [1, 3], [9, 3]])
#
# plt_trajectory([a, b], name='test')

mol_node_matrix = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.float32)
norm_mnm = mol_node_matrix / mol_node_matrix.sum(dim=1).unsqueeze(-1)
print(norm_mnm.numpy())
dis_mask = norm_mnm.t() @ norm_mnm
print(dis_mask.pow(2).sum().numpy())

# qm = np.load('data/QM9/QM9_nano.npz', allow_pickle=True)
# print(qm['Atoms'])
# a, b, *c = range(10)
# print(*c)
# e1 = torch.tensor([[0., 0.5, 0.5], [0, 0, 0]], dtype=torch.float32)
# e2 = torch.tensor([[0.5, 0., 0.5]], dtype=torch.float32)
# e = torch.cat([e1, e2])
# e[2, [False, False, True]] = -1
# print(e)

# e[e == 0] = 9e-8
# print(e)
# m = m.norm()
# print(m)
# a = np.array([1, 2, 3, 3])
# b = np.array([0, 1, 1, 0])
# print(np.logical_and(a == 3, b))

# from data.reader import load_qm9
# from data.encode import *
#
# _, mols, _ = load_qm9(force_save=True)
# print(len(mols))
# print(mols[0])
# print(mols[10000])
# print(num_atom_features())
# print(num_bond_features())
# m = Chem.MolFromSmiles('CO')
# print(dir(m.GetBonds()[0]))
# print(m.GetBonds()[0].GetBeginAtom().GetSymbol())
# b = m.GetBonds()[0]
# print(1 - int(b.GetBondType() == Chem.rdchem.BondType.SINGLE and
#               b.GetBeginAtom().GetSymbol() in ['C', 'N'] and
#               b.GetEndAtom().GetSymbol() in ['C', 'N']))
# print(encode_smiles(np.array(['C#N', 'C=O'])))
# BATCH = 3
# mask = list(range(20))
# n_seg = int(len(mask) / BATCH) + 1
# mask_list = [mask[i::n_seg] for i in range(n_seg)]
# print(mask_list)

# a = [0.451, 0.492, 0.00358, 0.00415, 0.00528, 26.839 , 0.00120, 0.898 , 0.893 , 0.893 , 0.893 , 0.252]
# b = [1.189, 6.299, 0.016  , 0.039  , 0.040  , 202.017, 0.026  , 31.072, 31.072, 31.072, 31.072, 3.204]
# a = np.array(a)
# b = np.array(b)
# c = a / b
# print(c)
# print(c.sum())
# print(np.expand_dims(a, 0))

# from train.config import BBBPConfig
# if 'P_DIM' in dir(BBBPConfig):
#     print(BBBPConfig.P_DIM)

# from data.reader import load_qm9
# s, i, p = load_qm9(10, force_save=True, use_pos=True)
# print(i[1]['nf'])

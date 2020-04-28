# import numpy as np
# import pandas as pd
import torch

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

from data.qm9_reader import load_qm9
from data.encode import *

_, mols, _ = load_qm9(force_save=True)
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

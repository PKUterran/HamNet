# from data.gdb9_reader import load_gdb9
# from train.gdb9_trainer import train_gdb9
import numpy as np
from train.qm9_trainer import train_qm9
from train.hiv_trainer import train_hiv
from train.lipop_trainer import train_lipop
from train.tox21_trainer import train_tox21

# molecules = load_gdb9(50000)
# print(molecules.molecules.__len__())
# print(molecules.atom_set)
# print(molecules.bond_set)
# print(molecules.molecules[-1].show())

train_qm9(use_cuda=True,
          limit=-1,
          use_tqdm=True,
          # use_model='AMPNN',
          )

# train_lipop(use_cuda=True,
#             limit=-1,
#             use_tqdm=False,
#             # use_model='AMPNN',
#             )

# train_tox21(use_cuda=True,
#             limit=-1,
#             use_tqdm=False,
#             # use_model='AMPNN',
#             )

# train_hiv(use_cuda=True,
#           limit=-1,
#           use_tqdm=False,
#           # use_model='AMPNN',
#           dataset='BBBP',
#           )

# from data.gdb9_reader import load_gdb9
# from train.gdb9_trainer import train_gdb9
import numpy as np
from train.fitter import fit_qm9
from train.qm9_trainer import train_qm9

# from train.hiv_trainer import train_hiv
from train.lipop_trainer import train_lipop
from train.tox21_trainer import train_tox21

# fit_qm9(use_cuda=True,
#         limit=-1,
#         use_tqdm=False,
#         force_save=False,
#         model_save_path='net/server.pt')

train_qm9(use_cuda=True,
          limit=-1,
          use_tqdm=False,
          use_pos=False,
          force_save=False,
          position_encoder_path='net/server.pt',
          )

# train_lipop(use_cuda=True,
#             limit=-1,
#             use_tqdm=False,
#             force_save=False,
#             position_encoder_path='net/server.pt',
#             )

# train_tox21(use_cuda=True,
#             limit=-1,
#             use_tqdm=False,
#             force_save=True,
#             position_encoder_path='net/server.pt',
#             )

# train_hiv(use_cuda=True,
#           limit=-1,
#           use_tqdm=False,
#           # use_model='AMPNN',
#           dataset='BBBP',
#           )

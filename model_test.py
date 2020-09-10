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
#         # special_config={'HGN_LAYERS': 20, 'DISTURB': False, 'ITERATION': 10, 'LR': 1e-3, 'DISSIPATE': False},
#         model_save_path='net/server0731-nod.pt',
#         tag='new_no_dissipate_0731_server',
#         )

# train_qm9(use_cuda=True,
#           limit=20000,
#           use_tqdm=True,
#           use_pos=False,
#           force_save=True,
#           special_config={'LR': 10 ** -4, 'DECAY': 10 ** -5, 'ITERATION': 50},
#           position_encoder_path='',
#           tag='qm9_nopos'
#           )
# train_qm9(use_cuda=True,
#           limit=20000,
#           use_tqdm=True,
#           use_pos=False,
#           force_save=True,
#           special_config={'LR': 10 ** -4, 'DECAY': 10 ** -4.4, 'ITERATION': 50},
#           position_encoder_path='',
#           tag='qm9-small_nopos2'
#           )
# train_qm9(use_cuda=True,
#           limit=20000,
#           use_tqdm=False,
#           use_pos=False,
#           force_save=False,
#           special_config={'LR': 10 ** -4, 'DECAY': 10 ** -4.4, 'ITERATION': 50},
#           position_encoder_path='net/test0808.pt',
#           q_only=True,
#           tag='test'
#           )
# fit_qm9(use_cuda=True,
#         limit=20000,
#         use_tqdm=True,
#         force_save=True,
#         special_config={'HGN_LAYERS': 20, 'TAU': 0.01, 'DISTURB': False, 'ITERATION': 10,
#                         'LR': 1e-3, 'DISSIPATE': False,
#                         },
#         model_save_path='net/test0817-mpnn.pt',
#         tag='test0817-mpnn',
#         mpnn_pos_encode=True,
#         )
fit_qm9(use_cuda=True,
        limit=2000,
        use_tqdm=True,
        force_save=True,
        special_config={'HGN_LAYERS': 20, 'TAU': 0.01, 'DISTURB': False, 'ITERATION': 1,
                        'LR': 1e-3, 'DISSIPATE': True, 'LSTM': True,
                        },
        model_save_path='net/test0821-rdkit.pt',
        use_rdkit=True,
        tag='test0910-rdkit',
        )
fit_qm9(use_cuda=True,
        limit=2000,
        use_tqdm=True,
        force_save=True,
        special_config={'HGN_LAYERS': 20, 'TAU': 0.025, 'DISTURB': False, 'ITERATION': 10,
                        'LR': 1e-3, 'DISSIPATE': True, 'LSTM': True,
                        },
        model_save_path='net/test0910.pt',
        tag='test0910',
        )
# fit_qm9(use_cuda=True,
#         limit=20000,
#         use_tqdm=True,
#         force_save=True,
#         special_config={'HGN_LAYERS': 20, 'TAU': 0.01, 'DISTURB': False, 'ITERATION': 10,
#                         'LR': 1e-3, 'DISSIPATE': True, 'LSTM': False,
#                         },
#         model_save_path='net/test0821.pt',
#         tag='test0821-nolstm',
#         )
# train_qm9(use_cuda=True,
#           limit=20000,
#           use_tqdm=True,
#           use_pos=False,
#           force_save=True,
#           special_config={'LR': 10 ** -4, 'DECAY': 10 ** -4.4, 'ITERATION': 50},
#           position_encoder_path='net/test0808.pt',
#           tag='qm9-small_pos'
#           )
#
# train_qm9(use_cuda=True,
#           limit=-1,
#           use_tqdm=False,
#           use_pos=False,
#           force_save=False,
#           special_config={'DROPOUT': 0.5},
#           position_encoder_path='net/server.pt',
#           tag='qm9_dropout5_pos'
#           )

# train_lipop(use_cuda=True,
#             limit=-1,
#             use_tqdm=True,
#             force_save=True,
#             position_encoder_path='net/rdkit.pt',
#             tag='pos',
#             )

# train_lipop(use_cuda=True,
#             limit=-1,
#             use_tqdm=True,
#             force_save=True,
#             position_encoder_path='net/test0824.pt',
#             tag='test-freesolv',
#             dataset='FreeSolv',
#             )
# train_lipop(use_cuda=True,
#             limit=-1,
#             use_tqdm=False,
#             force_save=False,
#             position_encoder_path='',
#             tag='nopos',
#             )

# train_tox21(use_cuda=True,
#             limit=-1,
#             use_tqdm=False,
#             force_save=True,
#             position_encoder_path='net/server0731.pt',
#             tag='pos',
#             )
# train_tox21(use_cuda=True,
#             limit=-1,
#             use_tqdm=False,
#             force_save=True,
#             position_encoder_path='',
#             tag='nopos',
#             )

# train_hiv(use_cuda=True,
#           limit=-1,
#           use_tqdm=False,
#           # use_model='AMPNN',
#           dataset='BBBP',
#           )

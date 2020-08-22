from train.qm9_trainer import train_qm9

PE_PATH = 'net/0821.pt'

train_qm9(use_cuda=True,
          limit=-1,
          use_tqdm=False,
          use_pos=False,
          force_save=True,
          position_encoder_path='',
          tag='qm9_nopos'
          )

train_qm9(use_cuda=True,
          limit=-1,
          use_tqdm=False,
          use_pos=True,
          force_save=True,
          position_encoder_path='',
          tag='qm9_3pos'
          )

train_qm9(use_cuda=True,
          limit=-1,
          use_tqdm=False,
          use_pos=False,
          force_save=True,
          position_encoder_path=PE_PATH,
          tag='qm9_pos'
          )

train_qm9(use_cuda=True,
          limit=-1,
          use_tqdm=False,
          use_pos=False,
          force_save=True,
          position_encoder_path=PE_PATH,
          q_only=True,
          tag='qm9_q_only'
          )

from train.tox21_trainer import train_tox21

PE_PATH = 'net/0821.pt'

train_tox21(use_cuda=True,
            limit=-1,
            use_tqdm=False,
            force_save=True,
            position_encoder_path='',
            tag='tox21_nopos'
            )
train_tox21(use_cuda=True,
            limit=-1,
            use_tqdm=False,
            force_save=False,
            position_encoder_path=PE_PATH,
            tag='tox21_pos'
            )

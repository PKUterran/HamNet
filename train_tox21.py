from train.tox21_trainer import train_tox21

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
            position_encoder_path='net/server0808.pt',
            tag='tox21_pos'
            )

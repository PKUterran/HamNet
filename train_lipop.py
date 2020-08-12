from train.lipop_trainer import train_lipop

train_lipop(use_cuda=True,
            limit=-1,
            use_tqdm=False,
            force_save=True,
            position_encoder_path='',
            tag='lipop_nopos'
            )
train_lipop(use_cuda=True,
            limit=-1,
            use_tqdm=False,
            force_save=False,
            position_encoder_path='net/server0808.pt',
            tag='lipop_pos'
            )

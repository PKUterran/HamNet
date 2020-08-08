from train.fitter import fit_qm9
from train.qm9_trainer import train_qm9

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
fit_qm9(use_cuda=True,
        limit=-1,
        use_tqdm=False,
        force_save=True,
        model_save_path='net/server0808.pt',
        tag='0808',
        )

train_qm9(use_cuda=True,
          limit=-1,
          use_tqdm=False,
          use_pos=False,
          force_save=True,
          position_encoder_path='net/test0808.pt',
          tag='qm9_pos'
          )

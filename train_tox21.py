from train.tox21_trainer import train_tox21
from utils.seed import SEEDS

PE_PATH = 'net/0821.pt'

for seed in SEEDS:
    train_tox21(seed=seed,
                use_cuda=True,
                limit=-1,
                use_tqdm=False,
                force_save=True,
                position_encoder_path='',
                tag='tox21_nopos@{}'.format(seed)
                )
    train_tox21(seed=seed,
                use_cuda=True,
                limit=-1,
                use_tqdm=False,
                force_save=False,
                position_encoder_path=PE_PATH,
                tag='tox21_pos@{}'.format(seed)
                )

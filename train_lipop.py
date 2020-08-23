from train.lipop_trainer import train_lipop
from utils.seed import SEEDS

PE_PATH = 'net/0821.pt'

for seed in SEEDS:
    train_lipop(seed=seed,
                use_cuda=True,
                limit=-1,
                use_tqdm=False,
                force_save=True,
                position_encoder_path='',
                tag='lipop_nopos@{}'.format(seed)
                )
    train_lipop(seed=seed,
                use_cuda=True,
                limit=-1,
                use_tqdm=False,
                force_save=False,
                position_encoder_path=PE_PATH,
                tag='lipop_pos@{}'.format(seed)
                )

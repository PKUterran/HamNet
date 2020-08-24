from train.tox21_trainer import train_tox21
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
arg = parser.parse_args()
seed = arg.seed

PE_PATH = 'net/0821.pt'

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

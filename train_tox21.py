from train.tox21_trainer import train_tox21
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--pos', type=int, default=1)
arg = parser.parse_args()
seed = arg.seed
pos = arg.pos

PE_PATH = 'net/0909.pt'
RDKIT_PATH = 'net/rdkit.pt'

if pos == 1:
    pep = PE_PATH
    tag = 'tox21_pos@{}'.format(seed)
elif pos == 2:
    pep = RDKIT_PATH
    tag = 'tox21_rdpos@{}'.format(seed)
else:
    pep = ''
    tag = 'tox21_nopos@{}'.format(seed)


train_tox21(seed=seed,
            use_cuda=True,
            limit=-1,
            use_tqdm=False,
            force_save=False,
            position_encoder_path=pep,
            tag=tag,
            )

from train.lipop_trainer import train_lipop
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
    tag = 'freesolv_pos@{}'.format(seed)
elif pos == 2:
    pep = RDKIT_PATH
    tag = 'freesolv_rdpos@{}'.format(seed)
else:
    pep = ''
    tag = 'freesolv_nopos@{}'.format(seed)

train_lipop(seed=seed,
            use_cuda=True,
            limit=-1,
            use_tqdm=False,
            force_save=False,
            position_encoder_path=pep,
            tag=tag,
            dataset='FreeSolv',
            special_config={
                'LR': 10 ** -2.5,
                'GAMMA': 0.995,
                'DECAY': 10 ** -5.0,
                'DROPOUT': 0.2,
                'ITERATION': 400,
                'BATCH': 1,
                'PACK': 128,
                'EVAL': 2000,
                'HE_DIM': 120,
                'C_DIMS': [120, 120],
                'F_DIM': 120,
                'M_RADIUS': 2,
            },
            )

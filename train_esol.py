from train.lipop_trainer import train_lipop
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--pos', type=int, default=0)
arg = parser.parse_args()
seed = arg.seed
pos = arg.pos

PE_PATH = 'net/0909.pt'
RDKIT_PATH = 'net/rdkit.pt'

if pos == 1:
    pep = PE_PATH
    tag = 'esol_pos@{}'.format(seed)
elif pos == 2:
    pep = RDKIT_PATH
    tag = 'esol_rdpos@{}'.format(seed)
else:
    pep = ''
    tag = 'esol_nopos@{}'.format(seed)

train_lipop(seed=seed,
            use_cuda=True,
            limit=-1,
            use_tqdm=False,
            force_save=False,
            position_encoder_path=pep,
            tag=tag,
            dataset='ESOL',
            special_config={
                'LR': 10 ** -3.0,
                'GAMMA': 0.98,
                'DECAY': 10 ** -5.0,
                'DROPOUT': 0.5,
                'ITERATION': 200,
                'BATCH': 2,
                'PACK': 40,
                'EVAL': 2000,
                'HE_DIM': 200,
                'C_DIMS': [200, 200],
                'F_DIM': 200,
                'M_RADIUS': 2,
            },
            )

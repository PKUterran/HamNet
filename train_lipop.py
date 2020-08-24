from train.lipop_trainer import train_lipop
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--pos', type=bool, default=True)
arg = parser.parse_args()
seed = arg.seed
pos = arg.pos

PE_PATH = 'net/0821.pt'

train_lipop(seed=seed,
            use_cuda=True,
            limit=-1,
            use_tqdm=False,
            force_save=False,
            position_encoder_path=PE_PATH if pos else '',
            tag='lipop_pos@{}'.format(seed) if pos else 'lipop_nopos@{}'.format(seed),
            )

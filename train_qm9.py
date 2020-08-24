from train.qm9_trainer import train_qm9
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
arg = parser.parse_args()
seed = arg.seed

PE_PATH = 'net/0821.pt'

train_qm9(seed=seed,
          use_cuda=True,
          limit=-1,
          use_tqdm=False,
          use_pos=False,
          force_save=True,
          position_encoder_path='',
          tag='qm9_nopos@{}'.format(seed)
          )

train_qm9(seed=seed,
          use_cuda=True,
          limit=-1,
          use_tqdm=False,
          use_pos=True,
          force_save=True,
          position_encoder_path='',
          tag='qm9_3pos@{}'.format(seed)
          )

train_qm9(seed=seed,
          use_cuda=True,
          limit=-1,
          use_tqdm=False,
          use_pos=False,
          force_save=True,
          position_encoder_path=PE_PATH,
          tag='qm9_pos@{}'.format(seed)
          )

train_qm9(seed=seed,
          use_cuda=True,
          limit=-1,
          use_tqdm=False,
          use_pos=False,
          force_save=True,
          position_encoder_path=PE_PATH,
          q_only=True,
          tag='qm9_q_only@{}'.format(seed)
          )

import argparse
from train.fitter import fit_qm9

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=32)
arg = parser.parse_args()
dim = arg.dim

TAG = '0920-d{}'.format(dim)
fit_qm9(use_cuda=True,
        limit=-1,
        use_tqdm=False,
        force_save=False,
        model_save_path='net/{}.pt'.format(TAG),
        special_config={'PQ_DIM': dim},
        tag=TAG,
        )

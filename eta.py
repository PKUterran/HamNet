import argparse
from train.fitter import fit_qm9

parser = argparse.ArgumentParser()
parser.add_argument('--eta', type=float, default=0.025)
arg = parser.parse_args()
eta = arg.eta

TAG = '0920-e{}'.format(eta)
fit_qm9(use_cuda=True,
        limit=-1,
        use_tqdm=False,
        force_save=False,
        model_save_path='net/{}.pt'.format(TAG),
        special_config={'TAU': eta},
        tag=TAG,
        )

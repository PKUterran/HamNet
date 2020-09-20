import argparse
from train.fitter import fit_qm9

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, default=20)
arg = parser.parse_args()
layer = arg.layer

TAG = '0920-l{}'.format(layer)
fit_qm9(use_cuda=True,
        limit=-1,
        use_tqdm=False,
        force_save=False,
        model_save_path='net/{}.pt'.format(TAG),
        special_config={'HGN_LAYERS': layer},
        tag=TAG,
        )

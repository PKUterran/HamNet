from train.fitter import fit_qm9

# use MPNN to encode p & q
TAG = '0909-mpnn'
fit_qm9(use_cuda=True,
        limit=-1,
        use_tqdm=False,
        force_save=False,
        model_save_path='net/{}.pt'.format(TAG),
        mpnn_pos_encode=True,
        tag=TAG,
        )

# no ADJ3 loss
TAG = '0909-noadj'
fit_qm9(use_cuda=True,
        limit=-1,
        use_tqdm=False,
        force_save=False,
        model_save_path='net/{}.pt'.format(TAG),
        special_config={'GAMMA_A': 0.000},
        tag=TAG,
        )

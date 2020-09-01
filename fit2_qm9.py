from train.fitter import fit_qm9

# normal
TAG = '0821-noadj'
fit_qm9(use_cuda=True,
        limit=-1,
        use_tqdm=False,
        force_save=False,
        model_save_path='net/{}.pt'.format(TAG),
        special_config={'GAMMA_A': 0.000},
        tag=TAG,
        )

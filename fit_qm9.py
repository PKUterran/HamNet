from train.fitter import fit_qm9

# rdkit
TAG = 'rdkit'
fit_qm9(use_cuda=True,
        limit=-1,
        use_tqdm=False,
        force_save=True,
        model_save_path='net/{}.pt'.format(TAG),
        special_config={'ITERATION': 1},
        tag=TAG,
        use_rdkit=True,
        )

# normal
TAG = '0909'
fit_qm9(use_cuda=True,
        limit=-1,
        use_tqdm=False,
        force_save=False,
        model_save_path='net/{}.pt'.format(TAG),
        tag=TAG,
        )

# no dissipation
TAG = '0909-nodis'
fit_qm9(use_cuda=True,
        limit=-1,
        use_tqdm=False,
        force_save=False,
        model_save_path='net/{}.pt'.format(TAG),
        special_config={'DISSIPATE': False},
        tag=TAG,
        )

# no hamiltonian kernel
TAG = '0909-noham'
fit_qm9(use_cuda=True,
        limit=-1,
        use_tqdm=False,
        force_save=False,
        model_save_path='net/{}.pt'.format(TAG),
        special_config={'HGN_LAYERS': 0},
        tag=TAG,
        )

# no LSTM when producing p0 & q0
TAG = '0909-nolstm'
fit_qm9(use_cuda=True,
        limit=-1,
        use_tqdm=False,
        force_save=False,
        model_save_path='net/{}.pt'.format(TAG),
        special_config={'LSTM': False},
        tag=TAG,
        )

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

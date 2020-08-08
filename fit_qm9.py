from train.fitter import fit_qm9

fit_qm9(use_cuda=True,
        limit=-1,
        use_tqdm=False,
        force_save=True,
        model_save_path='net/server0808.pt',
        tag='0808',
        )

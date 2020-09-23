import matplotlib.pyplot as plt
import json

POS_PATH = 'pos/'
FIG_PATH = 'fig/'


def plt_dl(tag: str):
    with open(POS_PATH + tag + '.json', encoding='utf-8') as fp:
        d = json.load(fp)
        logs = d['logs']

        epochs = []
        train_dls = []
        validate_dls = []
        test_dls = []
        for log in logs:
            epochs.append(log['epoch'])
            train_dls.append(log['train_rmsd_metric'])
            validate_dls.append(log['evaluate_rmsd_metric'])
            test_dls.append(log['test_rmsd_metric'])

        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_dls, label='train', c='red')
        plt.plot(epochs, validate_dls, label='validate', c='blue')
        plt.plot(epochs, test_dls, label='test', c='green')
        plt.legend()
        plt.savefig(FIG_PATH + tag + '.png')
        plt.close()


plt_dl('0920')
plt_dl('0920-noham')
plt_dl('0920-noadj')
plt_dl('0909-nodis')
plt_dl('0909-mpnn')

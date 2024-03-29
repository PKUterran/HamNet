import json
import numpy as np

SEEDS = [
    16880611,
    17760704,
    17890714,
    19491001,
    19900612
]

tag_path = (
    # ['pos', ['pos/0821.json'], False],
    # ['pos no dis', ['pos/0821-nodis.json'], False],
    # ['pos no ham', ['pos/0821-noham.json'], False],
    # ['pos no lstm', ['pos/0821-nolstm.json'], False],
    # ['tox21 no pos', ['TOX21/tox21_nopos@{}.json'.format(seed) for seed in SEEDS], True],
    # ['tox21 pos', ['TOX21/tox21_pos@{}.json'.format(seed) for seed in SEEDS], True],
    # ['tox21 rdpos', ['TOX21/tox21_rdpos@{}.json'.format(seed) for seed in SEEDS], True],
    # ['lipop no pos', ['Lipop/lipop_nopos@{}.json'.format(seed) for seed in SEEDS], False],
    # ['lipop pos', ['Lipop/lipop_pos@{}.json'.format(seed) for seed in SEEDS], False],
    # ['lipop rdpos', ['Lipop/lipop_rdpos@{}.json'.format(seed) for seed in SEEDS], False],
    ['freesolv pos', ['Lipop/freesolv_pos@{}.json'.format(seed) for seed in SEEDS], False],
    ['freesolv no pos', ['Lipop/freesolv_nopos@{}.json'.format(seed) for seed in SEEDS], False],
    ['freesolv rdpos', ['Lipop/freesolv_rdpos@{}.json'.format(seed) for seed in SEEDS], False],
    # ['lipop no pos', ['Lipop/lipop_nopos@16880611.json'], False],
    # ['lipop pos', ['Lipop/lipop_pos@16880611.json'], False],
    # ['lipop rdpos', ['Lipop/lipop_rdpos@16880611.json'], False],
    # ['lipop no pos', ['Lipop/lipop_nopos@17760704.json'], False],
    # ['lipop pos', ['Lipop/lipop_pos@17760704.json'], False],
    # ['lipop rdpos', ['Lipop/lipop_rdpos@17760704.json'], False],
    # ['lipop no pos', ['Lipop/lipop_nopos@17890714.json'], False],
    # ['lipop pos', ['Lipop/lipop_pos@17890714.json'], False],
    # ['lipop rdpos', ['Lipop/lipop_rdpos@17890714.json'], False],
    # ['lipop no pos', ['Lipop/lipop_nopos@19491001.json'], False],
    # ['lipop pos', ['Lipop/lipop_pos@19491001.json'], False],
    # ['lipop rdpos', ['Lipop/lipop_rdpos@19491001.json'], False],
    # ['lipop no pos', ['Lipop/lipop_nopos@19900612.json'], False],
    # ['lipop pos', ['Lipop/lipop_pos@19900612.json'], False],
    # ['lipop rdpos', ['Lipop/lipop_rdpos@19900612.json'], False],
    # ['QM9 no pos', ['QM9/qm9_nopos@16880611.json'], False],
    # ['QM9 3 pos', ['QM9/qm9_3pos@16880611.json'], False],
    # ['QM9 pos', ['QM9/qm9_pos@16880611.json'], False],
    # ['QM9 q only', ['QM9/qm9_q_only@16880611.json'], False],
    # ['QM9 no pos', ['QM9/qm9_nopos@{}.json'.format(seed) for seed in SEEDS[:4]], False],
    # ['QM9 3 pos', ['QM9/qm9_3pos@{}.json'.format(seed) for seed in SEEDS[:3]], False],
    # ['QM9 pos', ['QM9/qm9_pos@{}.json'.format(seed) for seed in SEEDS[:3]], False],
    # ['QM9 q only', ['QM9/qm9_q_only@{}.json'.format(seed) for seed in SEEDS[:3]], False],
    ['ESOL pos', ['Lipop/esol_pos@{}.json'.format(seed) for seed in SEEDS], False],
    ['ESOL no pos', ['Lipop/esol_nopos@{}.json'.format(seed) for seed in SEEDS], False],
    ['ESOL rdpos', ['Lipop/esol_rdpos@{}.json'.format(seed) for seed in SEEDS], False],
)

for tag, paths, higher_is_better in tag_path:
    cor_tests = []
    for path in paths:
        with open(path) as fp:
            d = json.load(fp)
            logs = d['logs']
            best_val = -1e8 if higher_is_better else 1e8
            cor_test = 0
            for dd in logs:
                val = dd['evaluate_metric']
                test = dd['test_metric']
                if (higher_is_better and val > best_val) or (not higher_is_better and val < best_val):
                    # print(dd['epoch'], dd['train_metric'], val, test)
                    best_val = val
                    cor_test = test
            cor_tests.append(cor_test)
            # print('{}: {}'.format(tag, cor_test))

    avg = np.average(cor_tests)
    bound = np.std(cor_tests)
    print('{}: {:.3f} +- {:.3f}'.format(tag, avg, bound))

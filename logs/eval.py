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
    # ['tox21 no pos', ['TOX21/tox21_nopos.json'], True],
    # ['tox21 pos', ['TOX21/tox21_pos.json'], True],
    # ['lipop no pos', ['Lipop/lipop_nopos@{}.json'.format(seed) for seed in SEEDS[:1]], False],
    ['lipop pos', ['Lipop/lipop_pos@{}.json'.format(seed) for seed in SEEDS], False],
    # ['QM9 no pos', ['QM9/qm9_nopos.json'], False],
    # ['QM9 3 pos', ['QM9/qm9_3pos.json'], False],
    # ['QM9 pos', ['QM9/qm9_pos.json'], False],
    # ['QM9 q only', ['QM9/qm9_q_only.json'], False],
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

    print('{}: {:.5f}'.format(tag, np.average(cor_tests)))

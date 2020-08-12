import json
import numpy as np

tag_path = (
    ['lipop no pos', 'Lipop/lipop_nopos.json', False],
    ['lipop pos', 'Lipop/lipop_pos.json', False],
    ['QM9 no pos', 'QM9/qm9_nopos.json', False],
    ['QM9 3 pos', 'QM9/qm9_3pos.json', False],
    ['QM9 pos', 'QM9/qm9_pos.json', False],
)

for tag, path, higher_is_better in tag_path:
    with open(path) as fp:
        d = json.load(fp)
        logs = d['logs']
        best_val = -1e8 if higher_is_better else 1e8
        cor_test = 0
        for dd in logs:
            val = dd['evaluate_metric']
            test = dd['test_metric']
            if higher_is_better and val > best_val or not higher_is_better and val < best_val:
                cor_test = test
        print('{}: {}'.format(tag, cor_test))

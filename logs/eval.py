import json
import numpy as np

tag_path = (
    ['lipop pos', 'Lipop/pos.json', False],
    ['lipop no pos', 'Lipop/nopos.json', False],
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

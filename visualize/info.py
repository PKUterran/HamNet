import json
import torch
import numpy as np

from net.models import PositionEncoder

PE_PATH = '../net/0821.pt'
RDKIT_PATH = '../net/rdkit.pt'
JSON_PATH = '../Visualization/json/'


def output(tag, smiles, pos):
    d = {'tag': tag, 'smiles': smiles, 'pos': pos}
    print(tag, smiles)
    with open('{}{}.json'.format(JSON_PATH, tag), 'w+', encoding='utf-8') as fp:
        json.dump(d, fp)


if __name__ == '__main__':
    test_tuples = [
        ("cyclohexane", "C1CCCCC1"),
        ("benzene", "c1ccccc1"),
        ("benzenethiol", "c1ccc(cc1)S"),
        ("but-1-yne", "CCC#C"),
        ("cyclohepta-1,3,5-triene", "C1C=CC=CC=C1"),
        ("anthracene", "c1ccc2cc3ccccc3cc2c1"),
        ("1,4-diamino-9,10-anthracenedione", "c1ccc2c(c1)C(=O)c3c(ccc(c3C2=O)N)N"),
    ]

    pe = torch.load(PE_PATH)
    rdk = torch.load(RDKIT_PATH)
    for tag, smiles in test_tuples:
        qs = pe.produce(smiles)
        for i, q in enumerate(qs):
            output('{}-{}'.format(tag, i), smiles, q.tolist())
        output('{}-g'.format(tag), smiles, (sum(qs) / len(qs)).tolist())

        pos = rdk.produce(smiles)
        output('{}-rdk'.format(tag), smiles, pos.tolist())

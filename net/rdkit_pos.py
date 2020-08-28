import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem


def rdkit_pos(smiles):
    ret = []
    for smile in smiles:
        m = Chem.MolFromSmiles(smile)
        position = [[0., 0., 0.] for _ in range(len(m.GetAtoms()))]
        try:
            AllChem.EmbedMolecule(m)
            conf = m.GetConformer()
            position = [list(conf.GetAtomPosition(s)) for s in range(len(m.GetAtoms()))]
        except ValueError:
            pass
        finally:
            ret.extend(position)
    return ret


if __name__ == '__main__':
    print(rdkit_pos(['C1CC(=O)C=C1N']))

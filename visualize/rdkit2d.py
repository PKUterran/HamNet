import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Draw

GRAPH_DIR = 'graphs/'


def rdkit2d(smiles: str, tag: str):
    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToFile(mol, GRAPH_DIR + tag + '.png')


if __name__ == '__main__':
    rdkit2d('C1C=CCO1', 'gdb_158')
    rdkit2d('CC1CN=CO1', 'gdb_460')
    rdkit2d('O=CC1CCO1', 'gdb_559')
    rdkit2d('CCCN1CC1', 'gdb_590')


'''
158: C1C=CCO1
460: CC1CN=CO1
559: O=CC1CCO1
590: CCCN1CC1
'''
import rdkit.Chem as Chem
from rdkit.Chem import Draw

mol = Chem.MolFromSmiles('O=CC(C#C)C1CC1')
Draw.MolToFile(mol, 'temp.png')

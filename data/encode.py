import numpy as np
from rdkit import Chem


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


ATOMS = [
    'B',
    'C',
    'N',
    'O',
    'F',
    'Si',
    'P',
    'S',
    'Cl',
    'As',
    'Se',
    'Br',
    'Te',
    'I',
    'At',
    'other'
]


def atom_features(atom,
                  explicit_H=True,
                  use_chirality=True,
                  default_atoms=None):
    if not default_atoms:
        default_atoms = ATOMS
    results = \
        one_of_k_encoding_unk(atom.GetSymbol(), default_atoms) + \
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) + \
        [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
        one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, 'other'
        ]) + \
        [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results, dtype=np.int)


def bond_features(bond):
    use_chirality = True
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats, dtype=np.int)


def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))


def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))


def encode_smiles(smiles: np.ndarray) -> list:
    ret = []
    print('Start encoding...')
    cnt = 0
    for smile in smiles:
        info = {}
        mol = Chem.MolFromSmiles(smile)
        info['nf'] = np.stack([atom_features(a) for a in mol.GetAtoms()])
        info['ef'] = np.stack([bond_features(b) for b in mol.GetBonds()]
                              # + [bond_features(b) for b in mol.GetBonds()]
                              ) if len(mol.GetBonds()) \
            else np.zeros(shape=[0, 10], dtype=np.int)
        info['us'] = np.array([b.GetBeginAtomIdx() for b in mol.GetBonds()]
                              # + [b.GetEndAtomIdx() for b in mol.GetBonds()]
                              , dtype=np.int)
        info['vs'] = np.array([b.GetEndAtomIdx() for b in mol.GetBonds()]
                              # + [b.GetBeginAtomIdx() for b in mol.GetBonds()]
                              , dtype=np.int)
        info['em'] = np.array([1 - int(b.GetBondType() == Chem.rdchem.BondType.SINGLE
                                       and b.GetBeginAtom().GetSymbol() in ['C', 'N']
                                       and b.GetEndAtom().GetSymbol() in ['C', 'N']
                                       and not b.IsInRing())
                               for b in mol.GetBonds()]
                              # + [1 - int(b.GetBondType() == Chem.rdchem.BondType.SINGLE
                              #            and b.GetBeginAtom().GetSymbol() in ['C', 'N']
                              #            and b.GetEndAtom().GetSymbol() in ['C', 'N']
                              #            and not b.IsInRing())
                              #    for b in mol.GetBonds()]
                              , dtype=np.int)
        ret.append(info)
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt, 'encoded.')
    return ret

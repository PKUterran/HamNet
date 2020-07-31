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

ATOMS_MASS = {
    'B': 10.811,
    'C': 12.0107,
    'N': 14.0067,
    'O': 15.9994,
    'F': 18.9984032,
    'Si': 28.2855,
    'P': 30.973762,
    'S': 32.065,
    'Cl': 35.453,
    'As': 74.92160,
    'Se': 78.96,
    'Br': 79.904,
    'Te': 127.60,
    'I': 126.90447,
    'At': 209.9871,
    'other': 50.0,
}


def get_atoms_massive_matrix(atoms: list) -> np.ndarray:
    massive = []
    for a in atoms:
        massive.append(ATOMS_MASS[a])
    massive = np.vstack([np.array(massive).reshape([-1, 1]), np.zeros([num_atom_features() - len(atoms), 1])])
    return massive


def get_default_atoms_massive_matrix() -> np.ndarray:
    return get_atoms_massive_matrix(ATOMS)


def atom_features(atom,
                  explicit_H=True,
                  use_chirality=True,
                  default_atoms=None,
                  position=0,
                  max_position=0):
    if not default_atoms:
        default_atoms = ATOMS
    results = \
        one_of_k_encoding_unk(atom.GetSymbol(), default_atoms) + \
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) + \
        [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
        one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, 'other'
        ]) + \
        [atom.GetIsAromatic()]
    if max_position:
        results = results + one_of_k_encoding_unk(position, list(range(max_position)))
    # In case of explicit hydrogen(QM8, QM9-small), avoid calling `GetTotalNumHs`
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


def encode_smiles(smiles: np.ndarray, return_mask=False,
                  mol_atom_pos: list = None,
                  max_position=0, central_atoms=0, max_dis=0):
    ret = []
    mask = []
    similar = 0
    print('Start encoding...')
    cnt = 0
    for idx, smile in enumerate(smiles):
        info = {}
        mol = Chem.MolFromSmiles(smile)
        atom_pos = mol_atom_pos[idx] if mol_atom_pos else None
        if return_mask:
            if not mol:
                cnt += 1
                continue
            else:
                mask.append(cnt)
        if central_atoms * max_dis:
            cad, sim = get_central_atoms_dis(mol, central_atoms, max_dis)
            similar += sim
            info['nf'] = np.concatenate([np.stack([atom_features(a, position=i, max_position=max_position)
                                                   for i, a in enumerate(mol.GetAtoms())]),
                                         cad], axis=1)
        else:
            info['nf'] = np.stack([atom_features(a, position=i, max_position=max_position)
                                   for i, a in enumerate(mol.GetAtoms())])
        if mol_atom_pos:
            # print(idx)
            # print(info['nf'])
            # print(atom_pos)
            info['nf'] = np.concatenate([info['nf'], atom_pos], axis=1)

        info['ef'] = np.stack([bond_features(b) for b in mol.GetBonds()]
                              # + [bond_features(b) for b in mol.GetBonds()]
                              ) if len(mol.GetBonds()) else np.zeros(shape=[0, 10], dtype=np.int)
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
    print('Encoded:', len(ret))
    print('Similarity:', similar / len(smiles))
    if return_mask:
        return ret, mask
    return ret


def get_central_atoms_dis(mol, atoms_num, max_dis) -> (np.ndarray, int):
    n = len(mol.GetAtoms())
    distance_matrix = np.array(Chem.GetDistanceMatrix(mol), dtype=np.int)
    distance_matrix[distance_matrix > max_dis] = max_dis
    if n <= atoms_num:
        central_idx = np.array([i % n for i in range(atoms_num)], dtype=np.int)
        central_atoms_dis = distance_matrix[:, central_idx]
        num_similar = 0
    else:
        # print(distance_matrix)
        different = [(distance_matrix[:, i: i + 1] != distance_matrix[i: i + 1, :]).astype(np.int) for i in range(n)]
        # print(different[0])
        # print(different[1])
        specialty = [np.sum(d) for d in different]
        # print(specialty)
        total = sum(different)
        central_alive = np.array([1] * n, dtype=np.int)
        # print(central_alive)
        order = np.argsort(specialty)
        # print(order)

        temp_dif = total
        while sum(central_alive) > atoms_num:
            flag = False  # if remove an atom
            for o in order:
                # print((((total - different[o]) == 0) == (np.eye(n) == 1)).sum() == n * n)
                if (((temp_dif - different[o]) == 0) == (np.eye(n) == 1)).sum() == n * n:
                    central_alive[o] = 0
                    temp_dif -= different[o]
                    flag = True
                    break
            if flag:
                continue

            for o in order:
                if central_alive[o]:
                    central_alive[o] = 0
                    temp_dif -= different[o]
                    if sum(central_alive) == atoms_num:
                        break
            break

        # print(temp_dif)
        num_similar = np.logical_and((temp_dif == 0), (np.eye(n) == 0)).sum() / 2
        central_atoms_dis = distance_matrix[:, central_alive == 1]
    # print(central_atoms_dis)
    # print(central_atoms_dis.shape)
    central_atoms_dis = np.eye(max_dis + 1)[central_atoms_dis]
    # print(central_atoms_dis)
    central_atoms_dis = np.reshape(central_atoms_dis, newshape=[n, atoms_num * (max_dis + 1)])
    # print(central_atoms_dis)
    return central_atoms_dis, num_similar

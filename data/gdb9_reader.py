import numpy as np
import pandas as pd

from .objects import Molecule, MoleculeSet


def load_gdb9(max_num: int = -1) -> (MoleculeSet, np.ndarray):
    molecules = MoleculeSet()
    cnt = 0

    with open('data/gdb9/gdb9.sdf') as content:
        while True:
            lines = []
            while True:
                line = content.readline()
                if not line:
                    break
                line = line.strip()
                lines.append(line)
                if line == '$$$$':
                    break
            if len(lines) == 0:
                break
            n, e, *ex = [int(t) for t in lines[3].split()[:8]]
            molecule = Molecule(lines[0], [int(i) for i in ex])
            i = 4
            for _ in range(n):
                x, y, z, a, *ex = lines[i].split()
                molecule.add_atom(token=a, pos=[float(x), float(y), float(z)],
                                  extensive=[int(i) for i in ex], edge_ex_len=4)
                i += 1

            for _ in range(e):
                u, v, t, *ex = lines[i].split()
                molecule.add_bond(int(u) - 1, int(v) - 1, int(t), extensive=[int(i) for i in ex])
                i += 1

            molecules.add(molecule)

            cnt += 1
            if cnt % 1e4 == 0:
                print(cnt, 'loaded.')
            if max_num != -1 and cnt >= max_num:
                break

    properties = np.array(pd.read_csv('data/gdb9/gdb9.sdf.csv'))
    if max_num != -1 and max_num < properties.shape[0]:
        properties = properties[:max_num, :]
    print(properties.shape[0], 'molecules and', properties.shape[1] - 1, 'properties.')
    assert properties.shape[0] == len(molecules.molecules)

    return molecules, properties


def load_mol_atom_pos(max_num: int = -1) -> list:
    mol_atom_pos = []
    cnt = 0

    with open('data/gdb9/gdb9.sdf') as content:
        while True:
            lines = []
            atom_pos = []
            while True:
                line = content.readline()
                if not line:
                    break
                line = line.strip()
                lines.append(line)
                if line == '$$$$':
                    break
            if len(lines) == 0:
                break
            n, *_ = [int(t) for t in lines[3].split()[:8]]
            for i in range(4, 4 + n):
                x, y, z, a, *_ = lines[i].split()
                if a == 'H':
                    continue
                atom_pos.append([x, y, z])
            mol_atom_pos.append(np.array(atom_pos, dtype=np.float))

            cnt += 1
            if cnt % 1e4 == 0:
                print(cnt, 'loaded.')
            if max_num != -1 and cnt >= max_num:
                break

    return mol_atom_pos


def load_mol_atom_pos_symbol(max_num: int = -1) -> (list, list):
    mol_atom_pos = []
    mol_atom_symbol = []
    cnt = 0

    with open('data/gdb9/gdb9.sdf') as content:
        while True:
            lines = []
            atom_pos = []
            atom_symbol = []
            while True:
                line = content.readline()
                if not line:
                    break
                line = line.strip()
                lines.append(line)
                if line == '$$$$':
                    break
            if len(lines) == 0:
                break
            n, *_ = [int(t) for t in lines[3].split()[:8]]
            for i in range(4, 4 + n):
                x, y, z, a, *_ = lines[i].split()
                if a == 'H':
                    continue
                atom_pos.append([x, y, z])
                atom_symbol.append(a)
            mol_atom_pos.append(np.array(atom_pos, dtype=np.float))
            mol_atom_symbol.append(atom_symbol)

            cnt += 1
            if cnt % 1e4 == 0:
                print(cnt, 'loaded.')
            if max_num != -1 and cnt >= max_num:
                break

    return mol_atom_pos, mol_atom_symbol


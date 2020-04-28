import numpy as np

from .config import *


class Atom:
    def __init__(self, lics: int = 0, token: str = '', pos: list = None, extensive: list = None):
        if lics:
            self.lics = lics
        elif token in LICS_MAP.keys():
            self.lics = LICS_MAP[token]
        else:
            assert False, 'Undefined atom type: lics = {}, token = {}.'.format(lics, token)
        self.pos = pos
        self.extensive = extensive if extensive is not None else []


class Bond:
    def __init__(self, distance: float, bond_type: int, extensive: list = None):
        self.distance = distance
        self.bond_type = bond_type
        self.extensive = extensive if extensive is not None else []


class Molecule:
    def __init__(self, name: str, extensive: list):
        self.name = name
        self.atoms = []
        self.bonds = []
        self.extensive = extensive
        self.atom_ex_len = 0
        self.bond_ex_len = 0

    def add_atom(self, lics: int = 0, token: str = '', pos: list = None, extensive: list = None, edge_ex_len: int = 0):
        self.atoms.append(Atom(lics, token, pos, extensive))
        if extensive:
            self.atom_ex_len = len(extensive)
        # i = len(self.atoms) - 1
        # self.bonds.append((i, i, Bond(0., 0, [0] * edge_ex_len)))

    def add_bond(self, a1: int, a2: int, bond_type: int, distance: float = None, extensive: list = None):
        if not distance:
            distance = float(np.linalg.norm(np.array(self.atoms[a1].pos) - np.array(self.atoms[a2].pos)))
        bond = Bond(distance, bond_type, extensive)
        self.bonds.append((a1, a2, bond))
        self.bonds.append((a2, a1, bond))
        if extensive:
            self.bond_ex_len = len(extensive)

    def show(self):
        for a in self.atoms:
            print(a.lics)
        for u, v, b in self.bonds:
            print(u, v, b.bond_type)


class MoleculeSet:
    def __init__(self):
        self.molecules = []

        self.atom_set = set()
        self.bond_set = set()
        self.molecule_ex_len = -1
        self.atom_ex_len = -1
        self.bond_ex_len = -1

    def __iter__(self):
        return self.molecules

    def add(self, molecule: Molecule):
        self.molecules.append(molecule)

        for a in molecule.atoms:
            self.atom_set.add(a.lics)
        for b in molecule.bonds:
            self.bond_set.add(b[2].bond_type)

        if self.molecule_ex_len == -1:
            self.molecule_ex_len = len(molecule.extensive)
        else:
            assert self.molecule_ex_len == len(molecule.extensive)

        if self.atom_ex_len == -1:
            self.atom_ex_len = molecule.atom_ex_len
        else:
            assert self.atom_ex_len == molecule.atom_ex_len

        if self.bond_ex_len == -1:
            self.bond_ex_len = molecule.bond_ex_len
        else:
            assert self.bond_ex_len == molecule.bond_ex_len

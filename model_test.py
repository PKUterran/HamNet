# from data.gdb9_reader import load_gdb9
# from train.gdb9_trainer import train_gdb9
from train.qm9_trainer import train_qm9


# molecules = load_gdb9(50000)
# print(molecules.molecules.__len__())
# print(molecules.atom_set)
# print(molecules.bond_set)
# print(molecules.molecules[-1].show())

train_qm9(use_cuda=True, limit=10000)

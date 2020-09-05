import numpy as np
import pandas as pd
import os
import rdkit.Chem as rdchem
import pickle

from .encode import encode_smiles
from .config import *
from .gdb9_reader import load_mol_atom_pos as load_qm9_mol_atom_pos

QM9_CSV_FILE = 'data/QM9/qm9.csv'
QM9_RESOURCE_FILE = 'data/QM9/qm9.pickle'
HIV_CSV_FILE = 'data/HIV/HIV.csv'
HIV_RESOURCE_FILE = 'data/HIV/hiv.pickle'
LIPOP_CSV_FILE = 'data/Lipop/Lipophilicity.csv'
LIPOP_RESOURCE_FILE = 'data/Lipop/Lipop.pickle'
FREESOLV_CSV_FILE = 'data/FreeSolv/SAMPL.csv'
FREESOLV_RESOURCE_FILE = 'data/FreeSolv/FreeSolv.pickle'
BBBP_CSV_FILE = 'data/BBBP/BBBP.csv'
# BBBP_RESOURCE_FILE = 'data/BBBP/BBBP.pickle'
TOX21_CSV_FILE = 'data/TOX21/tox21.csv'
TOX21_RESOURCE_FILE = 'data/TOX21/tox21.pickle'


def encode_onehot(p: np.ndarray):
    labels = list(set(p))
    ret = np.zeros(p.shape[0])
    for i in range(len(p)):
        ret[i] = labels.index(p[i])
    # print(ret.sum())
    return ret


def load_qm9(max_num: int = -1, force_save=False, use_pos=False) -> (list, list, np.ndarray):
    smile_properties_list = np.array(pd.read_csv(QM9_CSV_FILE))
    properties = smile_properties_list[:, 5: 17]
    smiles = smile_properties_list[:, 1]
    if max_num != -1 and max_num < smile_properties_list.shape[0]:
        smiles = smiles[: max_num]
        properties = properties[: max_num]

    if force_save or not os.path.exists(QM9_RESOURCE_FILE):
        info_list = encode_smiles(smiles, max_position=QM9_PC, central_atoms=QM9_CA, max_dis=QM9_MD,
                                  mol_atom_pos=load_qm9_mol_atom_pos(max_num) if use_pos else None)
        pickle.dump(info_list, open(QM9_RESOURCE_FILE, 'wb'))
    else:
        info_list = pickle.load(open(QM9_RESOURCE_FILE, 'rb'))

    if max_num != -1 and max_num < smile_properties_list.shape[0]:
        info_list = info_list[: max_num]
    return smiles, info_list, properties


def load_hiv(max_num: int = -1, force_save=False) -> (list, list, np.ndarray):
    smile_properties_list = np.array(pd.read_csv(HIV_CSV_FILE))
    properties = smile_properties_list[:, 2]
    smiles = smile_properties_list[:, 0]
    if force_save or not os.path.exists(HIV_RESOURCE_FILE):
        info_list = encode_smiles(smiles)
        pickle.dump(info_list, open(HIV_RESOURCE_FILE, 'wb'))
    else:
        info_list = pickle.load(open(HIV_RESOURCE_FILE, 'rb'))

    if max_num != -1 and max_num < info_list.shape[0]:
        info_list = info_list[: max_num]
        properties = properties[: max_num]
    return smiles, info_list, encode_onehot(properties).astype(np.int)


def load_bbbp(max_num: int = -1, force_save=False) -> (list, list, np.ndarray):
    smile_properties_list = np.array(pd.read_csv(BBBP_CSV_FILE))
    properties = smile_properties_list[:, 0]
    smiles = smile_properties_list[:, 1]
    info_list, mask = encode_smiles(smiles, return_mask=True,
                                    max_position=BBBP_PC, central_atoms=BBBP_CA, max_dis=BBBP_MD)
    smiles = smiles[mask]
    properties = properties[mask]

    if max_num != -1 and max_num < smile_properties_list.shape[0]:
        info_list = info_list[: max_num]
        properties = properties[: max_num]
    # print(smiles)
    # print(properties)
    return smiles, info_list, encode_onehot(properties).astype(np.int)


def load_tox21(max_num: int = -1, force_save=False) -> (list, list, np.ndarray):
    smile_properties_list = np.array(pd.read_csv(TOX21_CSV_FILE))
    properties = smile_properties_list[:, : 12]
    smiles = smile_properties_list[:, 13]
    if force_save or not os.path.exists(TOX21_RESOURCE_FILE):
        info_list = encode_smiles(smiles, max_position=TOX21_PC, central_atoms=TOX21_CA, max_dis=TOX21_MD)
        pickle.dump(info_list, open(TOX21_RESOURCE_FILE, 'wb'))
    else:
        info_list = pickle.load(open(TOX21_RESOURCE_FILE, 'rb'))

    if max_num != -1 and max_num < smile_properties_list.shape[0]:
        info_list = info_list[: max_num]
        properties = properties[: max_num]
    return smiles, info_list, properties.astype(np.float)


def load_lipop(max_num: int = -1, force_save=False) -> (list, list, np.ndarray):
    smile_properties_list = np.array(pd.read_csv(LIPOP_CSV_FILE))
    properties = smile_properties_list[:, 1: 2]
    smiles = smile_properties_list[:, 2]
    if force_save or not os.path.exists(LIPOP_RESOURCE_FILE):
        info_list = encode_smiles(smiles, max_position=LIPOP_PC, central_atoms=LIPOP_CA, max_dis=LIPOP_MD)
        pickle.dump(info_list, open(LIPOP_RESOURCE_FILE, 'wb'))
    else:
        info_list = pickle.load(open(LIPOP_RESOURCE_FILE, 'rb'))

    if max_num != -1 and max_num < smile_properties_list.shape[0]:
        info_list = info_list[: max_num]
        properties = properties[: max_num]
    return smiles, info_list, properties


def load_freesolv(max_num: int = -1, force_save=False) -> (list, list, np.ndarray):
    smile_properties_list = np.array(pd.read_csv(FREESOLV_CSV_FILE))
    properties = smile_properties_list[:, 2: 3]
    smiles = smile_properties_list[:, 1]
    if force_save or not os.path.exists(FREESOLV_RESOURCE_FILE):
        info_list = encode_smiles(smiles)
        pickle.dump(info_list, open(FREESOLV_RESOURCE_FILE, 'wb'))
    else:
        info_list = pickle.load(open(FREESOLV_RESOURCE_FILE, 'rb'))

    if max_num != -1 and max_num < smile_properties_list.shape[0]:
        info_list = info_list[: max_num]
        properties = properties[: max_num]
    return smiles, info_list, properties

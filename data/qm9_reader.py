import numpy as np
import pandas as pd
import os
import rdkit.Chem as rdchem
import pickle

from .encode import encode_smiles

CSV_FILE = 'data/QM9/qm9.csv'
RESOURCE_FILE = 'data/QM9/qm9.pickle'


def load_qm9(max_num: int = -1, force_save=False) -> (list, list, np.ndarray):
    smile_properties_list = np.array(pd.read_csv(CSV_FILE))
    properties = smile_properties_list[:, 5: 17]
    smiles = smile_properties_list[:, 1]
    if force_save or not os.path.exists(RESOURCE_FILE):
        info_list = encode_smiles(smiles)
        pickle.dump(info_list, open(RESOURCE_FILE, 'wb'))
    else:
        info_list = pickle.load(open(RESOURCE_FILE, 'rb'))

    if max_num != -1 and max_num < smile_properties_list.shape[0]:
        info_list = info_list[: max_num]
        properties = properties[: max_num]
    return smiles, info_list, properties

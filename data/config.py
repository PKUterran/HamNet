LICS_MAP = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5,
    'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
}

C_FG_MODES = {
    '>C=C<': 'C:(2C)',
    '-C---C-': 'C:(3C)',
    '-C---N': 'C:(3N)',
    '-C=N-': 'C:(2N)',
    '-COOH': 'C:(2O)(1O:(1H))',
    '-COO-': 'C:(2O)(1O)',
    '-CHO': 'C:(2O)(1H)',
    '>CO': 'C:(2O)',
    '-C-O-': 'C:(1O)',
}

S_FG_MODES = {
    '-C-S-C-': 'S:(1C)(1C)',
    '-SO3H': 'S:(2O)(2O)(1O:(1H))',
}

P_FG_MODES = {
    '-PH2': 'P:(1H)(1H)',
}

N_FG_MODES = {
    '-N=N-': 'N:(2N)',
    '-NO2': 'N:(2O)(2O)',
    '-NH2': 'N:(1H)(1H)',
    '-NH': 'N:(1H)',
    '-NC': 'N:(4C)'
}

O_FG_MODES = {
    '-OH': 'O:(1H)',
    '-C-O-C-': 'O:(1C)(1C)',
}

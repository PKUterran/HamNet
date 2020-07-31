MAX_HYDRO = 4
MAX_BOND = 5

FITTER_CONFIG = {
    'TRAIN_PER': 0.8,
    'VALIDATE_PER': 0.1,
    'TEST_PER': 0.1,

    'LR': 1e-4,
    'DECAY': 1e-5,
    'DROPOUT': 0.5,
    'ITERATION': 50,
    'BATCH': 32,

    'HGN_LAYERS': 20,
    'TAU': 0.05,
    'PQ_DIM': 32,
    'GAMMA_S': 0.000,
    'GAMMA_C': 0.000,
    'DISSIPATE': True,
    'DISTURB': False,
}

FITTER_CONFIG_QM9 = FITTER_CONFIG.copy()
FITTER_CONFIG_QM9.update({
    'MAX_DICT': 4096,
})

MODEL_CONFIG = {
    'TRAIN_PER': 0.8,
    'VALIDATE_PER': 0.1,
    'TEST_PER': 0.1,

    'LR': 1e-3,
    'DECAY': 1e-5,
    'DROPOUT': 0.5,
    'ITERATION': 100,
    'EVAL': 10,
    'BATCH': 32,

    'POS_DIM': FITTER_CONFIG['PQ_DIM'] * 2,
    'HE_DIM': 128,
    'C_DIMS': [128, 128, 128, 128],
    'F_DIM': 128,
    'M_RADIUS': 2,
    'MLP_DIMS': [256],
    'MAX_DICT': 1024,
}

MODEL_CONFIG_QM9 = MODEL_CONFIG.copy()
MODEL_CONFIG_QM9.update({
    'LR': 10 ** -4.9,
    'DECAY': 10 ** -5.9,
    'DROPOUT': 0.0,
    'ITERATION': 300,
    'EVAL': 30,
    'HE_DIM': 280,
    'C_DIMS': [280, 280],
    'F_DIM': 280,
    'MLP_DIMS': [],
    'MAX_DICT': 4096,
})

MODEL_CONFIG_LIPOP = MODEL_CONFIG.copy()
MODEL_CONFIG_LIPOP.update({
    'LR': 2e-4,
    'DECAY': 1e-4,
    'DROPOUT': 0.5,
    'ITERATION': 200,
    'BATCH': 64,
    'EVAL': 20,
    'HE_DIM': 256,
    'C_DIMS': [256, 256, 256],
    'F_DIM': 256,
    'M_RADIUS': 4,
})

MODEL_CONFIG_TOX21 = MODEL_CONFIG.copy()
MODEL_CONFIG_TOX21.update({
    'LR': 2e-4,
    'DECAY': 5e-5,
    'BATCH': 64,
})


class BaseConfig:
    # UNIVERSAL #
    TRAIN_PER = 0.8
    VALIDATE_PER = 0.1
    TEST_PER = 0.1

    # AMPNN #
    HE_DIM = 32
    C_DIMS = [256, 256, 256]
    GLOBAL_MASK = [1, 1, 1]
    AMPNN_LAYERS = 3

    # HamGN #
    DISCRETE = True
    USE_HAM = True
    LSTM_ENCODE = True


class QM9Config(BaseConfig):
    # UNIVERSAL #
    LR = 1e-4
    DECAY = 2e-6
    DROPOUT = 0.0
    ITERATION = 100
    EVAL = 10
    BATCH = 32
    F_DIM = 128

    # HamGN #
    P_DIM = 64
    Q_DIM = 64
    HamGN_LAYERS = 20
    GAMMA_S = 0.0001
    GAMMA_C = 0.001
    GAMMA_A = 0.01
    GAMMA = 1
    TAU = 0.05

    # CACHE #
    MAX_CACHE_UNIT = 2 ** 21 + 2 ** 20
    MAX_DICT = int(MAX_CACHE_UNIT / BATCH ** 2)


class LipopConfig(BaseConfig):
    # UNIVERSAL #
    LR = 2e-4
    DECAY = 5e-5
    DROPOUT = 0.5
    ITERATION = 200
    EVAL = 10
    BATCH = 32
    F_DIM = 32
    H_DIMS = [64]

    # HamGN #
    P_DIM = 16
    Q_DIM = 16
    HamGN_LAYERS = 20
    GAMMA_S = 0.0001
    GAMMA_C = 0.01
    GAMMA_A = 0.0001
    GAMMA_DECAY = 0.1
    GAMMA = 1
    TAU = 0.05

    # CACHE #
    MAX_CACHE_UNIT = 2 ** 20
    MAX_DICT = int(MAX_CACHE_UNIT / BATCH ** 2)


class TOX21Config(BaseConfig):
    # UNIVERSAL #
    LR = 2e-4
    DECAY = 5e-5
    DROPOUT = 0.5
    ITERATION = 200
    EVAL = 10
    BATCH = 64
    F_DIM = 32
    H_DIMS = [64]

    # HamGN #
    P_DIM = 16
    Q_DIM = 16
    HamGN_LAYERS = 20
    GAMMA_S = 0.0001
    GAMMA_C = 0.01
    GAMMA_A = 0.0001
    GAMMA_DECAY = 0.1
    GAMMA = 1
    TAU = 0.05

    # CACHE #
    MAX_CACHE_UNIT = 2 ** 20
    MAX_DICT = int(MAX_CACHE_UNIT / BATCH ** 2)


class BBBPConfig(BaseConfig):
    # UNIVERSAL #
    LR = 1e-3
    DECAY = 2e-4
    DROPOUT = 0.5
    ITERATION = 200
    EVAL = 10
    BATCH = 100
    F_DIM = 16
    H_DIMS = [32]

    # HamGN #
    P_DIM = 8
    Q_DIM = 8
    HamGN_LAYERS = 20
    GAMMA_S = 0.000
    GAMMA_C = 0.0000
    GAMMA_A = 0.00
    GAMMA = 1
    TAU = 0.05

    # CACHE #
    MAX_CACHE_UNIT = 2 ** 20
    MAX_DICT = int(MAX_CACHE_UNIT / BATCH ** 2)


class HIVConfig(BaseConfig):
    # UNIVERSAL #
    LR = 1e-3
    DECAY = 5e-4
    DROPOUT = 0.5
    ITERATION = 100
    EVAL = 10
    BATCH = 64
    F_DIM = 16
    H_DIMS = [32]

    # HamGN #
    P_DIM = 8
    Q_DIM = 8
    HamGN_LAYERS = 20
    GAMMA_S = 0.0000
    GAMMA_C = 0.0000
    GAMMA_A = 0.00
    GAMMA = 1
    TAU = 0.05

    # CACHE #
    MAX_CACHE_UNIT = 2 ** 20
    MAX_DICT = int(MAX_CACHE_UNIT / BATCH ** 2)


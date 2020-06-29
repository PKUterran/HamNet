import numpy as np
import torch
import torch.optim as optim
import gc
from tqdm import tqdm

from .config import FITTER_CONFIG_QM9 as DEFAULT_CONFIG
from .HeteroGraph import HeteroGraph
from data.reader import load_qm9
from data.gdb9_reader import load_mol_atom_pos
from utils.sample import sample
from utils.MatrixCache import MatrixCache
from net.models import PositionEncoder


def set_seed(seed: int, use_cuda: bool):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def fit_qm9(seed: int = 19700101, limit: int = -1, use_cuda: bool = True, use_tqdm=True, force_save=False,
            special_config: dict = None, model_save_path: str = 'net/pe.pt'):
    cfg = DEFAULT_CONFIG.copy()
    if special_config:
        cfg.update(special_config)
    for k, v in cfg.items():
        print(k, ':', v)
    set_seed(seed, use_cuda)
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    smiles, info_list, _ = load_qm9(limit, force_save=force_save)
    mol_atom_pos = load_mol_atom_pos(limit)
    molecules = [HeteroGraph(info['nf'], info['ef'], info['us'], info['vs'], info['em']) for info in info_list]
    n_dim = molecules[0].n_dim
    e_dim = molecules[0].e_dim
    node_num = len(molecules)

    train_mask, validate_mask, test_mask = sample(list(range(node_num)),
                                                  cfg['TRAIN_PER'],
                                                  cfg['VALIDATE_PER'],
                                                  cfg['TEST_PER'])
    n_seg = int(len(train_mask) / (cfg['BATCH'] + 1))
    train_mask_list = [train_mask[i::n_seg] for i in range(n_seg)]
    n_seg = int(len(validate_mask) / (cfg['BATCH'] + 1))
    validate_mask_list = [validate_mask[i::n_seg] for i in range(n_seg)]
    n_seg = int(len(test_mask) / (cfg['BATCH'] + 1))
    test_mask_list = [test_mask[i::n_seg] for i in range(n_seg)]
    print(train_mask[0], validate_mask[0], test_mask[0])
    print(len(train_mask_list), len(validate_mask_list), len(test_mask_list))

    model = PositionEncoder(n_dim=n_dim,
                            e_dim=e_dim,
                            config=cfg,
                            use_cuda=use_cuda)

    if use_cuda:
        model.cuda()
    params = list(model.parameters())
    for param in params:
        print(param.shape)
    optimizer = optim.Adam(params, lr=cfg['LR'], weight_decay=cfg['DECAY'])
    current_lr = cfg['LR']
    matrix_cache = MatrixCache(cfg['MAX_DICT'])
    best_val = -1e8

    def forward(mask: list, name=None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        nfs = torch.cat([molecules[i].node_features for i in mask])
        efs = torch.cat([molecules[i].edge_features for i in mask])
        atom_pos = torch.cat([torch.from_numpy(mol_atom_pos[i]).type(torch.float32) for i in mask])
        if use_cuda:
            nfs = nfs.cuda()
            efs = efs.cuda()
            atom_pos = atom_pos.cuda()

        us, vs, mm_tuple = matrix_cache.fetch(molecules, mask, nfs, name, use_cuda)

        d_loss, s_loss, c_loss = model.fit(nfs, efs, us, vs, mm_tuple, atom_pos, print_mode=name == 'test59')
        return d_loss, s_loss, c_loss

    def train(mask_list: list, name=None):
        model.train()
        d_losses = []
        s_losses = []
        c_losses = []

        t = enumerate(mask_list)
        if use_tqdm:
            t = tqdm(t, total=len(mask_list))
        for i, m in t:
            if name:
                name_ = name + str(i)
            else:
                name_ = None
            optimizer.zero_grad()
            d_loss, s_loss, c_loss = forward(m, name=name_)
            d_losses.append(d_loss.cpu().item() if 'cpu' in dir(d_loss) else d_loss)
            s_losses.append(s_loss.cpu().item() if 'cpu' in dir(s_loss) else s_loss)
            c_losses.append(c_loss.cpu().item() if 'cpu' in dir(c_loss) else c_loss)
            loss = d_loss + cfg['GAMMA_S'] * s_loss + cfg['GAMMA_C'] * c_loss
            loss.backward()
            optimizer.step()
            nonlocal current_lr
            current_lr *= 1 - cfg['DECAY']

        print('\t\tDistance loss: {:.4f}'.format(np.average(d_losses)))
        print('\t\tStationary loss: {:.4f}'.format(np.average(s_losses)))
        print('\t\tCentrality loss: {:.4f}'.format(np.average(c_losses)))

    def evaluate(mask_list: list, name=None):
        model.eval()
        losses = []
        d_losses = []

        t = enumerate(mask_list)
        if use_tqdm:
            t = tqdm(t, total=len(mask_list))
        for i, m in t:
            if name:
                name_ = name + str(i)
            else:
                name_ = None
            d_loss, s_loss, c_loss = forward(m, name=name_)
            loss = d_loss + cfg['GAMMA_S'] * s_loss + cfg['GAMMA_C'] * c_loss
            losses.append(loss.cpu().item())
            d_losses.append(d_loss.cpu().item())

        if name == 'evaluate':
            val = -np.average(losses)
            nonlocal best_val
            if val > best_val:
                print('\t\tSaving position encoder...')
                torch.save(model, model_save_path)
                best_val = val
                print('\t\tSaving finished!')
        print('\t\tLoss: {:.5f}'.format(np.average(losses)))
        print('\t\tDistance Loss: {:.5f}'.format(np.average(d_losses)))

    for epoch in range(cfg['ITERATION']):
        print('In iteration {}:'.format(epoch + 1))
        print('\tLearning rate: {:.8e}'.format(current_lr))
        print('\tTraining: ')
        train(train_mask_list, name='train')
        print('\tEvaluating training: ')
        evaluate(train_mask_list, name='train')
        print('\tEvaluating validation: ')
        evaluate(validate_mask_list, name='evaluate')
        print('\tEvaluating test: ')
        evaluate(test_mask_list, name='test')
        gc.collect()

import numpy as np
import time
import torch
import torch.optim as optim
import torch.autograd as autograd
import gc
import json
from tqdm import tqdm

from .config import FITTER_CONFIG_QM9 as DEFAULT_CONFIG
from .HeteroGraph import HeteroGraph
from data.reader import load_qm9
from data.gdb9_reader import load_mol_atom_pos
from utils.sample import sample
from utils.cache import MatrixCache
from utils.rotate import rotate_to
from utils.kabsch import kabsch
from visualize.molecule import plt_molecule_3d
from net.models import PositionEncoder

GRAPH_PATH = 'graphs/pos/'
LOG_PATH = 'logs/pos/'


def set_seed(seed: int, use_cuda: bool):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def fit_qm9(seed: int = 19700101, limit: int = -1, use_cuda: bool = True, use_tqdm=True, force_save=False,
            special_config: dict = None, model_save_path: str = 'net/pe.pt', tag='std', mpnn_pos_encode=False,
            use_rdkit=False):
    t0 = time.time()
    cfg = DEFAULT_CONFIG.copy()
    if special_config:
        cfg.update(special_config)
    for k, v in cfg.items():
        print(k, ':', v)
    set_seed(seed, use_cuda)
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    smiles, info_list, _ = load_qm9(limit, force_save=force_save)
    mol_atom_pos = load_mol_atom_pos(limit)

    with open('data/gdb9/incons.json') as fp:
        incons = json.load(fp)
        left = list(set(range(len(smiles))) - set(incons))
        print('{} / {}'.format(len(left), len(smiles)))
        smiles = [smiles[i] for i in left]
        info_list = [info_list[i] for i in left]
        mol_atom_pos = [mol_atom_pos[i] for i in left]

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
                            use_cuda=use_cuda,
                            use_mpnn=mpnn_pos_encode,
                            use_rdkit=use_rdkit)

    if use_cuda:
        model.cuda()
    for name, param in model.named_parameters():
        print(name, ":", param.shape)
    optimizer = optim.Adam(model.parameters(), lr=cfg['LR'], weight_decay=cfg['DECAY'])
    current_lr = cfg['LR']
    matrix_cache = MatrixCache(cfg['MAX_DICT'])
    best_val = -1e8
    logs = []
    graph_logs = []

    def visualize(smiles_list, pos: torch.Tensor, fit_pos: torch.Tensor, mol_node_matrix: torch.Tensor, vis=range(5)):
        if use_cuda:
            pos = pos.cpu()
            fit_pos = fit_pos.cpu()
            mol_node_matrix = mol_node_matrix.cpu()
        pos = pos.detach()
        fit_pos = fit_pos.detach()
        pos_list = []
        for i in vis:
            node_mask = mol_node_matrix[i] > 0
            pos_i = pos[node_mask == 1, :]
            fit_pos_i = fit_pos[node_mask == 1, :]
            new_pos_i, new_fit_pos_i = kabsch(pos_i, fit_pos_i,
                                              torch.full([1, pos_i.shape[0]], 1, dtype=torch.float32),
                                              use_cuda=False)
            pos_list.append({'smiles': smiles_list[i], 'src': new_pos_i.tolist(), 'tgt': new_fit_pos_i.tolist()})
            # plt_molecule_3d(new_pos_i.numpy(), smiles_list[i],
            #                 title='fit_qm9_{}_{}_{}'.format(tag, epoch, i), d=GRAPH_PATH)
            # plt_molecule_3d(new_fit_pos_i.numpy(), smiles_list[i],
            #                 title='fit_qm9_origin_{}'.format(i), d=GRAPH_PATH)
        graph_logs[-1].update({'pos': pos_list})

    def forward(mask: list, name=None):
        nfs = torch.cat([molecules[i].node_features for i in mask])
        efs = torch.cat([molecules[i].edge_features for i in mask])
        atom_pos = torch.cat([torch.from_numpy(mol_atom_pos[i]).type(torch.float32) for i in mask])
        if use_cuda:
            nfs = nfs.cuda()
            efs = efs.cuda()
            atom_pos = atom_pos.cuda()

        us, vs, mm_tuple = matrix_cache.fetch(molecules, mask, nfs, name, use_cuda)
        mask_smiles = [smiles[i] for i in mask]

        adj3_loss, dis_loss, rmsd_loss, s_loss, c_loss, pos = model.fit(nfs, efs, mask_smiles, us, vs, mm_tuple,
                                                                          atom_pos, print_mode=name == 'test0')
        if name == 'test0':
            visualize([smiles[i] for i in mask], pos, atom_pos, mm_tuple[0])
        return adj3_loss, dis_loss, rmsd_loss, s_loss, c_loss

    def train(mask_list: list, name=None):
        model.train()
        a_losses = []
        d_losses = []
        r_losses = []
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
            a_loss, d_loss, r_loss, s_loss, c_loss = forward(m, name=name_)
            a_losses.append(a_loss.cpu().item() if 'cpu' in dir(a_loss) else a_loss)
            d_losses.append(d_loss.cpu().item() if 'cpu' in dir(d_loss) else d_loss)
            r_losses.append(r_loss.cpu().item() if 'cpu' in dir(r_loss) else r_loss)
            s_losses.append(s_loss.cpu().item() if 'cpu' in dir(s_loss) else s_loss)
            c_losses.append(c_loss.cpu().item() if 'cpu' in dir(c_loss) else c_loss)
            # loss = a_loss + cfg['GAMMA_S'] * s_loss + cfg['GAMMA_C'] * c_loss
            # loss = d_loss + cfg['GAMMA_S'] * s_loss + cfg['GAMMA_C'] * c_loss
            loss = r_loss + cfg['GAMMA_S'] * s_loss + cfg['GAMMA_C'] * c_loss + cfg['GAMMA_A'] * a_loss
            loss.backward()
            optimizer.step()
            nonlocal current_lr
            current_lr *= 1 - cfg['DECAY']

        print('\t\tADJ3 loss: {:.4f}'.format(np.average(a_losses)))
        print('\t\tDistance loss: {:.4f}'.format(np.average(d_losses)))
        print('\t\tRMSD metric: {:.4f}'.format(np.average(r_losses)))
        print('\t\tStationary loss: {:.4f}'.format(np.average(s_losses)))
        print('\t\tCentrality loss: {:.4f}'.format(np.average(c_losses)))
        logs[-1].update({'on_train_loss': np.average(a_losses)})

    def evaluate(mask_list: list, name=None):
        model.eval()
        losses = []
        a_losses = []
        d_losses = []
        r_losses = []

        t = enumerate(mask_list)
        if use_tqdm:
            t = tqdm(t, total=len(mask_list))
        for i, m in t:
            if name:
                name_ = name + str(i)
            else:
                name_ = None
            a_loss, d_loss, r_loss, s_loss, c_loss = forward(m, name=name_)
            # loss = a_loss + cfg['GAMMA_S'] * s_loss + cfg['GAMMA_C'] * c_loss
            # loss = d_loss + cfg['GAMMA_S'] * s_loss + cfg['GAMMA_C'] * c_loss
            loss = r_loss + cfg['GAMMA_S'] * s_loss + cfg['GAMMA_C'] * c_loss + cfg['GAMMA_A'] * a_loss
            losses.append(loss.cpu().item() if 'cpu' in dir(loss) else loss)
            a_losses.append(a_loss.cpu().item() if 'cpu' in dir(a_loss) else a_loss)
            d_losses.append(d_loss.cpu().item() if 'cpu' in dir(d_loss) else d_loss)
            r_losses.append(r_loss.cpu().item() if 'cpu' in dir(r_loss) else r_loss)

        if name == 'evaluate':
            val = -np.average(losses)
            nonlocal best_val
            if val > best_val:
                print('\t\tSaving position encoder...')
                torch.save(model, model_save_path)
                best_val = val
                print('\t\tSaving finished!')
        print('\t\tLoss: {:.5f}'.format(np.average(losses)))
        print('\t\tADJ3 loss: {:.5f}'.format(np.average(a_losses)))
        print('\t\tDistance loss: {:.5f}'.format(np.average(d_losses)))
        print('\t\tRMSD metric: {:.5f}'.format(np.average(r_losses)))
        logs[-1].update({'{}_loss'.format(name): np.average(losses)})
        logs[-1].update({'{}_adj3_metric'.format(name): np.average(a_losses)})
        logs[-1].update({'{}_distance_metric'.format(name): np.average(d_losses)})
        logs[-1].update({'{}_rmsd_metric'.format(name): np.average(r_losses)})

    for epoch in range(cfg['ITERATION']):
        logs.append({'epoch': epoch + 1})
        graph_logs.append({'epoch': epoch + 1})
        print('In iteration {}:'.format(epoch + 1))
        print('\tLearning rate: {:.8e}'.format(current_lr))
        if not use_rdkit:
            print('\tTraining: ')
            train(train_mask_list, name='train')
            print('\tEvaluating training: ')
            evaluate(train_mask_list, name='train')
        print('\tEvaluating validation: ')
        evaluate(validate_mask_list, name='evaluate')
        print('\tEvaluating test: ')
        evaluate(test_mask_list, name='test')
        gc.collect()
        d = {'metric': 'Distance Loss', 'time': time.time() - t0, 'logs': logs}
        with open('{}{}.json'.format(LOG_PATH, tag), 'w+', encoding='utf-8') as fp:
            json.dump(d, fp)
        gd = graph_logs
        with open('{}{}.json'.format(GRAPH_PATH, tag), 'w+', encoding='utf-8') as fp:
            json.dump(gd, fp)

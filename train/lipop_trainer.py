import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import gc
from torch.nn import MSELoss
from itertools import chain
from statsmodels import robust
from tqdm import tqdm

from .config import *
from .HeteroGraph import HeteroGraph
from data.reader import load_lipop
from utils.sample import sample
from net.models import DynamicGraphEncoder, MLP, AMPNN
from visualize.regress import plt_multiple_scatter

GRAPH_PATH = 'graphs/Lipop/'


def set_seed(seed: int, use_cuda: bool):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def train_lipop(seed: int = 19700101, limit: int = -1, use_cuda: bool = True, use_tqdm=True,
              prop=list(range(1)), use_model='HamGN'):
    set_seed(seed, use_cuda)
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    smiles, info_list, properties = load_lipop(limit)
    properties = properties[:, prop]
    molecules = [HeteroGraph(info['nf'], info['ef'], info['us'], info['vs'], info['em']) for info in info_list]
    n_dim = molecules[0].n_dim
    e_dim = molecules[0].e_dim
    node_num = len(molecules)

    train_mask, validate_mask, test_mask = sample(list(range(node_num)), TRAIN_PER, VALIDATE_PER, TEST_PER)
    n_seg = int(len(train_mask) / (BATCH + 1))
    train_mask_list = [train_mask[i::n_seg] for i in range(n_seg)]
    n_seg = int(len(validate_mask) / (BATCH + 1))
    validate_mask_list = [validate_mask[i::n_seg] for i in range(n_seg)]
    n_seg = int(len(test_mask) / (BATCH + 1))
    test_mask_list = [test_mask[i::n_seg] for i in range(n_seg)]
    print(train_mask, validate_mask, test_mask)
    print(len(train_mask_list), len(validate_mask_list), len(test_mask_list))

    t_properties = properties[train_mask, :]
    prop_mean = np.mean(t_properties, axis=0)
    print('mean:', prop_mean)
    prop_std = np.std(t_properties.tolist(), axis=0, ddof=1)
    print('std:', prop_std)
    prop_mad = robust.mad(t_properties.tolist(), axis=0)
    print('mad:', prop_mad)
    norm_properties = (properties - prop_mean) / prop_std

    if use_model == 'HamGN':
        model = DynamicGraphEncoder(v_dim=n_dim,
                                    e_dim=e_dim,
                                    p_dim=P_DIM,
                                    q_dim=Q_DIM,
                                    f_dim=F_DIM,
                                    layers=HamGN_LAYERS,
                                    hamilton=True,
                                    discrete=True,
                                    gamma=GAMMA,
                                    tau=TAU,
                                    dropout=DROPOUT,
                                    use_cuda=use_cuda)
    elif use_model == 'AMPNN':
        model = AMPNN(n_dim=n_dim,
                      e_dim=e_dim,
                      h_dim=F_DIM,
                      c_dims=C_DIMS,
                      he_dim=HE_DIM,
                      layers=AMPNN_LAYERS,
                      residual=False,
                      use_cuda=use_cuda,
                      dropout=DROPOUT)
    else:
        assert False, 'Undefined model: {}!'.format(use_model)
    regression = MLP(F_DIM, MLP_HIDDEN, len(prop), dropout=DROPOUT)
    if use_cuda:
        model.cuda()
        regression.cuda()
    params = list(chain(model.parameters(), regression.parameters()))
    for param in params:
        print(param.shape)
    optimizer = optim.Adam(params, lr=LR, weight_decay=DECAY)
    loss_fuc = MSELoss()
    # forward_time = 0.
    bp_time = 0.
    matrix_mask_dicts = {}
    s_losses = []
    c_losses = []
    a_losses = []
    u_losses = []

    def forward(mask: list, name=None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        nfs = torch.cat([molecules[i].node_features for i in mask])
        efs = torch.cat([molecules[i].edge_features for i in mask])
        if use_cuda:
            nfs = nfs.cuda()
            efs = efs.cuda()
        ms = []
        us = []
        vs = []
        em = []
        ptr = 0
        for i, m in enumerate(mask):
            nn = molecules[m].node_features.shape[0]
            ms.extend([i] * nn)
            for u in molecules[m].us:
                us.append(u + ptr)
            for v in molecules[m].vs:
                vs.append(v + ptr)
            em.extend(molecules[m].edge_mask)
            ptr += nn

        if name and name in matrix_mask_dicts.keys():
            mm_tuple = matrix_mask_dicts[name]
        else:
            n_node = nfs.shape[0]
            mol_node_matrix, mol_node_mask = \
                AMPNN.produce_node_edge_matrix(max(ms) + 1, ms, ms, [1] * len(ms))
            node_edge_matrix_global, node_edge_mask_global = \
                AMPNN.produce_node_edge_matrix(n_node, us, vs, [1] * len(us))
            if use_cuda:
                mol_node_matrix = mol_node_matrix.cuda()
                mol_node_mask = mol_node_mask.cuda()
                node_edge_matrix_global = node_edge_matrix_global.cuda()
                node_edge_mask_global = node_edge_mask_global.cuda()
            mm_tuple = (mol_node_matrix, mol_node_mask,
                        node_edge_matrix_global, node_edge_mask_global,
                        )
            if name and len(matrix_mask_dicts.keys()) < MAX_DICT:
                matrix_mask_dicts[name] = mm_tuple

        if use_model == 'HamGN':
            embeddings, s_loss, c_loss, a_loss = model(nfs, efs, us, vs, mm_tuple)
            # if np.random.randint(0, 1000) == 0:
            #     print(embeddings.cpu().detach().numpy())
            s_losses.append(s_loss.cpu().item())
            c_losses.append(c_loss.cpu().item())
            a_losses.append(a_loss.cpu().item())
            std_loss = GAMMA_S * s_loss + GAMMA_C * c_loss + GAMMA_A * a_loss
        elif use_model == 'AMPNN':
            embeddings, _ = model(nfs, efs, us, vs, mm_tuple, GLOBAL_MASK)
            std_loss = 0
        else:
            assert False
        logits = regression(embeddings)
        target = norm_properties[mask, :]
        target = torch.tensor(target.astype(np.float32), dtype=torch.float32)
        if use_cuda:
            target = target.cuda()
        return logits, target, std_loss

    def train(mask_list: list, name=None):
        model.train()
        regression.train()

        s_losses.clear()
        c_losses.clear()
        a_losses.clear()
        u_losses.clear()

        t = enumerate(mask_list)
        if use_tqdm:
            t = tqdm(t, total=len(mask_list))
        for i, m in t:
            if name:
                name_ = name + str(i)
            else:
                name_ = None
            optimizer.zero_grad()
            logits, target, std_loss = forward(m, name=name_)
            # print(logits.cpu())
            # print(target.cpu())
            u_loss = loss_fuc(logits, target)
            u_losses.append(u_loss.cpu().item())
            loss = u_loss + std_loss
            loss.backward()
            optimizer.step()

        if use_model == 'HamGN':
            print('\t\tStationary loss: {:.4f}'.format(np.average(s_losses)))
            print('\t\tCentrality loss: {:.4f}'.format(np.average(c_losses)))
            print('\t\tAffinity loss: {:.4f}'.format(np.average(a_losses)))
        print('\t\tSemi-supervised loss: {:.4f}'.format(np.average(u_losses)))

    def evaluate(mask_list: list, name=None, visualize=None):
        model.eval()
        regression.eval()
        losses = []
        masks = []
        logits_list = []
        target_list = []
        t = enumerate(mask_list)
        if use_tqdm:
            t = tqdm(t, total=len(mask_list))
        for i, m in t:
            if name:
                name_ = name + str(i)
            else:
                name_ = None
            logits, target, _ = forward(m, name=name_)
            loss = loss_fuc(logits, target) * prop_std[0]
            losses.append(loss.cpu().item())

            if visualize:
                masks.extend(m)
                logits_list.append(logits.cpu().detach().numpy())
                target_list.append(target.cpu().detach().numpy())

        print('\t\tMSE Loss: {:.3f}'.format(np.average(losses)))
        print('\t\tRMSE Loss: {:.3f}'.format(np.average([l ** 0.5 for l in losses])))
        if visualize:
            all_logits = np.vstack(logits_list)
            all_target = np.vstack(target_list)
            best_ids, best_ds, worst_ids, worst_ds = \
                plt_multiple_scatter(GRAPH_PATH + visualize, masks, all_logits, all_target)
            print('\t\tBest performance on:')
            for i, d in zip(best_ids, best_ds):
                print('\t\t\t{}: {}'.format(smiles[i], d))
            print('\t\tWorst performance on:')
            for i, d in zip(worst_ids, worst_ds):
                print('\t\t\t{}: {}'.format(smiles[i], d))

    for epoch in range(ITERATION):
        print('In iteration {}:'.format(epoch + 1))
        print('\tTraining: ')
        train(train_mask_list, name='train')
        print('\tEvaluating training: ')
        evaluate(train_mask_list, name='train',
                 visualize='{}_train_{}'.format(use_model, epoch + 1) if (epoch + 1) % EVAL == 0 else None)
        print('\tEvaluating validation: ')
        evaluate(validate_mask_list, name='evaluate',
                 visualize='{}_val_{}'.format(use_model, epoch + 1) if (epoch + 1) % EVAL == 0 else None)
        print('\tEvaluating test: ')
        evaluate(test_mask_list, visualize='{}_test'.format(use_model) if epoch + 1 == ITERATION else None)
        gc.collect()

    # print(model.total_forward_time)
    # print(model.layer_forward_time)

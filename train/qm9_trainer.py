import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import gc
from torch.nn import MSELoss, CrossEntropyLoss
from itertools import chain
from statsmodels import robust
from tqdm import tqdm

from .config import *
from .HeteroGraph import HeteroGraph
from data.qm9_reader import load_qm9
from utils.sample import sample
from net.models import DynamicGraphEncoder, MLP, AMPNN


def set_seed(seed: int, use_cuda: bool):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def train_qm9(seed: int = 19700101, limit: int = -1, use_cuda: bool = True,
              prop: list = list(range(12))):
    set_seed(seed, use_cuda)
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    smiles, info_list, properties = load_qm9(limit)
    properties = properties[:, prop]
    molecules = [HeteroGraph(info['nf'], info['ef'], info['us'], info['vs'], info['em']) for info in info_list]
    n_dim = molecules[0].n_dim
    e_dim = molecules[0].e_dim
    hidden_dims = [H_DIM] * len(C_DIMS)
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
    ratio = (prop_std / prop_mad) ** 2
    norm_properties = (properties - prop_mean) / prop_std

    model = DynamicGraphEncoder(v_dim=n_dim,
                                e_dim=e_dim,
                                p_dim=P_DIM,
                                q_dim=Q_DIM,
                                f_dim=F_DIM,
                                layers=LAYERS,
                                hamilton=True,
                                discrete=True,
                                gamma=GAMMA,
                                tau=TAU,
                                dropout=DROPOUT,
                                use_cuda=use_cuda)
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

    def forward(mask: list, show_attention_cnt=0, name=None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
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

        embeddings, s_loss, c_loss, a_loss = model(nfs, efs, us, vs, mm_tuple)
        # if np.random.randint(0, 1000) == 0:
        #     print(embeddings.cpu().detach().numpy())
        s_losses.append(s_loss.cpu().item())
        c_losses.append(c_loss.cpu().item())
        a_losses.append(a_loss.cpu().item())
        std_loss = GAMMA_S * s_loss + GAMMA_C * c_loss + GAMMA_A * a_loss
        logits = regression(embeddings)
        target = norm_properties[mask, :]
        target = torch.tensor(target.astype(np.float32), dtype=torch.float32)
        if use_cuda:
            target = target.cuda()
        return logits, target, std_loss

    def calc_normalized_loss(logits, target):
        losses = []
        for i in range(len(prop)):
            losses.append(loss_fuc(logits[:, i], target[:, i]) * ratio[i])
        return sum(losses)

    def train(mask_list: list, name=None):
        model.train()
        regression.train()

        s_losses.clear()
        c_losses.clear()
        a_losses.clear()
        u_losses.clear()

        with tqdm(enumerate(mask_list), total=len(mask_list)) as t:
            for i, m in t:
                if name:
                    name_ = name + str(i)
                else:
                    name_ = None
                optimizer.zero_grad()
                logits, target, std_loss = forward(m, name=name_)
                u_loss = calc_normalized_loss(logits, target)
                u_losses.append(u_loss.cpu().item())
                loss = u_loss + std_loss
                loss.backward()
                optimizer.step()

        print('Stationary loss: {:.4f}'.format(np.average(s_losses)))
        print('Centrality loss: {:.4f}'.format(np.average(c_losses)))
        print('Affinity loss: {:.4f}'.format(np.average(a_losses)))
        print('Semi-supervised loss: {:.4f}'.format(np.average(u_losses)))

    def evaluate(mask_list: list, name=None):
        model.eval()
        regression.eval()
        losses = []
        maes = []
        with tqdm(enumerate(mask_list), total=len(mask_list)) as t:
            for i, m in t:
                if name:
                    name_ = name + str(i)
                else:
                    name_ = None
                logits, target, _ = forward(m, name=name_)
                loss = calc_normalized_loss(logits, target)
                mae = torch.abs(logits - target).mean(dim=0)
                losses.append(loss.cpu().item())
                maes.append(mae.cpu().detach().numpy())
        print('\t\tLoss: {:.3f}'.format(np.average(losses)))
        print('\t\tMAE: {}.'.format(np.average(maes, axis=0) * prop_std))

    for epoch in range(ITERATION):
        print('In iteration {}:'.format(epoch + 1))
        print('\tTraining: ')
        train(train_mask_list, 'train')
        print('\tEvaluating training: ')
        evaluate(train_mask_list, 'train')
        print('\tEvaluating validation: ')
        evaluate(validate_mask_list, 'evaluate')
        gc.collect()

    print('Training finished.')
    print('\tEvaluating test: ')
    evaluate(test_mask_list)
    print(bp_time)
    print(model.total_forward_time)
    print(model.layer_forward_time)

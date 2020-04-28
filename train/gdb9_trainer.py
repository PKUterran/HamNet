import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import MSELoss, CrossEntropyLoss
from itertools import chain
from statsmodels import robust

from .config import *
from .HeteroGraph import HeteroGraph
from data.gdb9_reader import *
from utils.sample import sample
from net.models import AMPNN, MLP


def set_seed(seed: int, use_cuda: bool):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def encode_molecules(molecule_set: MoleculeSet) -> (list, int, int):
    max_hydro = MAX_HYDRO
    max_bond = MAX_BOND
    node_onehot = {n: i for i, n in enumerate(molecule_set.atom_set)}
    edge_onehot = {e: i for i, e in enumerate(molecule_set.bond_set)}
    hetero_graphs = []
    n_dim = 0
    e_dim = 0

    def encode_atom_type(lics: int):
        ret = [0] * len(molecule_set.atom_set)
        ret[node_onehot[lics]] = 1
        return ret

    def encode_atom_hydro(hydro: int):
        ret = [0] * (max_hydro + 1)
        ret[hydro] = 1
        return ret

    def encode_atom_bond(bond: int):
        ret = [0] * (max_bond + 1)
        ret[bond] = 1
        return ret

    def encode_edge_type(bond_type: int):
        ret = [0] * len(molecule_set.bond_set)
        ret[edge_onehot[bond_type]] = 1
        return ret

    for molecule in molecule_set.molecules:
        heavy_atoms = [i for i, a in enumerate(molecule.atoms) if a.lics != 1]
        hydros = [set() for _ in range(len(heavy_atoms))]
        bonds = [set() for _ in range(len(heavy_atoms))]
        id2id = {k: i for i, k in enumerate(heavy_atoms)}
        us = []
        vs = []
        edges = []
        edge_mask = []
        for u, v, b in molecule.bonds:
            if u not in heavy_atoms and v not in heavy_atoms:
                continue
            if u not in heavy_atoms:
                hydros[id2id[v]].add(u)
            elif v not in heavy_atoms:
                hydros[id2id[u]].add(v)
            else:
                us.append(id2id[u])
                vs.append(id2id[v])
                edges.append(b)
                if b.bond_type != 1 or molecule.atoms[u].lics not in [6, 7] or molecule.atoms[v].lics not in [6, 7]:
                    edge_mask.append(1)
                else:
                    edge_mask.append(0)
                if u != v:
                    bonds[id2id[u]].add(id2id[v])
                    bonds[id2id[v]].add(id2id[u])

        # print(hydros)
        # print(bonds)
        node_features = [
            encode_atom_type(molecule.atoms[k].lics) +
            encode_atom_hydro(len(hydros[i])) +
            encode_atom_bond(len(bonds[i])) +
            molecule.atoms[k].extensive
            for i, k in enumerate(heavy_atoms)
        ]

        edge_features = [
            [e.distance] +
            encode_edge_type(e.bond_type) +
            e.extensive
            for e in edges
        ]

        hg = HeteroGraph(node_features, edge_features, us, vs, edge_mask)
        hetero_graphs.append(hg)
        if n_dim:
            assert n_dim == hg.n_dim
        else:
            n_dim = hg.n_dim
        if e_dim:
            assert e_dim == hg.e_dim
        else:
            e_dim = hg.e_dim

    return hetero_graphs, n_dim, e_dim


def train_gdb9(seed: int = 19700101, limit: int = -1, residual: bool = True, use_cuda: bool = False, prop: list = [9]):
    set_seed(seed, use_cuda)
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    molecule_set, properties = load_gdb9(limit)
    properties = properties[:, prop]
    molecules, n_dim, e_dim = encode_molecules(molecule_set)
    hidden_dims = [H_DIM] * len(C_DIMS)

    node_num = len(molecules)
    train_mask, validate_mask, test_mask = sample(list(range(node_num)), TRAIN_PER, VALIDATE_PER, TEST_PER)
    print(train_mask, validate_mask, test_mask)
    t_properties = properties[train_mask, :]
    prop_mean = np.mean(t_properties, axis=0)
    print('mean:', prop_mean)
    prop_std = np.std(t_properties.tolist(), axis=0, ddof=1)
    print('std:', prop_std)
    prop_mad = robust.mad(t_properties.tolist(), axis=0)
    print('mad:', prop_mad)
    ratio = (prop_std / prop_mad) ** 2
    norm_properties = (properties - prop_mean) / prop_std

    model = AMPNN(n_dim, e_dim, H_DIM, C_DIMS, HE_DIM, HEAD_NUM, len(hidden_dims),
                  residual=residual, use_cuda=use_cuda, dropout=DROPOUT)
    if residual:
        r_dim = (n_dim + sum(hidden_dims))
    else:
        r_dim = hidden_dims[-1]
    regression = MLP(int(r_dim * HEAD_NUM), MLP_HIDDEN, len(prop), dropout=DROPOUT)
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

    def forward(mask: list, show_attention_cnt=0) -> (torch.Tensor, torch.Tensor, list):
        as_ = []
        embeddings = []
        target = norm_properties[mask, :]
        for i in mask:
            hg = molecules[i]
            if use_cuda:
                embedding, a = model(hg.node_features.cuda(), hg.edge_features.cuda(), hg.us, hg.vs,
                                     hg.edge_mask, GLOBAL_MASK)

            else:
                embedding, a = model(hg.node_features, hg.edge_features, hg.us, hg.vs,
                                     hg.edge_mask, GLOBAL_MASK)
            embeddings.append(embedding)
            as_.append(a)
            if show_attention_cnt:
                print('### For molecule {} ###'.format(i))
                molecule_set.molecules[i].show()
                if use_cuda:
                    a = a.cpu()
                print(a.detach().numpy())
                show_attention_cnt -= 1

        embeddings = torch.stack(embeddings)
        target = torch.tensor(target.astype(np.float32), dtype=torch.float32)
        if use_cuda:
            embeddings = embeddings.cuda()
            target = target.cuda()
        logits = regression(embeddings)
        return logits, target, as_

    def calc_normalized_loss(logits, target):
        losses = []
        for i in range(len(prop)):
            losses.append(loss_fuc(logits[:, i], target[:, i]) * ratio[i])
        return sum(losses)

    t_losses = []
    v_losses = []
    t_maes = []
    v_maes = []
    for epoch in range(ITERATION):
        optimizer.zero_grad()
        if len(train_mask) > TRN_BATCH:
            temp_train_mask = np.random.permutation(train_mask)[:TRN_BATCH]
        else:
            temp_train_mask = train_mask
        if len(validate_mask) > VAL_BATCH:
            temp_validate_mask = np.random.permutation(validate_mask)[:VAL_BATCH]
        else:
            temp_validate_mask = validate_mask

        # t1 = time.time()
        t_logits, t_target, tas = forward(temp_train_mask)
        v_logits, v_target, vas = forward(temp_validate_mask)
        # forward_time += time.time() - t1

        t_loss = calc_normalized_loss(t_logits, t_target)
        # tas_loss = 0.0 * t_loss.cpu().item() * sum([as_.sum(1).norm() for as_ in tas]) / len(tas)
        # total_loss = t_loss + tas_loss
        v_loss = calc_normalized_loss(v_logits, v_target)
        t_mae = torch.abs(t_logits - t_target).mean(dim=0)
        v_mae = torch.abs(v_logits - v_target).mean(dim=0)
        t_losses.append(t_loss.cpu().item())
        v_losses.append(v_loss.cpu().item())
        t_maes.append(t_mae.cpu().detach().numpy())
        v_maes.append(v_mae.cpu().detach().numpy())
        if (epoch + 1) % EVAL == 0:
            print('In iteration {}, training: {:.3f}; validation: {:.3f}'.
                  format(epoch, np.average(t_losses[-EVAL:]), np.average(v_losses[-EVAL:])))
            print('\tFor training:   {}.'.format(np.average(t_maes[-EVAL:], axis=0) * prop_std))
            print('\tFor validation: {}.'.format(np.average(v_maes[-EVAL:], axis=0) * prop_std))
            # print('\tBias: {}.'.format(regression.linear1.bias.cpu().detach().numpy()))
            # print(tas_loss.cpu().item())
        t1 = time.time()
        t_loss.backward()
        optimizer.step()
        bp_time += time.time() - t1

    if len(test_mask) > TST_BATCH:
        temp_test_mask = np.random.permutation(test_mask)[:TST_BATCH]
    else:
        temp_test_mask = test_mask
    e_logits, e_target, eas = forward(temp_test_mask, show_attention_cnt=10)
    print(e_logits.cpu().detach().numpy() * prop_std + prop_mean)
    print(e_target.cpu().detach().numpy() * prop_std + prop_mean)
    e_loss = calc_normalized_loss(e_logits, e_target)
    print('target MSE:', e_loss.cpu().item())
    e_mae = torch.abs(e_logits - e_target).mean(dim=0)
    print('target MAE:', e_mae.cpu().detach().numpy() * prop_std)
    # print(forward_time)
    print(bp_time)
    print(model.total_forward_time)
    print(model.layer_forward_time)

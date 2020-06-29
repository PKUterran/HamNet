import numpy as np
# import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import gc
from torch.nn import CrossEntropyLoss
from itertools import chain
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from .config import BBBPConfig, HIVConfig
from .HeteroGraph import HeteroGraph
from data.reader import load_hiv, load_bbbp
from utils.sample import sample
from net.models import MLP, AMPNN
from visualize.regress import plt_multiple_scatter

HIV_GRAPH_PATH = 'graphs/HIV/'
BBBP_GRAPH_PATH = 'graphs/BBBP/'


def set_seed(seed: int, use_cuda: bool):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def train_hiv(seed: int = 19700101, limit: int = -1, use_cuda: bool = True, use_tqdm=True, use_model='HamGN',
              dataset='HIV'):
    set_seed(seed, use_cuda)
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    if dataset == 'HIV':
        smiles, info_list, properties = load_hiv(limit)
        graph_path = HIV_GRAPH_PATH
        default_config = HIVConfig
    elif dataset == 'BBBP':
        smiles, info_list, properties = load_bbbp(limit)
        graph_path = BBBP_GRAPH_PATH
        default_config = BBBPConfig
    else:
        assert False, "Unknown dataset: {}.".format(dataset)
    n_label = properties.max() + 1
    properties = torch.tensor(properties, dtype=torch.int64)
    if use_cuda:
        properties = properties.cuda()
    molecules = [HeteroGraph(info['nf'], info['ef'], info['us'], info['vs'], info['em']) for info in info_list]
    n_dim = molecules[0].n_dim
    e_dim = molecules[0].e_dim
    node_num = len(molecules)

    train_mask, validate_mask, test_mask = sample(list(range(node_num)),
                                                  default_config.TRAIN_PER,
                                                  default_config.VALIDATE_PER,
                                                  default_config.TEST_PER)
    n_seg = int(len(train_mask) / (default_config.BATCH + 1))
    train_mask_list = [train_mask[i::n_seg] for i in range(n_seg)]
    n_seg = int(len(validate_mask) / (default_config.BATCH + 1))
    validate_mask_list = [validate_mask[i::n_seg] for i in range(n_seg)]
    n_seg = int(len(test_mask) / (default_config.BATCH + 1))
    test_mask_list = [test_mask[i::n_seg] for i in range(n_seg)]
    print(train_mask, validate_mask, test_mask)
    print(len(train_mask_list), len(validate_mask_list), len(test_mask_list))

    if use_model == 'HamGN':
        model = DynamicGraphEncoder(n_dim=n_dim,
                                    e_dim=e_dim,
                                    default_config=default_config,
                                    use_cuda=use_cuda)
    elif use_model == 'AMPNN':
        model = AMPNN(n_dim=n_dim,
                      e_dim=e_dim,
                      default_config=default_config,
                      use_cuda=use_cuda)
    else:
        assert False, 'Undefined model: {}!'.format(use_model)
    classifier = MLP(default_config.F_DIM, n_label, h_dims=default_config.H_DIMS, dropout=default_config.DROPOUT,
                     activation='softmax')
    if use_cuda:
        model.cuda()
        classifier.cuda()
    params = list(chain(model.parameters(), classifier.parameters()))
    for param in params:
        print(param.shape)
    optimizer = optim.Adam(params, lr=default_config.LR, weight_decay=default_config.DECAY)
    current_lr = default_config.LR
    loss_fuc = CrossEntropyLoss()
    # forward_time = 0.
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
            if name and len(matrix_mask_dicts.keys()) < default_config.MAX_DICT:
                matrix_mask_dicts[name] = mm_tuple

        if use_model == 'HamGN':
            embeddings, s_loss, c_loss, a_loss = model(nfs, efs, us, vs, mm_tuple)
            # if np.random.randint(0, 1000) == 0:
            #     print(embeddings.cpu().detach().numpy())
            s_losses.append(s_loss.cpu().item())
            c_losses.append(c_loss.cpu().item())
            a_losses.append(a_loss.cpu().item())
            std_loss = default_config.GAMMA_S * s_loss + \
                       default_config.GAMMA_C * c_loss + \
                       default_config.GAMMA_A * a_loss
        elif use_model == 'AMPNN':
            embeddings, _ = model(nfs, efs, us, vs, mm_tuple)
            std_loss = 0
        else:
            assert False
        logits = classifier(embeddings)
        # print(logits.cpu())
        target = properties[mask]
        if use_cuda:
            target = target.cuda()
        return logits, target, std_loss

    def train(mask_list: list, name=None):
        model.train()
        classifier.train()

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
            u_loss = loss_fuc(logits, target)
            u_losses.append(u_loss.cpu().item())
            loss = u_loss + std_loss
            loss.backward()
            optimizer.step()
            nonlocal current_lr
            current_lr *= 1 - default_config.DECAY

        if use_model == 'HamGN':
            print('\t\tStationary loss: {:.4f}'.format(np.average(s_losses)))
            print('\t\tCentrality loss: {:.4f}'.format(np.average(c_losses)))
            print('\t\tAffinity loss: {:.4f}'.format(np.average(a_losses)))
        print('\t\tSemi-supervised loss: {:.4f}'.format(np.average(u_losses)))

    def evaluate(mask_list: list, name=None, visualize=None):
        model.eval()
        classifier.eval()
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
            loss = loss_fuc(logits, target)
            losses.append(loss.cpu().item())

            logits_list.append(logits.cpu().detach().numpy())
            target_list.append(target.cpu().detach().numpy())
            if visualize:
                masks.extend(m)

        all_logits = np.vstack(logits_list)
        all_target = np.concatenate(target_list)
        print('\t\tLoss: {:.3f}'.format(np.average(losses)))
        print('\t\tROC: {:.3f}'.format(roc_auc_score(all_target, all_logits[:, 1])))

        if visualize:
            best_ids, best_ds, worst_ids, worst_ds = \
                plt_multiple_scatter(graph_path + visualize, masks, all_logits, all_target)
            print('\t\tBest performance on:')
            for i, d in zip(best_ids, best_ds):
                print('\t\t\t{}: {}'.format(smiles[i], d))
            print('\t\tWorst performance on:')
            for i, d in zip(worst_ids, worst_ds):
                print('\t\t\t{}: {}'.format(smiles[i], d))

    for epoch in range(default_config.ITERATION):
        print('In iteration {}:'.format(epoch + 1))
        print('\tLearning rate: {:.8e}'.format(current_lr))
        print('\tTraining: ')
        train(train_mask_list, name='train')
        print('\tEvaluating training: ')
        evaluate(train_mask_list, name='train')
        print('\tEvaluating validation: ')
        evaluate(validate_mask_list, name='evaluate')
        print('\tEvaluating test: ')
        evaluate(test_mask_list)
        gc.collect()

    # print(model.total_forward_time)
    # print(model.layer_forward_time)

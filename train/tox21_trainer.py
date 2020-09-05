import numpy as np
import torch
import torch.optim as optim
import gc
import os
import json
from torch.nn import BCELoss
from itertools import chain
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from .config import MODEL_CONFIG_TOX21 as DEFAULT_CONFIG
from .HeteroGraph import HeteroGraph
from data.reader import load_tox21
from utils.sample import sample
from utils.cache import MatrixCache
from net.models import MLP, AMPNN
from visualize.regress import plt_multiple_scatter

LOG_PATH = 'logs/TOX21/'
TOX21_GRAPH_PATH = 'graphs/TOX21/'


def set_seed(seed: int, use_cuda: bool):
    np.random.seed(16880611)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def train_tox21(seed: int = 19700101, limit: int = -1, use_cuda: bool = True, use_tqdm=True,
                force_save=False, special_config: dict = None,
                position_encoder_path: str = 'net/pe.pt', dataset='TOX21', tag='std'):
    cfg = DEFAULT_CONFIG.copy()
    if special_config:
        cfg.update(special_config)
    for k, v in cfg.items():
        print(k, ':', v)
    set_seed(seed, use_cuda)
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    if dataset == 'TOX21':
        smiles, info_list, properties = load_tox21(limit)
        graph_path = TOX21_GRAPH_PATH
    else:
        assert False, "Unknown dataset: {}.".format(dataset)
    n_label = properties.shape[-1]
    is_nan = np.isnan(properties)
    properties[is_nan] = 0.0
    not_nan = np.logical_not(is_nan).astype(np.float)
    not_nan_mask = not_nan.astype(np.int)
    not_nan = torch.tensor(not_nan, dtype=torch.float32)
    properties = torch.tensor(properties, dtype=torch.float32)
    if use_cuda:
        not_nan = not_nan.cuda()
        properties = properties.cuda()
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
    print(train_mask, validate_mask, test_mask)
    print(len(train_mask_list), len(validate_mask_list), len(test_mask_list))

    if position_encoder_path and os.path.exists(position_encoder_path):
        position_encoder = torch.load(position_encoder_path)
        position_encoder.eval()
    else:
        print('NO POSITION ENCODER IS BEING USED!!!')
        position_encoder = None
    model = AMPNN(n_dim=n_dim,
                  e_dim=e_dim,
                  config=cfg,
                  position_encoder=position_encoder,
                  use_cuda=use_cuda)
    classifier = MLP(cfg['F_DIM'], n_label, h_dims=cfg['MLP_DIMS'], dropout=cfg['DROPOUT'], activation='sigmoid')
    if use_cuda:
        model.cuda()
        classifier.cuda()
    params = list(chain(model.parameters(), classifier.parameters()))
    for param in params:
        print(param.shape)
    optimizer = optim.Adam(params, lr=cfg['LR'], weight_decay=cfg['DECAY'])
    current_lr = cfg['LR']
    matrix_cache = MatrixCache(cfg['MAX_DICT'])
    loss_fuc = BCELoss()
    logs = []

    def forward(mask: list, name=None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        nfs = torch.cat([molecules[i].node_features for i in mask])
        efs = torch.cat([molecules[i].edge_features for i in mask])
        if use_cuda:
            nfs = nfs.cuda()
            efs = efs.cuda()

        us, vs, mm_tuple = matrix_cache.fetch(molecules, mask, nfs, name, use_cuda)

        embeddings, _ = model(nfs, efs, us, vs, mm_tuple, name, [smiles[i] for i in mask])
        std_loss = 0
        logits = classifier(embeddings) * not_nan[mask, :]
        target = properties[mask, :]
        if use_cuda:
            target = target.cuda()
        return logits, target, std_loss

    def train(mask_list: list, name=None):
        model.train()
        classifier.train()
        u_losses = []

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
            current_lr *= 1 - cfg['DECAY']

        print('\t\tSemi-supervised loss: {:.4f}'.format(np.average(u_losses)))
        logs[-1].update({'on_train_loss': np.average(u_losses)})

    def calc_masked_roc(logits, target, mask) -> float:
        rocs = []
        for i in range(n_label):
            l = logits[:, i]
            t = target[:, i]
            # nnm = not_nan_mask[mask, i]
            # l = l[nnm == 1]
            # t = t[nnm == 1]
            rocs.append(roc_auc_score(t, l))
            # print(l.shape)
        # print(rocs)
        return np.average(rocs)

    def evaluate(mask_list: list, name=None):
        model.eval()
        classifier.eval()
        losses = []
        mask = []
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

            mask.extend(m)
            logits_list.append(logits.cpu().detach().numpy())
            target_list.append(target.cpu().detach().numpy())

        all_logits = np.vstack(logits_list)
        all_target = np.vstack(target_list)
        # print(all_logits[: 10])
        # print(all_target[: 10])
        roc = calc_masked_roc(all_logits, all_target, mask)
        print('\t\tLoss: {:.3f}'.format(np.average(losses)))
        print('\t\tROC: {:.3f}'.format(roc))
        logs[-1].update({'{}_loss'.format(name): np.average(losses)})
        logs[-1].update({'{}_metric'.format(name): roc})

    for epoch in range(cfg['ITERATION']):
        logs.append({'epoch': epoch + 1})
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
        d = {'metric': 'Multi-ROC', 'logs': logs}
        with open('{}{}.json'.format(LOG_PATH, tag), 'w+', encoding='utf-8') as fp:
            json.dump(d, fp)

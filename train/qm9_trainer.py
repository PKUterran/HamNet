import numpy as np
import torch
import torch.optim as optim
import gc
import os
import json
from torch.nn import MSELoss
from itertools import chain
from statsmodels import robust
from tqdm import tqdm

from .config import MODEL_CONFIG_QM9 as DEFAULT_CONFIG
from .HeteroGraph import HeteroGraph
from data.reader import load_qm9
from utils.sample import sample
from utils.cache import MatrixCache
from net.models import MLP, AMPNN
from visualize.regress import plt_multiple_scatter

GRAPH_PATH = 'graphs/QM9/'
LOG_PATH = 'logs/QM9/'


def set_seed(seed: int, use_cuda: bool):
    np.random.seed(16880611)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def train_qm9(seed: int = 19700101, limit: int = -1, use_cuda: bool = True, use_tqdm=True, use_pos=False,
              prop=list(range(12)), force_save=False, special_config: dict = None, q_only=False,
              position_encoder_path: str = 'net/pe.pt', tag='std'):
    cfg = DEFAULT_CONFIG.copy()
    if special_config:
        cfg.update(special_config)
    for k, v in cfg.items():
        print(k, ':', v)
    set_seed(seed, use_cuda)
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    smiles, info_list, properties = load_qm9(limit, force_save=force_save, use_pos=use_pos)
    properties = properties[:, prop]
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

    t_properties = properties[train_mask, :]
    prop_mean = np.mean(t_properties, axis=0)
    print('mean:', prop_mean)
    prop_std = np.std(t_properties.tolist(), axis=0, ddof=1)
    print('std:', prop_std)
    prop_mad = robust.mad(t_properties.tolist(), axis=0)
    print('mad:', prop_mad)
    ratio = (prop_std / prop_mad) ** 2
    norm_properties = (properties - prop_mean) / prop_std

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
                  q_only=q_only,
                  use_pos=use_pos,
                  use_cuda=use_cuda)
    regression = MLP(cfg['F_DIM'], len(prop), cfg['MLP_DIMS'], dropout=cfg['DROPOUT'])
    if use_cuda:
        model.cuda()
        regression.cuda()
    for name, param in chain(model.named_parameters(), regression.named_parameters()):
        if param.requires_grad:
            print(name, ":", param.shape)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, chain(model.parameters(), regression.parameters())),
                           lr=cfg['LR'], weight_decay=cfg['DECAY'])
    current_lr = cfg['LR']
    matrix_cache = MatrixCache(cfg['MAX_DICT'])
    loss_fuc = MSELoss()
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
            u_loss = calc_normalized_loss(logits, target)
            u_losses.append(u_loss.cpu().item())
            loss = u_loss + std_loss
            loss.backward()
            optimizer.step()
            nonlocal current_lr
            current_lr *= 1 - cfg['DECAY']

        u_loss = np.average(u_losses)
        print('\t\tSemi-supervised loss: {:.4f}'.format(u_loss))
        logs[-1].update({'on_train_loss': u_loss})

    def evaluate(mask_list: list, name=None, visualize=None):
        model.eval()
        regression.eval()
        losses = []
        maes = []
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
            loss = calc_normalized_loss(logits, target)
            mae = torch.abs(logits - target).mean(dim=0)
            losses.append(loss.cpu().item())
            maes.append(mae.cpu().detach().numpy())

            if visualize:
                masks.extend(m)
                logits_list.append(logits.cpu().detach().numpy())
                target_list.append(target.cpu().detach().numpy())

        u_loss = np.average(losses)
        print('\t\tLoss: {:.3f}'.format(u_loss))
        print('\t\tMAE: {}.'.format(np.average(maes, axis=0) * prop_mad))
        print('\t\tMulti-MAE: {:.3f}.'.format(np.average(maes, axis=0).sum()))
        logs[-1].update({'{}_loss'.format(name): u_loss})
        logs[-1].update({'{}_metric'.format(name): sum(np.average(maes, axis=0))})

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

    for epoch in range(cfg['ITERATION']):
        logs.append({'epoch': epoch + 1})
        print('In iteration {}:'.format(epoch + 1))
        print('\tLearning rate: {:.8e}'.format(current_lr))
        print('\tTraining: ')
        train(train_mask_list, name='train')
        print('\tEvaluating training: ')
        evaluate(train_mask_list, name='train',
                 # visualize='train_{}'.format(epoch + 1) if (epoch + 1) % cfg['EVAL'] == 0 else None,
                 )
        print('\tEvaluating validation: ')
        evaluate(validate_mask_list, name='evaluate',
                 # visualize='val_{}'.format(epoch + 1) if (epoch + 1) % cfg['EVAL'] == 0 else None,
                 )
        print('\tEvaluating test: ')
        evaluate(test_mask_list, name='test',
                 # visualize='test' if epoch + 1 == cfg['ITERATION'] else None,
                 )
        gc.collect()
        d = {'metric': 'Multi-MAE', 'logs': logs}
        with open('{}{}.json'.format(LOG_PATH, tag), 'w+', encoding='utf-8') as fp:
            json.dump(d, fp)

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import gc
import os
import json
from torch.nn import MSELoss
from itertools import chain
from statsmodels import robust
from tqdm import tqdm

from .config import MODEL_CONFIG_LIPOP as DEFAULT_CONFIG
from .HeteroGraph import HeteroGraph
from data.reader import load_lipop, load_freesolv, load_esol
from utils.sample import sample
from utils.cache import MatrixCache
from net.models import MLP, AMPNN
from visualize.regress import plt_multiple_scatter

GRAPH_PATH = 'graphs/Lipop/'
LOG_PATH = 'logs/Lipop/'


def set_seed(seed: int, use_cuda: bool):
    np.random.seed(19700101)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def train_lipop(seed: int = 19700101, limit: int = -1, use_cuda: bool = True, use_tqdm=True,
                force_save=False, special_config: dict = None,
                position_encoder_path: str = 'net/pe.pt', tag='std', dataset='Lipop'):
    cfg = DEFAULT_CONFIG.copy()
    if special_config:
        cfg.update(special_config)
    for k, v in cfg.items():
        print(k, ':', v)
    set_seed(seed, use_cuda)
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    if dataset == 'FreeSolv':
        smiles, info_list, properties = load_freesolv(limit, force_save=force_save)
    elif dataset == 'ESOL':
        smiles, info_list, properties = load_esol(limit, force_save=force_save)
    else:
        smiles, info_list, properties = load_lipop(limit, force_save=force_save)
    molecules = [HeteroGraph(info['nf'], info['ef'], info['us'], info['vs'], info['em']) for info in info_list]
    n_dim = molecules[0].n_dim
    e_dim = molecules[0].e_dim
    node_num = len(molecules)

    train_mask, validate_mask, test_mask = sample(list(range(node_num)),
                                                  cfg['TRAIN_PER'],
                                                  cfg['VALIDATE_PER'],
                                                  cfg['TEST_PER'])
    n_seg = int(len(train_mask) / (cfg['BATCH'] + 1))
    n_seg = min(len(train_mask), n_seg)
    train_mask_list = [train_mask[i::n_seg] for i in range(n_seg)]
    n_seg = int(len(validate_mask) / (cfg['BATCH'] + 1))
    n_seg = min(len(validate_mask), n_seg)
    validate_mask_list = [validate_mask[i::n_seg] for i in range(n_seg)]
    n_seg = int(len(test_mask) / (cfg['BATCH'] + 1))
    n_seg = min(len(test_mask), n_seg)
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
                  use_cuda=use_cuda)
    regression = MLP(cfg['F_DIM'], 1, h_dims=cfg['MLP_DIMS'], dropout=cfg['DROPOUT'])
    if use_cuda:
        model.cuda()
        regression.cuda()
    for name, param in chain(model.named_parameters(), regression.named_parameters()):
        if param.requires_grad:
            print(name, ":", param.shape)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, chain(model.parameters(), regression.parameters())),
                           lr=cfg['LR'], weight_decay=cfg['DECAY'])
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=cfg['GAMMA'])
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

    def train(mask_list: list, name=None):
        model.train()
        regression.train()
        u_losses = []
        losses = []

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
            # loss.backward()
            # optimizer.step()
            losses.append(loss)
            if len(losses) >= cfg['PACK'] or i == len(mask_list) - 1:
                (sum(losses) / len(losses)).backward()
                optimizer.step()
                losses.clear()

        u_loss = np.average(u_losses)
        print('\t\tSemi-supervised loss: {:.4f}'.format(u_loss))
        logs[-1].update({'on_train_loss': u_loss})

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
            loss = loss_fuc(logits, target)
            losses.append(loss.cpu().item())

            if visualize:
                masks.extend(m)
                logits_list.append(logits.cpu().detach().numpy())
                target_list.append(target.cpu().detach().numpy())

        mse_loss = np.average(losses) * (prop_std[0] ** 2)
        rmse_loss = np.average([loss ** 0.5 for loss in losses]) * prop_std[0]
        print('\t\tMSE Loss: {:.3f}'.format(mse_loss))
        print('\t\tRMSE Loss: {:.3f}'.format(rmse_loss))
        logs[-1].update({'{}_loss'.format(name): mse_loss})
        logs[-1].update({'{}_metric'.format(name): rmse_loss})

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
        scheduler.step(epoch=epoch)
        print('In iteration {}:'.format(epoch + 1))
        print('\tTraining: ')
        train(train_mask_list, name='train')
        print('\tEvaluating training: ')
        evaluate(train_mask_list, name='train',
                 # visualize='train_{}'.format(epoch + 1) if (epoch + 1) % cfg['EVAL'] == 0 else None
                 )
        print('\tEvaluating validation: ')
        evaluate(validate_mask_list, name='evaluate',
                 # visualize='val_{}'.format(epoch + 1) if (epoch + 1) % cfg['EVAL'] == 0 else None
                 )
        print('\tEvaluating test: ')
        evaluate(test_mask_list,  name='test',
                 # visualize='test' if epoch + 1 == cfg['ITERATION'] else None
                 )
        gc.collect()
        d = {'metric': 'RMSE', 'logs': logs}
        with open('{}{}.json'.format(LOG_PATH, tag), 'w+', encoding='utf-8') as fp:
            json.dump(d, fp)

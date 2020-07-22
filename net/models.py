import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, ReLU, Dropout, ModuleList, Parameter
from itertools import chain
from functools import reduce
from numpy.linalg import norm

from .layers import ConcatMesPassing, PosConcatMesPassing, GRUAggregation, AlignAttendPooling, \
    HamiltonianDerivation, DissipativeHamiltonianDerivation, LstmPQEncoder
from visualize.trajectory import plt_trajectory
from utils.func import re_index


class AMPNN(Module):
    def __init__(self, n_dim: int, e_dim: int, config, position_encoder=None, use_cuda=False):
        super(AMPNN, self).__init__()
        self.h_dim = config['F_DIM']
        self.c_dims = config['C_DIMS']
        self.he_dim = config['HE_DIM']
        self.pos_dim = config['POS_DIM']
        self.layers = len(config['C_DIMS'])
        self.m_radius = config['M_RADIUS']
        self.dropout = config['DROPOUT']
        self.position_encoder = position_encoder
        self.use_cuda = use_cuda

        in_dims = [self.h_dim] * (self.layers + 1)
        self.e_dim = e_dim
        self.FC_N = Linear(n_dim, self.h_dim, bias=True)
        self.FC_E = Linear(e_dim, self.he_dim, bias=True)
        if self.position_encoder:
            self.Ms = ModuleList([PosConcatMesPassing(in_dims[i], self.he_dim, self.pos_dim, self.c_dims[i],
                                                      dropout=self.dropout)
                                  for i in range(self.layers)])
        else:
            self.Ms = ModuleList([ConcatMesPassing(in_dims[i], self.he_dim, self.c_dims[i], dropout=self.dropout)
                                  for i in range(self.layers)])
        self.Us = ModuleList([GRUAggregation(self.c_dims[i], in_dims[i]) for i in range(self.layers)])
        self.R = AlignAttendPooling(in_dims[-1], in_dims[-1], self.m_radius, use_cuda, dropout=self.dropout)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, us: list, vs: list,
                matrix_mask_tuple: tuple, name: str = '') -> (torch.Tensor, torch.Tensor):
        assert edge_features.shape[0] == len(us) and edge_features.shape[0] == len(vs), \
            '{}, {}, {}.'.format(edge_features.shape, len(us), len(vs))
        if edge_features.shape[0] == 0:
            edge_features = edge_features.reshape([0, self.e_dim])
        if self.position_encoder:
            pos_features = self.position_encoder.transform(node_features, edge_features, us, vs,
                                                           matrix_mask_tuple, name)[0]
            uv_pos_features = pos_features[us] - pos_features[vs]
        else:
            uv_pos_features = None
        node_features = F.leaky_relu(self.FC_N(node_features))
        edge_features = F.leaky_relu(self.FC_E(edge_features))
        mol_node_matrix, mol_node_mask = matrix_mask_tuple[0], matrix_mask_tuple[1]
        node_edge_matrix, node_edge_mask = matrix_mask_tuple[2], matrix_mask_tuple[3]
        for i in range(self.layers):
            u_features = node_features[us]
            v_features = node_features[vs]
            if self.position_encoder:
                context_features, new_edge_features = self.Ms[i](u_features, v_features, edge_features, uv_pos_features,
                                                                 node_edge_matrix, node_edge_mask)
            else:
                assert not uv_pos_features
                context_features, new_edge_features = self.Ms[i](u_features, v_features, edge_features,
                                                                 node_edge_matrix, node_edge_mask)
            new_node_features = self.Us[i](node_features, context_features)

            if i != self.layers - 1:
                new_node_features = F.relu(new_node_features)
            node_features = new_node_features
            edge_features = new_edge_features

        readout, a = self.R(node_features, mol_node_matrix, mol_node_mask)
        return readout, a

    @staticmethod
    def produce_node_edge_matrix(n_node: int, us: list, vs: list, edge_mask: list) -> (torch.Tensor, torch.Tensor):
        us = np.array(us)
        vs = np.array(vs)
        mat = np.full([n_node, us.shape[0]], 0., dtype=np.int)
        mask = np.full([n_node, us.shape[0]], -1e6, dtype=np.int)
        for i in range(n_node):
            node_edge = np.logical_and(np.logical_or(us == i, vs == i), edge_mask)
            mat[i, node_edge] = 1
            mask[i, node_edge] = 0
        mat = torch.from_numpy(mat).type(torch.float32)
        mask = torch.from_numpy(mask).type(torch.float32)
        return mat, mask


class PositionEncoder(Module):
    def __init__(self, n_dim, e_dim, config, use_cuda=True):
        super(PositionEncoder, self).__init__()
        self.p_dim = config['PQ_DIM']
        self.q_dim = config['PQ_DIM']
        self.layers = config['HGN_LAYERS']
        self.tau = config['TAU']
        self.dropout = config['DROPOUT']
        self.use_cuda = use_cuda

        self.e_encoder = Linear(n_dim + e_dim + n_dim, 1)
        self.pq_encoder = LstmPQEncoder(n_dim, self.p_dim)
        # self.derivation = HamiltonianDerivation(self.p_dim, self.q_dim, dropout=0.0)
        self.derivation = DissipativeHamiltonianDerivation(n_dim, self.p_dim, self.q_dim, dropout=0.0)
        self.dn23 = Linear(self.q_dim, 3)

        self.cache = {}

    def forward(self, v_features: torch.Tensor, e_features: torch.Tensor, us: list, vs: list, matrix_mask_tuple: tuple):
        mol_node_matrix, mol_node_mask = matrix_mask_tuple[0], matrix_mask_tuple[1]
        node_edge_matrix, node_edge_mask = matrix_mask_tuple[2], matrix_mask_tuple[3]

        u_e_v_features = torch.cat([v_features[us], e_features, v_features[vs]], dim=1)
        e_weight = torch.diag(torch.sigmoid(self.e_encoder(u_e_v_features)).view([-1]))
        e = node_edge_matrix @ e_weight @ node_edge_matrix.t()

        p0, q0 = self.pq_encoder(v_features, mol_node_matrix, e)
        ps = [p0]
        qs = [q0]
        s_losses = []
        c_losses = []
        h = None
        d = None

        for i in range(self.layers):
            dp, dq, h, d = self.derivation(v_features, ps[i], qs[i], e, mol_node_matrix, mol_node_mask,
                                           return_energy=True)
            ps.append(ps[i] + self.tau * dp)
            qs.append(qs[i] + self.tau * dq)

            s_losses.append((dq - ps[i]).norm())
            c_losses.append((mol_node_matrix @ (ps[i + 1] - ps[i])).norm())

        s_loss = sum(s_losses)
        c_loss = sum(c_losses)
        # self.verbose_print(ps, qs, mol_node_matrix, node_edge_matrix, us, vs, verbose)

        return ps[-1], qs[-1], s_loss, c_loss, h, d

    def fit(self, v_features: torch.Tensor, e_features: torch.Tensor, us: list, vs: list,
            matrix_mask_tuple: tuple, fit_pos: torch.Tensor, print_mode=False):
        _, q, s_loss, c_loss, h, d = self(v_features, e_features, us, vs, matrix_mask_tuple)
        pos = self.dn23(q)
        mol_node_matrix = matrix_mask_tuple[0]
        norm_mnm = mol_node_matrix / mol_node_matrix.sum(dim=1).unsqueeze(-1)
        dis_mask = norm_mnm.t() @ norm_mnm
        dis = (pos.unsqueeze(0) - pos.unsqueeze(1)).norm(dim=2)
        fit_dis = (fit_pos.unsqueeze(0) - fit_pos.unsqueeze(1)).norm(dim=2)
        d_loss = ((dis - fit_dis) * dis_mask).pow(2).sum() / mol_node_matrix.shape[0]
        if print_mode:
            print(h.cpu().detach().numpy()[:8])
            print(d.cpu().detach().numpy()[:8])
            print(dis_mask.cpu().detach().numpy()[:8, :8])
            print(dis.cpu().detach().numpy()[:8, :8])
            print(fit_dis.cpu().detach().numpy()[:8, :8])
        return d_loss, s_loss, c_loss, pos

    def transform(self, v_features: torch.Tensor, e_features: torch.Tensor, us: list, vs: list,
                  matrix_mask_tuple: tuple, name=''):
        if name and name in self.cache.keys():
            pq = self.cache[name]
        else:
            p, q, _, _, _, _ = self(v_features, e_features, us, vs, matrix_mask_tuple)
            pq = torch.cat([p.detach(), q.detach()], dim=1)
            if name:
                self.cache[name] = pq
        return pq, 0, 0

    def verbose_print(self, ps, qs, mnm, nem, us, vs, verbose):
        if self.use_cuda:
            ps = [p.cpu() for p in ps]
            qs = [q.cpu() for q in qs]
            mnm = mnm.cpu()
            nem = nem.cpu()
        ps = [p.detach().numpy() for p in ps]
        qs = [q.detach().numpy() for q in qs]
        mnm = mnm.detach().numpy()
        nem = nem.detach().numpy()
        us = np.array(us)
        vs = np.array(vs)
        for m in verbose:
            print('\t\t\tIn mol {}:'.format(m))
            one_hot = np.zeros(shape=[1, mnm.shape[0]], dtype=np.int)
            one_hot[0, m] = 1
            node_mask = one_hot @ mnm
            q = qs[-1][node_mask[0, :] > 0, :]
            distances = norm(np.expand_dims(q, 0) - np.expand_dims(q, 1), axis=2)
            print('{}'.format(distances))
            print('\t\t\t\tAverage distance: {}'.format(np.average(distances) * (1 + 1.0 / (node_mask.sum() - 1))))

            edge_mask = node_mask @ nem
            us_ = re_index(us[edge_mask[0, :] > 0], node_mask[0, :])
            vs_ = re_index(vs[edge_mask[0, :] > 0], node_mask[0, :])
            for u, v in zip(us_, vs_):
                d = norm(qs[-1][u, :] - qs[-1][v, :])
                print('\t\t\t\tBetween {} and {}, distance is {}'.format(u, v, d))
            plt_trajectory([q[node_mask[0, :] > 0, :] for q in qs], us_, vs_, name=str(int(time.time() * 1000)))


class MLP(Module):
    def __init__(self, i_dim: int, o_dim: int, h_dims: list = list(), dropout=0., activation=None):
        super(MLP, self).__init__()

        in_dims = [i_dim] + h_dims
        out_dims = h_dims + [o_dim]
        self.linears = ModuleList([Linear(in_dim, out_dim, bias=True) for in_dim, out_dim in zip(in_dims, out_dims)])
        self.relu = ReLU()
        self.dropout = Dropout(p=dropout)
        self.activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.dropout(inputs)
        for i, linear in enumerate(self.linears):
            h = linear(h)
            if i < len(self.linears) - 1:
                h = self.relu(h)
        if self.activation == 'sigmoid':
            h = torch.sigmoid(h)
        elif self.activation == 'softmax':
            h = torch.softmax(h, dim=1)
        elif not self.activation:
            pass
        else:
            assert False, 'Undefined activation: {}.'.format(self.activation)
        return h


if __name__ == '__main__':
    # model = AMPNN(16, 8, 3, 2)
    # for p in model.parameters():
    #     print(p.shape)
    pass

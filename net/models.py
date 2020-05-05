import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.nn import Parameter, Module, Linear, ReLU, Dropout
from itertools import chain
from functools import reduce

from .config import *
from .layers import ConcatMesPassing, GRUAggregation, AttentivePooling, DirectDerivation, HamiltonianDerivation, \
    GraphConvolutionLayer


class AMPNN(Module):
    def __init__(self, n_dim: int, e_dim: int, h_dim: int, c_dims: list, he_dim: int, layers: int,
                 residual=True, use_cuda=False, dropout=0.):
        super(AMPNN, self).__init__()
        assert len(c_dims) == layers, '{},{}'.format(c_dims, layers)

        ### DEBUG MODE ###
        self.total_forward_time = 0.
        self.layer_forward_time = [[0., 0., 0., 0.] for _ in range(layers)]
        ### ---------- ###

        in_dims = [h_dim] * (layers + 1)
        if residual:
            for i in range(1, layers + 1):
                in_dims[i] += in_dims[i - 1]
        self.e_dim = e_dim
        self.FC_N = Linear(n_dim, h_dim, bias=True)
        self.FC_E = Linear(e_dim, he_dim, bias=True)
        self.Ms = [ConcatMesPassing(in_dims[i], he_dim, c_dims[i], dropout=dropout) for i in range(layers)]
        # self.Us = [DirectAggregation() for _ in range(layers)]
        self.Us = [GRUAggregation(c_dims[i], in_dims[i]) for i in range(layers)]
        self.R = AttentivePooling(in_dims[-1], in_dims[-1], use_cuda, dropout=dropout)
        # self.R = MapPooling(in_dims[-1], int(head_num * in_dims[-1]))
        self.layers = layers
        self.residual = residual
        self.use_cuda = use_cuda
        for i, p in enumerate(self.get_inner_parameters()):
            self.register_parameter('param_' + str(i), p)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, us: list, vs: list,
                matrix_mask_tuple: tuple, global_mask: list) -> (torch.Tensor, torch.Tensor):
        assert edge_features.shape[0] == len(us) and edge_features.shape[0] == len(vs), \
            '{}, {}, {}.'.format(edge_features.shape, len(us), len(vs))
        t0 = time.time()  ### DEBUG
        if edge_features.shape[0] == 0:
            edge_features = edge_features.reshape([0, self.e_dim])
        node_features = F.leaky_relu(self.FC_N(node_features))
        edge_features = F.leaky_relu(self.FC_E(edge_features))
        layer_node_features = [node_features]
        layer_edge_features = [edge_features]
        mol_node_matrix, mol_node_mask = matrix_mask_tuple[0], matrix_mask_tuple[1]
        node_edge_matrix, node_edge_mask = matrix_mask_tuple[2], matrix_mask_tuple[3]
        # node_edge_matrix_global, node_edge_mask_global = matrix_mask_tuple[2], matrix_mask_tuple[3]
        # node_edge_matrix_local, node_edge_mask_local = matrix_mask_tuple[4], matrix_mask_tuple[5]
        for i in range(self.layers):
            t1 = time.time()  ### DEBUG
            u_features = layer_node_features[-1][us]
            v_features = layer_node_features[-1][vs]
            # node_edge_matrix = node_edge_matrix_global if global_mask[i] else node_edge_matrix_local
            # node_edge_mask = node_edge_mask_global if global_mask[i] else node_edge_mask_local
            self.layer_forward_time[i][0] += time.time() - t1  ### DEBUG

            t1 = time.time()  ### DEBUG
            context_features, new_edge_features = self.Ms[i](u_features, v_features, layer_edge_features[-1],
                                                             node_edge_matrix, node_edge_mask)
            self.layer_forward_time[i][1] += time.time() - t1  ### DEBUG

            t1 = time.time()  ### DEBUG
            new_node_features = self.Us[i](layer_node_features[-1], context_features)
            self.layer_forward_time[i][2] += time.time() - t1  ### DEBUG

            t1 = time.time()  ### DEBUG
            if i != self.layers - 1:
                new_node_features = F.relu(new_node_features)
            if self.residual:
                cat_node_features = torch.cat([layer_node_features[-1], new_node_features], dim=1)
                layer_node_features.append(cat_node_features)
            else:
                layer_node_features.append(new_node_features)
            layer_edge_features.append(new_edge_features)
            self.layer_forward_time[i][3] += time.time() - t1  ### DEBUG

        readout, a = self.R(layer_node_features[-1], mol_node_matrix, mol_node_mask)
        self.total_forward_time += time.time() - t0  ### DEBUG
        return readout, a

    def get_inner_parameters(self):
        return chain(
            reduce(lambda x, y: chain(x, y), map(lambda x: x.parameters(), self.Ms)),
            reduce(lambda x, y: chain(x, y), map(lambda x: x.parameters(), self.Us)),
        )

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


class DynamicGraphEncoder(Module):
    def __init__(self, v_dim, e_dim, p_dim, q_dim, f_dim, layers, hamilton=False, discrete=True, tau=0.2, gamma=1,
                 dropout=0., use_cuda=True):
        super(DynamicGraphEncoder, self).__init__()
        self.layers = layers
        self.hamilton = hamilton
        self.discrete = discrete
        self.tau = tau
        self.gamma = gamma
        self.use_cuda = use_cuda

        self.e_encoder = Linear(v_dim + e_dim + v_dim, 1)
        self.p_encoder = GraphConvolutionLayer(v_dim, p_dim, activation=torch.tanh)
        self.q_encoder = GraphConvolutionLayer(v_dim, q_dim, activation=torch.tanh)
        if hamilton:
            self.derivation = HamiltonianDerivation(p_dim, q_dim)
        else:
            self.derivation = DirectDerivation(p_dim, q_dim)
        self.readout = AttentivePooling(v_dim + p_dim + q_dim, f_dim, use_cuda=use_cuda, dropout=dropout)

    def forward(self, v_features: torch.Tensor, e_features: torch.Tensor, us: list, vs: list, matrix_mask_tuple: tuple):
        mol_node_matrix, mol_node_mask = matrix_mask_tuple[0], matrix_mask_tuple[1]
        node_edge_matrix, node_edge_mask = matrix_mask_tuple[2], matrix_mask_tuple[3]

        u_e_v_features = torch.cat([v_features[us], e_features, v_features[vs]], dim=1)
        e_weight = torch.diag(torch.sigmoid(self.e_encoder(u_e_v_features)).view([-1]))
        e = node_edge_matrix @ e_weight @ node_edge_matrix.t()
        if self.use_cuda:
            e_noi = e - torch.eye(e.shape[0]).cuda() * e
        else:
            e_noi = e - torch.eye(e.shape[0]) * e

        ps = [self.p_encoder(v_features, e)]
        qs = [self.q_encoder(v_features, e)]
        c_losses = [(mol_node_matrix @ qs[0]).norm()]

        for i in range(self.layers):
            dp, dq = self.derivation(ps[i], qs[i], e)
            if self.discrete:
                ps.append(ps[i] + self.tau * dp)
                qs.append(qs[i] + self.tau * dq)
            else:
                raise NotImplementedError()
            c_losses.append((mol_node_matrix @ qs[i + 1]).norm())

        f, _ = self.readout(torch.cat([v_features, ps[-1], qs[-1]], dim=1), mol_node_matrix, mol_node_mask)

        s_loss = torch.abs(ps[-1]).sum()
        c_loss = sum(c_losses)
        dis = (qs[-1].unsqueeze(0) - qs[-1].unsqueeze(1)).norm(dim=2)
        # if np.random.randint(0, 100) == 0:
        #     print(dis.cpu().detach().numpy())
        #     print(e_noi)
        #     print((e_noi * F.relu(dis - self.gamma)).cpu().detach().numpy())
        a_loss = (e_noi * (dis - self.gamma) ** 2).sum()

        return f, s_loss, c_loss, a_loss


class MLP(Module):
    def __init__(self, i_dim: int, h_dim: int, o_dim: int, dropout=0.):
        super(MLP, self).__init__()

        # self.linear1 = Linear(i_dim, h_dim, bias=False)
        # self.act = ReLU()
        # self.linear2 = Linear(h_dim, o_dim, bias=True)

        self.linear1 = Linear(i_dim, o_dim, bias=True)

        self.dropout = Dropout(p=dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # return self.linear2(self.act(self.linear1(self.dropout(inputs))))
        return self.linear1(self.dropout(inputs))


if __name__ == '__main__':
    # model = AMPNN(16, 8, 3, 2)
    # for p in model.parameters():
    #     print(p.shape)
    pass

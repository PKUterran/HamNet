import torch
import torch.autograd as autograd
from torch.nn import Parameter, Module, Linear, Softmax, LogSoftmax, Sigmoid, GRUCell, ELU, LeakyReLU, Dropout, Tanh, \
    LSTM, Softplus, ReLU, ModuleList
from torch.nn.utils.rnn import pad_sequence
from itertools import chain

from data.encode import get_default_atoms_massive_matrix


class MolGATMesPassing(Module):
    def __init__(self, n_dim: int, e_dim: int, p_dim: int, c_dim: int, dropout=0., use_cuda=False):
        super(MolGATMesPassing, self).__init__()
        self.linear = Linear(n_dim + p_dim + n_dim, c_dim, bias=True)
        self.linear_e = Linear(n_dim + p_dim + n_dim, e_dim, bias=True)
        self.relu1 = LeakyReLU()
        self.relu2 = LeakyReLU()
        self.relu_e = LeakyReLU()
        self.attention = Linear(e_dim + p_dim, 1, bias=True)
        self.softmax = Softmax(dim=1)
        self.elu = ELU()
        self.dropout = Dropout(p=dropout)
        self.use_cuda = use_cuda

    def forward(self, u_features: torch.Tensor, v_features: torch.Tensor,
                edge_features: torch.Tensor, pos_features: torch.Tensor,
                node_edge_matrix: torch.Tensor, node_edge_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        if pos_features is not None:
            u_p_v_features = torch.cat([u_features, pos_features, v_features], dim=1)
        else:
            u_p_v_features = torch.cat([u_features, v_features], dim=1)
        if u_p_v_features.shape[0]:
            u_p_v_features = self.dropout(u_p_v_features)
        new_edge_features = self.relu_e(self.linear_e(u_p_v_features))
        neighbor_features = self.relu1(self.linear(u_p_v_features))
        if neighbor_features.shape[0]:
            neighbor_features = self.dropout(neighbor_features)
        if pos_features is not None:
            a = self.relu2(self.attention(torch.cat([edge_features, pos_features], dim=1)))
        else:
            a = self.relu2(self.attention(edge_features))
        d = a.view([-1]).diag()
        node_edge_weight = node_edge_matrix @ d + node_edge_mask
        node_edge_weight = self.softmax(node_edge_weight)
        context_features = self.elu(node_edge_weight @ neighbor_features)
        # context_features = self.elu((node_edge_weight @ neighbor_features) *
        #                             self.get_booster(neighbor_features, node_edge_matrix))
        return context_features, new_edge_features

    def get_booster(self, n, nem):
        n_num = nem.shape[0]
        e_num = nem.shape[1]
        eye = torch.eye(e_num)
        if self.use_cuda:
            eye = eye.cuda()
        m = eye.expand([n_num, e_num, e_num])
        e_ = nem.unsqueeze(1).expand([n_num, e_num, e_num])
        e1 = e_ * m
        a = e1 @ n
        t = torch.max(a, dim=-2)[0]
        return t


class ConcatMesPassing(Module):
    def __init__(self, n_dim: int, e_dim: int, c_dim: int, dropout=0.):
        super(ConcatMesPassing, self).__init__()
        self.linear = Linear(n_dim + e_dim + n_dim, c_dim, bias=True)
        self.linear_e = Linear(n_dim + e_dim + n_dim, e_dim, bias=True)
        self.relu1 = LeakyReLU()
        self.relu2 = LeakyReLU()
        self.relu_e = LeakyReLU()
        self.attention = Linear(n_dim + e_dim + n_dim, 1, bias=True)
        self.softmax = Softmax(dim=1)
        self.elu = ELU()
        self.dropout = Dropout(p=dropout)

    def forward(self, u_features: torch.Tensor, v_features: torch.Tensor, edge_features: torch.Tensor,
                node_edge_matrix: torch.Tensor, node_edge_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        u_e_v_features = torch.cat([u_features, edge_features, v_features], dim=1)
        if u_e_v_features.shape[0]:
            u_e_v_features = self.dropout(u_e_v_features)
        new_edge_features = self.relu_e(self.linear_e(u_e_v_features))
        u_e_v_features = torch.cat([u_features, new_edge_features, v_features], dim=1)
        if u_e_v_features.shape[0]:
            u_e_v_features = self.dropout(u_e_v_features)
        neighbor_features = self.relu1(self.linear(u_e_v_features))
        if neighbor_features.shape[0]:
            neighbor_features = self.dropout(neighbor_features)
        a = self.relu2(self.attention(u_e_v_features))
        d = a.view([-1]).diag()
        node_edge_weight = node_edge_matrix @ d + node_edge_mask
        # print(node_edge_weight)
        node_edge_weight = self.softmax(node_edge_weight)
        # print(node_edge_weight)
        context_features = self.elu(node_edge_weight @ neighbor_features)
        return context_features, new_edge_features


class PosConcatMesPassing(Module):
    def __init__(self, n_dim: int, e_dim: int, p_dim: int, c_dim: int, dropout=0.):
        super(PosConcatMesPassing, self).__init__()
        self.linear = Linear(n_dim + e_dim + p_dim + n_dim, c_dim, bias=True)
        self.linear_e = Linear(n_dim + e_dim + p_dim + n_dim, e_dim, bias=True)
        self.relu1 = LeakyReLU()
        self.relu2 = LeakyReLU()
        self.relu_e = LeakyReLU()
        self.attention = Linear(n_dim + e_dim + p_dim + n_dim, 1, bias=True)
        self.softmax = Softmax(dim=1)
        self.elu = ELU()
        self.dropout = Dropout(p=dropout)

    def forward(self, u_features: torch.Tensor, v_features: torch.Tensor,
                edge_features: torch.Tensor, pos_features: torch.Tensor,
                node_edge_matrix: torch.Tensor, node_edge_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        u_e_p_v_features = torch.cat([u_features, edge_features, pos_features, v_features], dim=1)
        if u_e_p_v_features.shape[0]:
            u_e_p_v_features = self.dropout(u_e_p_v_features)
        new_edge_features = self.relu_e(self.linear_e(u_e_p_v_features))
        u_e_p_v_features = torch.cat([u_features, new_edge_features, pos_features, v_features], dim=1)
        if u_e_p_v_features.shape[0]:
            u_e_p_v_features = self.dropout(u_e_p_v_features)
        neighbor_features = self.relu1(self.linear(u_e_p_v_features))
        if neighbor_features.shape[0]:
            neighbor_features = self.dropout(neighbor_features)
        a = self.relu2(self.attention(u_e_p_v_features))
        d = a.view([-1]).diag()
        node_edge_weight = node_edge_matrix @ d + node_edge_mask
        # print(node_edge_weight)
        node_edge_weight = self.softmax(node_edge_weight)
        # print(node_edge_weight)
        context_features = self.elu(node_edge_weight @ neighbor_features)
        return context_features, new_edge_features


class DirectAggregation(Module):
    def __init__(self):
        super(DirectAggregation, self).__init__()

    def forward(self, node_features: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        return context_features


class GRUAggregation(Module):
    def __init__(self, c_dim, h_dim):
        super(GRUAggregation, self).__init__()
        self.gru = GRUCell(c_dim, h_dim)

    def forward(self, node_features: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        new_node_features = self.gru(context_features, node_features)
        return new_node_features


class AttentivePooling(Module):
    def __init__(self, dim: int, out_dim: int, use_cuda=False, dropout=0.):
        super(AttentivePooling, self).__init__()
        self.use_cuda = use_cuda
        self.attend = Linear(dim, out_dim, bias=True)
        self.align = Linear(dim, 1, bias=True)
        self.softmax = Softmax(dim=1)
        self.dropout = Dropout(p=dropout)
        self.relu = LeakyReLU()

    def forward(self, node_features: torch.Tensor, mol_node_matrix: torch.Tensor, mol_node_mask: torch.Tensor) \
            -> (torch.Tensor, torch.Tensor):
        h = self.relu(self.attend(self.dropout(node_features)))
        a = self.align(self.dropout(node_features))
        d = a.view([-1]).diag()
        mol_node_weight = mol_node_matrix @ d + mol_node_mask
        mol_node_weight = self.softmax(mol_node_weight)
        return mol_node_weight @ h, mol_node_weight


class AlignAttendPooling(Module):
    def __init__(self, c_dim: int, m_dim: int, p_dim: int, radius=2, use_cuda=False, dropout=0., use_gru=True):
        super(AlignAttendPooling, self).__init__()
        self.use_cuda = use_cuda
        self.use_gru = use_gru
        self.radius = radius
        self.map = Linear(c_dim, m_dim)
        self.relu = LeakyReLU()
        self.relu1 = LeakyReLU()
        if use_gru:
            self.gru = GRUCell(c_dim, m_dim)
        else:
            self.linear = Linear(c_dim + m_dim, m_dim)
        self.attend = Linear(c_dim + p_dim, c_dim)
        self.align = Linear(m_dim + c_dim + p_dim, 1)
        self.softmax = Softmax(dim=1)
        self.elu = ELU()
        self.relu2 = ReLU()
        self.dropout = Dropout(p=dropout)

    def forward(self, node_features: torch.Tensor, pos_features: torch.Tensor,
                mol_node_matrix: torch.Tensor, mol_node_mask: torch.Tensor) \
            -> (torch.Tensor, torch.Tensor):
        mol_features = mol_node_matrix @ self.relu(self.map(node_features))

        for i in range(self.radius):
            if pos_features is not None:
                # print(node_features.shape)
                # print(pos_features.shape)
                h = self.attend(self.dropout(torch.cat([node_features, pos_features], dim=-1)))
                a = self.relu1(self.align(torch.cat([mol_node_matrix.t() @ mol_features, pos_features, node_features],
                                                    dim=-1)))
                # a = self.align(torch.cat([mol_node_matrix.t() @ mol_features, pos_features, node_features],
                #                          dim=-1))
            else:
                h = self.attend(self.dropout(node_features))
                a = self.relu1(self.align(torch.cat([mol_node_matrix.t() @ mol_features, node_features],
                                                    dim=-1)))
                # a = self.align(torch.cat([mol_node_matrix.t() @ mol_features, node_features],
                #                          dim=-1))
            d = a.view([-1]).diag()
            mol_node_weight = self.softmax(mol_node_matrix @ d + mol_node_mask)
            context = self.elu(mol_node_weight @ h)
            if self.use_gru:
                mol_features = self.relu2(self.gru(context, mol_features))
            else:
                mol_features = self.relu2(self.linear(torch.cat([context, mol_features], dim=-1)))

        return mol_features, None


class GraphConvolutionLayer(Module):
    def __init__(self, i_dim: int, o_dim: int, h_dims: list = list((128,)),
                 activation=None, dropout=0.0, residual=False):
        super(GraphConvolutionLayer, self).__init__()
        in_dims = [i_dim] + h_dims
        out_dims = h_dims + [o_dim]
        self.linears = ModuleList([Linear(in_dim, out_dim, bias=True) for in_dim, out_dim in zip(in_dims, out_dims)])
        self.relu = LeakyReLU()
        self.activation = activation
        self.dropout = Dropout(dropout)
        self.residual = residual

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        hs = [self.dropout(x)]
        for i, linear in enumerate(self.linears):
            h = a @ linear(hs[i])
            if i < len(self.linears) - 1:
                h = self.relu(h)
            hs.append(h)
        if self.residual:
            h = torch.cat(hs, dim=-1)
        else:
            h = hs[-1]
        if self.activation == 'tanh':
            h = torch.tanh(h)
        elif not self.activation:
            pass
        else:
            assert False, 'Undefined activation: {}.'.format(self.activation)
        return h


class LstmPQEncoder(Module):
    def __init__(self, in_dim, pq_dim, h_dim=128, num_layers=1, use_cuda=False, disturb=False, use_lstm=True):
        super(LstmPQEncoder, self).__init__()
        self.use_cuda = use_cuda
        self.disturb = disturb
        self.use_lstm = use_lstm
        self.pq_dim = pq_dim
        if self.use_lstm:
            self.gcl = GraphConvolutionLayer(in_dim, h_dim, h_dims=[h_dim], residual=True)
            self.relu = ELU()
            self.rnn = LSTM(in_dim + h_dim * 2, 2 * pq_dim, num_layers)
        else:
            self.gcl = GraphConvolutionLayer(in_dim, 2 * pq_dim, h_dims=[h_dim], residual=False)

    def forward(self, node_features: torch.Tensor, mol_mode_matrix: torch.Tensor, e: torch.Tensor) \
            -> (torch.Tensor, torch.Tensor):
        if self.use_lstm:
            hidden_node_features = self.relu(self.gcl(node_features, e))
            seqs = [hidden_node_features[n == 1, :] for n in mol_mode_matrix]
            lengths = [s.shape[0] for s in seqs]
            m = pad_sequence(seqs)
            output, _ = self.rnn(m)
            ret = torch.cat([output[:lengths[i], i, :] for i in range(len(lengths))])
        else:
            ret = self.gcl(node_features, e)
        if self.disturb:
            d = torch.normal(mean=torch.zeros(ret.size()), std=torch.full(ret.size(), 0.1))
            if self.use_cuda:
                d = d.cuda()
            ret = ret + d
        ret = ret - torch.sum(ret, dim=0) / ret.shape[0]
        return ret[:, :self.pq_dim], ret[:, self.pq_dim:]


class DirectDerivation(Module):
    def __init__(self, p_dim, q_dim, h_dim=128):
        super(DirectDerivation, self).__init__()
        self.p_dim = p_dim
        self.q_dim = q_dim
        self.gcl = GraphConvolutionLayer(p_dim + q_dim, h_dim, h_dims=[])
        self.relu = ELU()
        self.linear = Linear(h_dim, p_dim + q_dim)
        # for param in chain(self.gcl.parameters(), self.linear.parameters()):
        #     self.register_parameter("_", param)

    def forward(self, p, q, e):
        u = torch.cat([p, q], dim=1)
        dp_dq = self.linear(self.relu(self.gcl(u, e)))
        dp = dp_dq[:, :self.p_dim]
        dq = dp_dq[:, self.p_dim:]
        return dp, dq


class Massive(Module):
    def __init__(self, n_dim, use_cuda=False):
        super(Massive, self).__init__()
        self.massive_matrix = torch.from_numpy(get_default_atoms_massive_matrix()).type(torch.float32)
        if use_cuda:
            self.massive_matrix = self.massive_matrix.cuda()

    def forward(self, n):
        m = n @ self.massive_matrix / 50
        # print('m:', m[:5])
        return m


class KineticEnergy(Module):
    def __init__(self, p_dim, h_dim=128):
        super(KineticEnergy, self).__init__()
        self.W = Linear(p_dim, h_dim, bias=False)

    def forward(self, p, m):
        alpha = 1 / m
        pw = self.W(p)
        apwwp = alpha * (pw ** 2)
        if torch.isnan(apwwp.sum()):
            apwwp[torch.isnan(apwwp)] = 0
        t = torch.sum(apwwp, dim=1, keepdim=True)
        # print('t:', t[:1])
        return t


class PotentialEnergy(Module):
    def __init__(self, q_dim, h_dim=32, dropout=0.0, use_cuda=False):
        super(PotentialEnergy, self).__init__()
        self.use_cuda = use_cuda
        self.linear1 = Linear(q_dim, h_dim, bias=True)
        self.softplus = Softplus()

    def forward(self, n, m, q, e, nnm):
        norm_m = m
        mm = norm_m * norm_m.reshape([1, -1])
        eye = torch.eye(nnm.shape[1], dtype=torch.float32)
        if self.use_cuda:
            eye = eye.cuda()
        mask = nnm * mm
        delta_p = torch.unsqueeze(q, dim=0) - torch.unsqueeze(q, dim=1)
        root = self.linear1(delta_p)
        distance = (self.softplus(torch.sum(root ** 2, dim=2))) * (-eye + 1) + eye
        # print(distance)
        energy = mask * (distance ** -2 - distance ** -1)
        if torch.isnan(energy.sum()):
            energy[torch.isnan(energy)] = 0
        # print(energy)
        p = torch.sum(energy, dim=1, keepdim=True)
        # print('p:', p[:1])
        return p


class DissipatedEnergy(Module):
    def __init__(self, p_dim, h_dim=32):
        super(DissipatedEnergy, self).__init__()
        self.W = Linear(p_dim, h_dim, bias=False)

    def forward(self, p, m):
        alpha2 = 1 / (m ** 2)
        pw = self.W(p)
        a2pwwp = alpha2 * (pw ** 2)
        if torch.isnan(a2pwwp.sum()):
            a2pwwp[torch.isnan(a2pwwp)] = 0
        f = torch.sum(a2pwwp, dim=1, keepdim=True)
        # print('f:', f[:1])
        return f


class DissipativeHamiltonianDerivation(Module):
    def __init__(self, n_dim, p_dim, q_dim, use_cuda=False, dropout=0.0):
        super(DissipativeHamiltonianDerivation, self).__init__()
        self.massive = Massive(n_dim, use_cuda=use_cuda)
        self.T = KineticEnergy(p_dim)
        self.U = PotentialEnergy(q_dim, dropout=dropout, use_cuda=use_cuda)
        self.F = DissipatedEnergy(p_dim)

    def forward(self, n, p, q, e, mol_node_matrix, mol_node_mask, return_energy=False, dissipate=True):
        nnm = mol_node_matrix.t() @ mol_node_matrix
        m = self.massive(n)
        hamiltonians = self.T(p, m) + self.U(n, m, q, e, nnm)
        dissipations = self.F(p, m)
        hamilton = hamiltonians.sum()
        # print('hamilton:', hamilton)
        dissipated = dissipations.sum()
        # print('dissipate:', dissipate)
        dq = autograd.grad(hamilton, p, create_graph=True)[0]
        if dissipate:
            dp = -1 * (autograd.grad(hamilton, q, create_graph=True)[0] +
                       autograd.grad(dissipated, p, create_graph=True)[0] * m)
        else:
            dp = -1 * autograd.grad(hamilton, q, create_graph=True)[0]
        # print('dp:', dp[:1])
        # print('dq:', dq[:1])
        if return_energy:
            return dp, dq, hamiltonians, dissipations
        return dp, dq


class HamiltonianDerivation(Module):
    def __init__(self, p_dim, q_dim, h_dim=128, dropout=0.0):
        super(HamiltonianDerivation, self).__init__()
        # self.gcl = GraphConvolutionLayer(p_dim + q_dim, h_dim, h_dims=[], dropout=dropout)
        self.align_attend = AlignAttendPooling(p_dim + q_dim, h_dim, radius=1, dropout=dropout, use_gru=False)
        self.relu = ELU()
        self.linear = Linear(h_dim, 1)
        self.softplus = Softplus()

    def forward(self, p, q, e, mol_node_matrix, mol_node_mask):
        u = torch.cat([p, q], dim=1)
        # mol_features = self.gcl(u, e)
        mol_features = self.align_attend(u, mol_node_matrix, mol_node_mask)[0]
        hamiltonians = self.softplus(self.linear(self.relu(mol_features)))
        hamilton = hamiltonians.sum()
        dq = autograd.grad(hamilton, p, create_graph=True)[0]
        dp = -1 * autograd.grad(hamilton, q, create_graph=True)[0]
        return dp, dq

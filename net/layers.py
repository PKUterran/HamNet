import torch
import torch.autograd as autograd
from torch.nn import Parameter, Module, Linear, Softmax, LogSoftmax, Sigmoid, GRUCell, ELU, LeakyReLU, Dropout, Tanh, \
    LSTM, Softplus
from torch.nn.utils.rnn import pad_sequence
from itertools import chain


class ConcatMesPassing(Module):
    def __init__(self, n_dim: int, e_dim: int, c_dim: int, dropout=0.):
        super(ConcatMesPassing, self).__init__()
        self.linear = Linear(n_dim + e_dim + n_dim, c_dim, bias=True)
        self.linear_e = Linear(n_dim + e_dim + n_dim, e_dim, bias=True)
        self.relu1 = LeakyReLU()
        self.relu2 = LeakyReLU()
        self.relu_e = LeakyReLU()
        self.attention = Linear(e_dim, 1, bias=True)
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
        a = self.relu2(self.attention(new_edge_features))
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
        self.linear1 = Linear(dim, out_dim, bias=True)
        self.linear2 = Linear(dim, 1, bias=True)
        self.softmax = Softmax(dim=1)
        self.use_cuda = use_cuda
        self.dropout = Dropout(p=dropout)
        self.tanh = Tanh()

    def forward(self, node_features: torch.Tensor, mol_node_matrix: torch.Tensor, mol_node_mask: torch.Tensor) \
            -> (torch.Tensor, torch.Tensor):
        h = self.tanh(self.linear1(self.dropout(node_features)))
        a = self.linear2(self.dropout(node_features))
        d = a.view([-1]).diag()
        mol_node_weight = mol_node_matrix @ d + mol_node_mask
        mol_node_weight = self.softmax(mol_node_weight)
        return mol_node_weight @ h, mol_node_weight


class GraphConvolutionAttentivePooling(Module):
    def __init__(self, in_dim: int, out_dim: int, use_cuda=False, dropout=0.):
        super(GraphConvolutionAttentivePooling, self).__init__()
        self.gcn = GraphConvolutionLayer(in_dim, out_dim, h_dims=[out_dim], dropout=dropout)
        self.linear2 = Linear(out_dim, 1, bias=True)
        self.softmax = Softmax(dim=1)
        self.use_cuda = use_cuda
        self.dropout = Dropout(p=dropout)
        self.tanh = Tanh()

    def forward(self, node_features: torch.Tensor, e: torch.Tensor,
                mol_node_matrix: torch.Tensor, mol_node_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        h = self.tanh(self.gcn(node_features, e))
        a = self.linear2(self.dropout(h))
        d = a.view([-1]).diag()
        mol_node_weight = mol_node_matrix @ d + mol_node_mask
        mol_node_weight = self.softmax(mol_node_weight)
        return mol_node_weight @ h, mol_node_weight


class GraphConvolutionPooling(Module):
    def __init__(self, in_dim: int, out_dim: int, use_cuda=False, dropout=0.):
        super(GraphConvolutionPooling, self).__init__()
        self.gcn = GraphConvolutionLayer(in_dim, out_dim, h_dims=[], dropout=dropout)
        self.softmax = Softmax(dim=1)
        self.use_cuda = use_cuda
        self.tanh = Tanh()

    def forward(self, node_features: torch.Tensor, e: torch.Tensor,
                mol_node_matrix: torch.Tensor, mol_node_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        h = self.tanh(self.gcn(node_features, e))
        mol_node_weight = mol_node_mask
        mol_node_weight = self.softmax(mol_node_weight)
        return mol_node_weight @ h, mol_node_weight


class MapPooling(Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(MapPooling, self).__init__()
        self.linear = Linear(in_dim, out_dim, False)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.linear(node_features), dim=0)


class GraphConvolutionLayer(Module):
    def __init__(self, i_dim: int, o_dim: int, h_dims: list = list((128,)), activation=None, dropout=0.0):
        super(GraphConvolutionLayer, self).__init__()
        in_dims = [i_dim] + h_dims
        out_dims = h_dims + [o_dim]
        self.linears = [Linear(in_dim, out_dim, bias=True) for in_dim, out_dim in zip(in_dims, out_dims)]
        self.relu = LeakyReLU()
        self.activation = activation
        self.dropout = Dropout(dropout)
        for i, linear in enumerate(self.linears):
            self.register_parameter('{}_weight_{}'.format('GCN', i), linear.weight)
            self.register_parameter('{}_bias_{}'.format('GCN', i), linear.bias)

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        h = self.dropout(x)
        for i, linear in enumerate(self.linears):
            h = a @ linear(h)
            if i < len(self.linears) - 1:
                h = self.relu(h)
        if self.activation == 'tanh':
            h = torch.tanh(h)
        elif not self.activation:
            pass
        else:
            assert False, 'Undefined activation: {}.'.format(self.activation)
        return h


class LstmPQEncoder(Module):
    def __init__(self, in_dim, pq_dim, h_dim=128, num_layers=1):
        super(LstmPQEncoder, self).__init__()
        self.pq_dim = pq_dim
        self.gcl = GraphConvolutionLayer(in_dim, h_dim, h_dims=[])
        self.relu = ELU()
        self.rnn = LSTM(h_dim, 2 * pq_dim, num_layers)

    def forward(self, node_features: torch.Tensor, mol_mode_matrix: torch.Tensor, e: torch.Tensor) \
            -> (torch.Tensor, torch.Tensor):
        hidden_node_features = self.relu(self.gcl(node_features, e))
        seqs = [hidden_node_features[n == 1, :] for n in mol_mode_matrix]
        lengths = [s.shape[0] for s in seqs]
        m = pad_sequence(seqs)
        output, _ = self.rnn(m)
        ret = torch.cat([output[:lengths[i], i, :] for i in range(len(lengths))])
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


class HamiltonianDerivation(Module):
    def __init__(self, p_dim, q_dim, h_dim=128, dropout=0.0):
        super(HamiltonianDerivation, self).__init__()
        self.gcl = GraphConvolutionLayer(p_dim + q_dim, h_dim, h_dims=[], dropout=dropout)
        self.relu = ELU()
        self.linear = Linear(h_dim, 1)
        self.softplus = Softplus()
        # for param in chain(self.gcl.parameters(), self.linear.parameters()):
        #     self.register_parameter("_", param)

    def forward(self, p, q, e):
        u = torch.cat([p, q], dim=1)
        hamiltonians = self.softplus(self.linear(self.relu(self.gcl(u, e))))
        hamilton = hamiltonians.sum()
        dq = autograd.grad(hamilton, p, create_graph=True)[0]
        dp = -1 * autograd.grad(hamilton, q, create_graph=True)[0]
        return dp, dq, hamiltonians

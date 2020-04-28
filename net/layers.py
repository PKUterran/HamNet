import torch
import torch.autograd as autograd
from torch.nn import Parameter, Module, Linear, Softmax, LogSoftmax, Sigmoid, GRUCell, ELU, LeakyReLU, Dropout


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
    def __init__(self, dim: int, use_cuda=False, dropout=0.):
        super(AttentivePooling, self).__init__()
        self.linear1 = Linear(dim, dim, bias=True)
        self.linear2 = Linear(dim, 1, bias=True)
        self.softmax = Softmax(dim=1)
        self.use_cuda = use_cuda
        self.dropout = Dropout(p=dropout)

    def forward(self, node_features: torch.Tensor, mol_node_matrix: torch.Tensor, mol_node_mask: torch.Tensor)\
            -> (torch.Tensor, torch.Tensor):
        h = self.linear1(self.dropout(node_features))
        a = self.linear2(self.dropout(node_features))
        d = a.view([-1]).diag()
        mol_node_weight = mol_node_matrix @ d + mol_node_mask
        mol_node_weight = self.softmax(mol_node_weight)
        return mol_node_weight @ h, a


class MapPooling(Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(MapPooling, self).__init__()
        self.linear = Linear(in_dim, out_dim, False)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.linear(node_features), dim=0)


class DirectDerivation(Module):
    def __init__(self, p_dim, q_dim, c_dim, h_dim=128):
        super(DirectDerivation, self).__init__()
        self.p_dim = p_dim
        self.q_dim = q_dim
        self.W_dp_dq1 = Linear(p_dim + q_dim + c_dim, h_dim)
        self.relu = LeakyReLU()
        self.W_dp_dq2 = Linear(h_dim, p_dim + q_dim)

    def forward(self, p, q, c):
        u = torch.cat([p, q, c], dim=1)
        dp_dq = self.W_dp_dq2(self.relu(self.W_dp_dq1(u)))
        dp = dp_dq[:, :self.p_dim]
        dq = dp_dq[:, self.p_dim:]
        return dp, dq


class HamiltonianDerivation(Module):
    def __init__(self, p_dim, q_dim, c_dim, h_dim=128):
        super(HamiltonianDerivation, self).__init__()
        self.p_dim = p_dim
        self.q_dim = q_dim
        self.W_dp_dq1 = Linear(p_dim + q_dim + c_dim, h_dim)
        self.relu = LeakyReLU()
        self.W_dp_dq2 = Linear(h_dim, 1)

    def forward(self, p, q, c):
        u = torch.cat([p, q, c], dim=1)
        hamilton = self.W_dp_dq2(self.relu(self.W_dp_dq1(u))).sum()
        dq = autograd.grad(hamilton, p, create_graph=True)[0]
        dp = -1 * autograd.grad(hamilton, q, create_graph=True)[0]
        return dp, dq

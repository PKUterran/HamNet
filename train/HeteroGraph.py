import torch


class HeteroGraph:
    def __init__(self, node_features: list, edge_features: list, us: list, vs: list, edge_mask: list):
        '''
        :param node_features: node features
        :param edge_features: edge features
        :param us: start node index of each edge
        :param vs: end node index of each edge
        :param edge_mask: if it's an edge inside functional group
        '''
        assert len(edge_features) == len(us) and len(edge_features) == len(vs), \
            '{}, {}, {}.'.format(len(edge_features), len(us), len(vs))
        # print(node_features)
        self.node_features = torch.tensor(node_features, dtype=torch.float32)
        # print(edge_features)
        self.edge_features = torch.tensor(edge_features, dtype=torch.float32)
        self.us = us
        self.vs = vs
        self.edge_mask = edge_mask

        self.n_dim = self.node_features.shape[-1]
        self.e_dim = self.edge_features.shape[-1]

    def show(self):
        print('node features:', self.node_features)
        print('edge features:', self.edge_features)
        print('us features:', self.us)
        print('vs features:', self.vs)
        print('edge mask:', self.edge_mask)

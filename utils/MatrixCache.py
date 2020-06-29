import torch
from net.models import AMPNN


class MatrixCache:
    def __init__(self, max_dict=1000):
        self.max_dict = max_dict
        self.matrix_mask_dicts = {}

    def fetch(self, molecules: list, mask: list, nfs: torch.Tensor, name=None, use_cuda=False):
        ms = []
        us = []
        vs = []
        ptr = 0
        for i, m in enumerate(mask):
            nn = molecules[m].node_features.shape[0]
            ms.extend([i] * nn)
            for u in molecules[m].us:
                us.append(u + ptr)
            for v in molecules[m].vs:
                vs.append(v + ptr)
            ptr += nn

        if name and name in self.matrix_mask_dicts.keys():
            mm_tuple = self.matrix_mask_dicts[name]
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
            if name and len(self.matrix_mask_dicts.keys()) < self.max_dict:
                self.matrix_mask_dicts[name] = mm_tuple

        return us, vs, mm_tuple

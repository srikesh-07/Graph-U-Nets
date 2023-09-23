import random

import torch
from tqdm import tqdm
import networkx as nx
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from functools import partial


class G_data(object):
    def __init__(self, num_class, feat_dim, g_list):
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.g_list = g_list
        self.fold_idx = 1
        self.sep_data()

    def sep_data(self, seed=0):
        random.shuffle(self.g_list)
        # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        # labels = [g.label for g in self.g_list]
        # self.idx_list = list(skf.split(np.zeros(len(labels)), labels))

    def use_fold_data(self, fold_idx):
        total_graphs = len(self.g_list)
        self.train_gs = self.g_list[: int(total_graphs * 0.7)]
        self.val_gs = self.g_list[len(self.train_gs): len(self.train_gs) + int(total_graphs * 0.1)]
        self.test_gs = self.g_list[len(self.train_gs) + int(total_graphs * 0.1):]


class FileLoader(object):
    def __init__(self, args):
        self.args = args

    def line_genor(self, lines):
        for line in lines:
            yield line

    def gen_graph(self, f, i, label_dict, feat_dict, deg_as_tag):
        row = next(f).strip().split()
        n, label = [int(w) for w in row]
        if label not in label_dict:
            label_dict[label] = len(label_dict)
        g = nx.Graph()
        g.add_nodes_from(list(range(n)))
        node_tags = []
        for j in range(n):
            row = next(f).strip().split()
            tmp = int(row[1]) + 2
            row = [int(w) for w in row[:tmp]]
            if row[0] not in feat_dict:
                feat_dict[row[0]] = len(feat_dict)
            for k in range(2, len(row)):
                if j != row[k]:
                    g.add_edge(j, row[k])
            if len(row) > 2:
                node_tags.append(feat_dict[row[0]])
        g.label = label
        g.remove_nodes_from(list(nx.isolates(g)))
        if deg_as_tag:
            g.node_tags = list(dict(g.degree).values())
        else:
            g.node_tags = node_tags
        return g

    def process_g(self, label_dict, tag2index, tagset, g):
        g.label = label_dict[g.label]
        g.feas = torch.tensor([tag2index[tag] for tag in g.node_tags])
        g.feas = F.one_hot(g.feas, len(tagset))
        A = torch.FloatTensor(nx.to_numpy_matrix(g))
        g.A = A + torch.eye(g.number_of_nodes())
        return g

    def load_data(self):
        args = self.args
        print('loading data ...')
        g_list = []
        label_dict = {}
        feat_dict = {}

        if args.data == "PROTEINS":
            K = [0, 371, 742, 1113]
        elif args.data == "PTC":
            K = [0, 115, 230, 344]
        elif args.data == "IMDBBINARY":
            K = [0, 333, 666, 1000]
        elif args.data == "DD":
            K = [0, 393, 785, 1178]
        elif args.data == "FRANK":
            K = [0, 1445, 2890, 4337]

        with open('data/%s/%s.txt' % (args.data, args.data), 'r') as f:
            lines = f.readlines()
        f = self.line_genor(lines)
        n_g = int(next(f).strip())
        nodes = torch.zeros(n_g)
        for i in tqdm(range(n_g), desc="Create graph", unit='graphs'):
            g = self.gen_graph(f, i, label_dict, feat_dict, args.deg_as_tag)
            nodes[i] = g.number_of_nodes()
            g_list.append(g)

        _, ind = torch.sort(nodes, descending=True)

        for i in ind[K[0]:K[1]]:
            g_list[i].nodegroup = 2
        for i in ind[K[1]:K[2]]:
            g_list[i].nodegroup = 1
        for i in ind[K[2]:K[3]]:
            g_list[i].nodegroup = 0

        tagset = set([])
        for g in g_list:
            tagset = tagset.union(set(g.node_tags))
        tagset = list(tagset)
        tag2index = {tagset[i]: i for i in range(len(tagset))}

        f_n = partial(self.process_g, label_dict, tag2index, tagset)
        new_g_list = []
        for g in tqdm(g_list, desc="Process graph", unit='graphs'):
            new_g_list.append(f_n(g))
        num_class = len(label_dict)
        feat_dim = len(tagset)

        print('# classes: %d' % num_class, '# maximum node tag: %d' % feat_dim)
        return G_data(num_class, feat_dim, new_g_list)

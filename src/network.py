import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ops import GCN, GraphUnet, Initializer, norm_g


class GNet(nn.Module):
    def __init__(self, in_dim, n_classes, args):
        super(GNet, self).__init__()
        self.n_act = getattr(nn, args.act_n)()
        self.c_act = getattr(nn, args.act_c)()
        self.s_gcn = GCN(in_dim, args.l_dim, self.n_act, args.drop_n)
        self.g_unet = GraphUnet(
            args.ks, args.l_dim, args.l_dim, args.l_dim, self.n_act,
            args.drop_n)
        self.out_l_1 = nn.Linear(3*args.l_dim*(args.l_num+1), args.h_dim)
        self.out_l_2 = nn.Linear(args.h_dim, n_classes)
        self.out_drop = nn.Dropout(p=args.drop_c)
        Initializer.weights_init(self)

    def forward(self, gs, hs, labels, nodegroup):
        hs = self.embed(gs, hs)
        logits = self.classify(hs)
        return self.metric(logits, labels, nodegroup)

    def embed(self, gs, hs):
        o_hs = []
        for g, h in zip(gs, hs):
            h = self.embed_one(g, h)
            o_hs.append(h)
        hs = torch.stack(o_hs, 0)
        return hs

    def embed_one(self, g, h):
        g = norm_g(g)
        h = self.s_gcn(g, h)
        hs = self.g_unet(g, h)
        h = self.readout(hs)
        return h

    def readout(self, hs):
        h_max = [torch.max(h, 0)[0] for h in hs]
        h_sum = [torch.sum(h, 0) for h in hs]
        h_mean = [torch.mean(h, 0) for h in hs]
        h = torch.cat(h_max + h_sum + h_mean)
        return h

    def classify(self, h):
        h = self.out_drop(h)
        h = self.out_l_1(h)
        h = self.c_act(h)
        h = self.out_drop(h)
        h = self.out_l_2(h)
        return F.log_softmax(h, dim=1)

    def metric(self, logits, labels, nodegroup):
        loss = F.nll_loss(logits, labels)
        _, preds = torch.max(logits, 1)
        correct_all = preds.eq(labels.view_as(
            preds)).sum().cpu().item()
        nodegroup = torch.tensor(nodegroup, dtype=torch.long)
        mask_head = (nodegroup == 2)
        mask_medium = (nodegroup == 1)
        mask_tail = (nodegroup == 0)
        correct_head = preds[mask_head].eq(labels[mask_head].view_as(
            preds[mask_head])).sum().cpu().item()
        correct_medium = preds[mask_medium].eq(labels[mask_medium].view_as(
            preds[mask_medium])).sum().cpu().item()
        correct_tail = preds[mask_tail].eq(labels[mask_tail].view_as(
            preds[mask_tail])).sum().cpu().item()
        return loss, [(correct_all, labels.shape[0]) ,
                      (correct_head, mask_head.sum().cpu().item()),
                      (correct_medium, mask_medium.sum().cpu().item()),
                      (correct_tail, mask_tail.sum().cpu().item()),
                      ]

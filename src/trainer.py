import torch
from tqdm import tqdm
import torch.optim as optim
from utils.dataset import GraphData


class Trainer:
    def __init__(self, args, net, G_data):
        self.args = args
        self.net = net
        self.feat_dim = G_data.feat_dim
        self.fold_idx = G_data.fold_idx
        self.init(args, G_data.train_gs, G_data.val_gs, G_data.test_gs)
        if torch.cuda.is_available():
            self.net.cuda()

    def init(self, args, train_gs, val_gs, test_gs):
        print('#train: %d, #test: %d' % (len(train_gs), len(test_gs)))
        train_data = GraphData(train_gs, self.feat_dim)
        val_data = GraphData(val_gs, self.feat_dim)
        test_data = GraphData(test_gs, self.feat_dim)
        self.train_d = train_data.loader(self.args.batch, True)
        self.val_d = val_data.loader(self.args.batch, True)
        self.test_d = test_data.loader(self.args.batch, False)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.args.lr, amsgrad=True,
            weight_decay=0.0008)

    def to_cuda(self, gs):
        if torch.cuda.is_available():
            if type(gs) == list:
                return [g.cuda() for g in gs]
            return gs.cuda()
        return gs

    def run_epoch(self, epoch, data, model, optimizer):
        losses, accs, n_samples = [], [], 0
        head_samples, med_samples, tail_samples = 0, 0, 0
        head_accs, med_accs, tail_accs = [], [], []
        for batch in tqdm(data, desc=str(epoch), unit='b'):
            cur_len, gs, hs, ys, nodegroup = batch
            gs, hs, ys = map(self.to_cuda, [gs, hs, ys])
            loss, acc = model(gs, hs, ys, nodegroup)
            losses.append(loss * cur_len)
            accs.append(acc[0][0])
            n_samples += acc[0][1]
            head_accs.append(acc[1][0])
            head_samples += acc[1][1]
            med_accs.append(acc[2][0])
            med_samples += acc[2][1]
            tail_accs.append(acc[3][0])
            tail_samples += acc[3][1]
            # accs.append(acc[0] * cur_len)
            # head_accs.append(acc[1] * cur_len)
            # med_accs.append(acc[2] * cur_len)
            # tail_accs.append(acc[3] * cur_len)
            # accs.append(acc*cur_len)
            n_samples += cur_len
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        avg_loss, avg_acc = sum(losses) / n_samples, sum(accs) / n_samples
        avg_head, avg_med, avg_tail = sum(head_accs) / head_samples, sum(med_accs) / med_samples, sum(tail_accs) / tail_samples
        return avg_loss.item(), round(avg_acc, 5), round(avg_head, 5), round(avg_med, 5), round(avg_tail, 5)

    def train(self):
        max_acc = 0.0
        train_str = 'Train epoch %d: loss %.5f acc %.5f head_acc %.5f med_acc %.5f tail_acc %.5f'
        val_str = 'Val epoch %d: loss %.5f acc %.5f max %.5f head_acc %.5f med_acc %.5f tail_acc %.5f'
        test_str = 'Test epoch %d: loss %.5f acc %.5f max %.5f head_acc %.5f med_acc %.5f tail_acc %.5f'
        line_str = '%d:\t%.5f\n'
        for e_id in range(self.args.num_epochs):
            self.net.train()
            loss, acc, head_acc, med_acc, tail_acc = self.run_epoch(
                e_id, self.train_d, self.net, self.optimizer)
            print(train_str % (e_id, loss, acc, head_acc, med_acc, tail_acc))

            with torch.no_grad():
                self.net.eval()
                loss, acc, head_acc, med_acc, tail_acc = self.run_epoch(e_id, self.val_d, self.net, None)
                print(val_str % (e_id, loss, acc, max_acc, head_acc, med_acc, tail_acc))
                if acc > max_acc:
                    max_acc = acc
                    best_loss, best_acc, best_head_acc, best_med_acc, best_tail_acc = self.run_epoch(e_id, self.test_d, self.net, None)
                    print(test_str % (e_id, best_loss, best_acc, max_acc, best_head_acc, best_med_acc, best_tail_acc))

        return max_acc, best_acc, best_head_acc, best_med_acc, best_tail_acc



        # with open(self.args.acc_file, 'a+') as f:
        #     f.write(line_str % (self.fold_idx, max_acc))

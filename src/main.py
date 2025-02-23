import argparse
import random
import time
import torch
import numpy as np
from network import GNet
from trainer import Trainer
from utils.data_loader import FileLoader


def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-seed', type=int, default=1, help='seed')
    parser.add_argument('-data', default='DD', help='data folder name')
    parser.add_argument('-fold', type=int, default=1, help='fold (1..10)')
    parser.add_argument('-num_epochs', type=int, default=2, help='epochs')
    parser.add_argument('-batch', type=int, default=2, help='batch size')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-deg_as_tag', type=int, default=0, help='1 or degree')
    parser.add_argument('-l_num', type=int, default=3, help='layer num')
    parser.add_argument('-h_dim', type=int, default=512, help='hidden dim')
    parser.add_argument('-l_dim', type=int, default=48, help='layer dim')
    parser.add_argument('-drop_n', type=float, default=0.3, help='drop net')
    parser.add_argument('-drop_c', type=float, default=0.2, help='drop output')
    parser.add_argument('-act_n', type=str, default='ELU', help='network act')
    parser.add_argument('-act_c', type=str, default='ELU', help='output act')
    parser.add_argument('-ks', nargs='+', type=float, default='0.9 0.8 0.7')
    parser.add_argument('-acc_file', type=str, default='re', help='acc file')
    args, _ = parser.parse_known_args()
    return args


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def app_run(args, G_data, fold_idx):
    G_data.use_fold_data(fold_idx)
    net = GNet(G_data.feat_dim, G_data.num_class, args)
    trainer = Trainer(args, net, G_data)
    metrics = trainer.train()
    return metrics


def main():
    args = get_args()
    print(args)
    seeds = range(0, 5)
    test_record = torch.zeros(len(seeds))
    valid_record = torch.zeros(len(seeds))
    tail_record = torch.zeros(len(seeds))
    medium_record = torch.zeros(len(seeds))
    head_record = torch.zeros(len(seeds))
    for seed in seeds:
        print("Training with Seed - ", seed)
        set_random(seed)
        start = time.time()
        G_data = FileLoader(args).load_data()
        print('load data using ------>', time.time()-start)
        # if args.fold == 0:
        #     for fold_idx in range(10):
        #         print('start training ------> fold', fold_idx+1)
        #         app_run(args, G_data, fold_idx)
        # else:
        print('start training ------> fold', args.fold)
        metrics = app_run(args, G_data, args.fold-1)
        valid_record[seed] = metrics[0]
        test_record[seed] = metrics[1]
        head_record[seed] = metrics[2]
        medium_record[seed] = metrics[3]
        tail_record[seed] = metrics[4]

    print('Valid mean: %.4f, std: %.4f' %
          (valid_record.mean().item(), valid_record.std().item()))
    print('Test mean: %.4f, std: %.4f' %
          (test_record.mean().item(), test_record.std().item()))
    print('Head mean: %.4f, std: %.4f' %
          (head_record.mean().item(), head_record.std().item()))
    print('Medium mean: %.4f, std: %.4f' %
          (medium_record.mean().item(), medium_record.std().item()))
    print('Tail mean: %.4f, std: %.4f' %
          (tail_record.mean().item(), tail_record.std().item()))

    with open("metrics.txt", "a") as txt_file:
        txt_file.write(f"Dataset: {args.data}, \n"
                       f"Valid Mean: {round(valid_record.mean().item(), 4)}, \n"
                       f"Std Valid Mean: {round(valid_record.std().item(), 4)}, \n"
                       f"Test Mean: {round(test_record.mean().item(), 4)}, \n"
                       f"Std Test Mean: {round(test_record.std().item(), 4)}, \n"
                       f"Head Mean: {round(head_record.mean().item(), 4)}, \n"
                       f"Std Head Mean: {round(head_record.std().item(), 4)}, \n"
                       f"Medium Mean: {round(medium_record.mean().item(), 4)}, \n"
                       f"Std Medium Mean: {round(medium_record.std().item(), 4)}, \n"
                       f"Tail Mean: {round(tail_record.mean().item(), 4)}, \n"
                       f"Std Tail Mean: {round(tail_record.std().item(), 4)} \n\n"
        )


if __name__ == "__main__":
    main()

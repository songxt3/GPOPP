import logging
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from GCNpredictor.dataset import Nb101Dataset
from GCNpredictor.model import NeuralPredictor
from GCNpredictor.utils import AverageMeter, AverageMeterGroup, get_logger, reset_seed, to_cuda

from scipy.stats import kendalltau


def accuracy_mse(predict, target, scale=100.):
    predict = Nb101Dataset.denormalize(predict.detach()) * scale
    target = Nb101Dataset.denormalize(target) * scale
    return F.mse_loss(predict, target)


def visualize_scatterplot(predict, target, scale=100.):
    def _scatter(x, y, subplot, threshold=None):
        plt.subplot(subplot)
        plt.grid(linestyle="--")
        plt.xlabel("Validation Accuracy")
        plt.ylabel("Prediction")
        plt.scatter(target, predict, s=1)
        if threshold:
            ax = plt.gca()
            ax.set_xlim(threshold, 95)
            ax.set_ylim(threshold, 95)

    predict = Nb101Dataset.denormalize(predict) * scale
    target = Nb101Dataset.denormalize(target) * scale
    plt.figure(figsize=(12, 6))
    _scatter(predict, target, 121)
    _scatter(predict, target, 122, threshold=90)
    plt.savefig("assets/scatterplot.png", bbox_inches="tight")
    plt.close()


def main():
    valid_splits = ["172", "334", "860", "91-172", "91-334", "91-860", "denoise-91", "denoise-80", "all"]
    parser = ArgumentParser()
    parser.add_argument("--train_split", choices=valid_splits, default="334")
    parser.add_argument("--eval_split", choices=valid_splits, default="all")
    parser.add_argument("--gcn_hidden", type=int, default=144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_batch_size", default=10, type=int)
    parser.add_argument("--eval_batch_size", default=1000, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("--wd", "--weight_decay", default=1e-3, type=float)
    parser.add_argument("--train_print_freq", default=None, type=int)
    parser.add_argument("--eval_print_freq", default=10, type=int)
    parser.add_argument("--visualize", default=False, action="store_true")
    args = parser.parse_args()

    reset_seed(args.seed)

    dataset = Nb101Dataset(split=args.train_split)
    dataset_test = Nb101Dataset(split=args.eval_split)
    data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(dataset_test, batch_size=args.eval_batch_size)
    net = NeuralPredictor(gcn_hidden=args.gcn_hidden)
    net.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    logger = get_logger()

    max_acc = 0
    min_acc = 1
    for step, batch in enumerate(data_loader):
        # record the max acc and the min acc
        max_target = batch["val_acc"].cpu().max().item()
        min_target = batch["val_acc"].cpu().min().item()
        max_target = Nb101Dataset.denormalize(max_target)
        min_target = Nb101Dataset.denormalize(min_target)
        if max_target > max_acc:
            max_acc = max_target
        if min_target < min_acc:
            min_acc = min_target

    net.train()
    for epoch in range(args.epochs):
        meters = AverageMeterGroup()
        lr = optimizer.param_groups[0]["lr"]
        for step, batch in enumerate(data_loader):
            batch = to_cuda(batch)
            target = batch["val_acc"]
            predict = net(batch)
            optimizer.zero_grad()
            loss = criterion(predict, target)
            loss.backward()
            optimizer.step()
            mse = accuracy_mse(predict, target)
            meters.update({"loss": loss.item(), "mse": mse.item()}, n=target.size(0))
            if (args.train_print_freq and step % args.train_print_freq == 0) or \
                    step + 1 == len(data_loader):
                logger.info("Epoch [%d/%d] Step [%d/%d] lr = %.3e  %s",
                            epoch + 1, args.epochs, step + 1, len(data_loader), lr, meters)
        lr_scheduler.step()

    net.eval()
    meters = AverageMeterGroup()
    predict_, target_, idx_ = [], [], []
    with torch.no_grad():
        for step, batch in enumerate(test_data_loader):
            batch = to_cuda(batch)
            target = batch["val_acc"]
            predict = net(batch)
            predict_.append(predict.cpu().numpy())
            target_.append(target.cpu().numpy())
            meters.update({"loss": criterion(predict, target).item(),
                           "mse": accuracy_mse(predict, target).item()}, n=target.size(0))
            pre_idx = [idx + step * args.eval_batch_size for idx, pre_acc in enumerate(predict) if
                       Nb101Dataset.denormalize(pre_acc) > max_acc or Nb101Dataset.denormalize(pre_acc) < min_acc]
            idx_.extend(pre_idx)

            if (args.eval_print_freq and step % args.eval_print_freq == 0) or \
                    step % 10 == 0 or step + 1 == len(test_data_loader):
                logger.info("Evaluation Step [%d/%d]  %s", step + 1, len(test_data_loader), meters)
    predict_ = np.concatenate(predict_)
    target_ = np.concatenate(target_)
    logger.info("Kendalltau: %.6f", kendalltau(np.argsort(np.argsort(predict_)), np.argsort(np.argsort(target_)))[0])
    logger.info("-----------------------------------------------------------")
    logger.info('max_acc: {}, min_acc: {}'.format(max_acc, min_acc))
    logger.info("Acc out of range: {}".format(Nb101Dataset.denormalize(predict_[idx_])))
    logger.info("Acc out of range (ground truth): {}".format(Nb101Dataset.denormalize(target_[idx_])))
    logger.info("Number of Acc out of range: %d", len(idx_))
    logger.info("Kendalltau for acc out of range(min_acc, max_acc): %.6f",
                kendalltau(np.argsort(np.argsort(predict_[idx_])), np.argsort(np.argsort(target_[idx_])))[0])
    logger.info("MSE for acc out of range: %.6f",
                accuracy_mse(torch.from_numpy(predict_[idx_]), torch.from_numpy(target_[idx_])).item())
    in_range_idx_ = list(set(np.arange(len(test_data_loader.dataset))) - set(idx_))
    logger.info("Number of Acc in range: %d", len(in_range_idx_))
    logger.info("Kendalltau for acc in range(min_acc, max_acc): %.6f",
                kendalltau(np.argsort(np.argsort(predict_[in_range_idx_])),
                           np.argsort(np.argsort(target_[in_range_idx_])))[0])
    logger.info("MSE for acc in range: %.6f",
                accuracy_mse(torch.from_numpy(predict_[in_range_idx_]),
                             torch.from_numpy(target_[in_range_idx_])).item())

    if args.visualize:
        visualize_scatterplot(predict_, target_)

if __name__ == "__main__":
    main()

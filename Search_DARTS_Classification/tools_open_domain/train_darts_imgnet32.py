import argparse
import os
import time
import glob
import sys
import logging
import torch
import torch.nn as nn
import numpy as np
import copy
import random
import torch.backends.cudnn as cudnn

sys.path.append(os.getcwd())
from ptflops import get_model_complexity_info
from nas_lib.models_darts.datrs_neuralnet import DartsCifar10NeuralNet
from nas_lib.utils.utils_darts import create_exp_dir, AvgrageMeter, top_accuracy, \
    save_checkpoint, load_model_from_last_population
from nas_lib.models_darts import genotypes
from nas_lib.data.Imagenet32 import Imagenet32
import torchvision.transforms as transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(train_queue, model, criterion, optimizer, device):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits, logits_aux = model(input, device)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1, prec5 = top_accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        if step % args.report_freq == 0 and step != 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, device):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
        input = input.to(device)
        target = target.to(device)

        logits, _ = model(input, device)
        loss = criterion(logits, target)
        prec1, prec5 = top_accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0 and step != 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Searched model for imagenet')
    parser.add_argument('--gen_no', type=str, default=15, help='name of output files')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=40, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=40, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=321654, help='random seed')
    parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
    parser.add_argument('--save_dir', type=str,
                        help='name of save directory')
    args = parser.parse_args()
    seed = args.seed

    # genotype, hash_key = load_model_from_last_population(args.save_dir, args.gen_no)
    genotype = eval("genotypes.GPOPP") # The architecture searched by GPOPP

    args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m/%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    CLASSES = 1000

    # set seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DartsCifar10NeuralNet(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype,
                                  1, stem_mult=3)
    model.drop_path_prob = 0
    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
        logging.info('{:<30}  {:<8}'.format('Computational complexity: ', flops))
        logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    transform_train = transforms.Compose(
            [transforms.ToTensor()])
    transform_test = transforms.Compose(
            [transforms.ToTensor()])
    train_data = Imagenet32(root='', train=True, transform=transform_train) # your path of ImageNet32
    test_data = Imagenet32(root='', train=False, transform=transform_test)
    model_train_data = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=32, pin_memory=True)
    model_test_data = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=32, pin_memory=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0, -1)

    best_acc_top1 = 0
    best_model_wts = None
    for epoch in range(args.epochs):
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(model_train_data, model, criterion, optimizer, device)
        logging.info('train_acc %f', train_acc)

        valid_acc_top1, valid_acc_top5, valid_obj = infer(model_test_data, model, criterion, device)
        logging.info('test_acc_top1 %f', valid_acc_top1)
        logging.info('test_acc_top5 %f', valid_acc_top5)
        scheduler.step(epoch=epoch)
        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
            best_model_wts = copy.deepcopy(model.state_dict())

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc_top1': best_acc_top1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save)

    model.load_state_dict(best_model_wts)
    test_acc_top1, test_acc_top5, _ = infer(model_test_data, model, criterion, device)
    logging.info('test_acc_top1 %f', test_acc_top1)
    logging.info('test_acc_top5 %f', test_acc_top5)

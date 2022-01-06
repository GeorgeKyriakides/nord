import glob
import logging
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as dset
from nord.data_curators.images import get_fashion_mnist
from nord.neural_nets.differentiable.genotypes import DARTS
from nord.neural_nets.differentiable.neural_builder_gradient import DARTSNet
from torch.autograd import Variable

from . import utils
from .architect import Architect

CIFAR_CLASSES = 10


class CLR:
    def __init__(self, train_dataloader, base_lr=1e-5, max_lr=100):
        # The lower boundary for learning rate (initial lr)
        self.base_lr = base_lr
        self.max_lr = max_lr  # The upper boundary for learning rate
        # Total number of iterations used for this test run (lr is calculated based on this)
        self.bn = len(train_dataloader) - 1
        ratio = self.max_lr / self.base_lr  # n
        self.mult = ratio ** (1 / self.bn)  # q = (max_lr/init_lr)^(1/n)
        self.best_loss = 1e9  # our assumed best loss
        self.iteration = 0  # current iteration, initialized to 0
        self.lrs = []
        self.losses = []

    def calc_lr(self, loss):
        self.iteration += 1
        # stopping criteria (if current loss > 4*best loss)
        if math.isnan(loss) or loss > 4 * self.best_loss:
            return -1
        # if current_loss < best_loss, replace best_loss with current_loss
        if loss < self.best_loss and self.iteration > 1:
            self.best_loss = loss
        mult = self.mult ** self.iteration  # q = q^i
        lr = self.base_lr * mult  # lr_i = init_lr * q
        self.lrs.append(lr)  # append the learing rate to lrs
        self.losses.append(loss)  # append the loss to losses
        return lr

    def plot(self, start=10, end=-5):  # plot lrs vs losses
        plt.xlabel("Learning Rate")
        plt.ylabel("Losses")
        plt.plot(self.lrs[start:end], self.losses[start:end])
        plt.xscale("log")  # learning rates are in log scale

    def export(self):
        with open("lr_exports.csv", "w") as f:
            f.write("LR, LOSS\n")
            for i in range(len(self.lrs)):
                f.write(str(self.lrs[i] + ", " + str(self.losses[i] + "\n")))


def main(parser):

    args = parser.parse_args()

    def lr_check(train_queue, valid_queue, model, architect, criterion, optimizer, clr):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        running_loss = 0.0
        avg_beta = 0.98  # useful in calculating smoothed loss

        for step, (input, target) in enumerate(train_queue):
            lr = 1e-4
            model.train()
            n = input.size(0)

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda()

            # get a random minibatch from the search queue with replacement
            input_search, target_search = next(iter(valid_queue))
            input_search = Variable(input_search, requires_grad=False).cuda()
            target_search = Variable(target_search, requires_grad=False).cuda()

            architect.step(
                input,
                target,
                input_search,
                target_search,
                lr,
                optimizer,
                unrolled=args.unrolled,
            )

            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)

            # calculate the smoothed loss
            running_loss = (
                avg_beta * running_loss + (1 - avg_beta) * loss
            )  # the running loss
            # smoothening effect of the loss
            smoothed_loss = running_loss / (1 - avg_beta ** (step + 1))

            # calculate learning rate using CLR
            lr = clr.calc_lr(smoothed_loss)
            if lr == -1:  # the stopping criteria
                break
            for pg in optimizer.param_groups:  # update learning rate
                pg["lr"] = lr

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info("train %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg

    def infer(valid_queue, model, criterion):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()

        with torch.no_grad():
            for step, (input, target) in enumerate(valid_queue):
                input = Variable(input).cuda()
                target = Variable(target).cuda()

                logits = model(input)
                loss = criterion(logits, target)

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                if step % args.report_freq == 0:
                    logging.info(
                        "valid %03d %e %f %f", step, objs.avg, top1.avg, top5.avg
                    )

            return top1.avg, objs.avg

    args.save = "search-{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py"))

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("gpu device = %d" % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = DARTSNet(
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, data_channels=1
    )
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # train_transform, valid_transform = utils._data_transforms_cifar10(args)
    # train_data = dset.CIFAR10(
    #     root=args.data, train=True, download=True, transform=train_transform)

    trainloader, testloader, _ = get_fashion_mnist()
    train_data = trainloader.dataset

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=0,
    )
    # pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=0,
    )
    # pin_memory=True, num_workers=2)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
    #                                                 max_lr=args.learning_rate*20, steps_per_epoch=len(train_queue)+int(args.epochs), epochs=int(args.epochs))

    architect = Architect(model, args)

    clr = CLR(train_queue)

    logging.info("epoch %d lr %e", 0)

    genotype = model.genotype()
    logging.info("genotype = %s", genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = lr_check(
        train_queue, valid_queue, model, architect, criterion, optimizer, clr
    )
    logging.info("train_acc %f", train_acc)

    clr.export()
    clr.plot()
    plt.show()

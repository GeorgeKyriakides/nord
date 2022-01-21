import torch.multiprocessing as mp
import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as dset
from nord.data_curators.images import get_fashion_mnist
from nord.neural_nets.differentiable.genotypes import DARTS
from nord.neural_nets.differentiable.neural_builder_gradient import DistributedDARTSNet
from torch.autograd import Variable

from . import utils
from .architect import Architect


CIFAR_CLASSES = 10


def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    # initialize the process group
    dist.init_process_group(args.backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_model_params(model):
    return [p for name, p in model.named_parameters() if 'arch' not in name]


def get_arch_params(model):
    return [p for name, p in model.named_parameters() if 'arch' in name]


def main(local_rank, args, world_size):
    rank = args.rank
    if local_rank > 0:
        rank = rank+local_rank

    # Important to setup the world and assign gpus here for nccl
    setup(rank, world_size, args)
    my_gpu = rank % torch.cuda.device_count()

    np.random.seed(args.seed)
    torch.cuda.set_device(my_gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        print('*'*30, 'TRAIN_QUEUE: ', len(train_queue))
        for step, (input, target) in enumerate(train_queue):
            model.train()
            n = input.size(0)

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda()

            # get a random minibatch from the search queue with replacement
            input_search, target_search = next(iter(valid_queue))
            input_search = Variable(input_search, requires_grad=False).cuda()
            target_search = Variable(
                target_search, requires_grad=False).cuda()

            # if rank == 0:
            #     architect.step(input, target, input_search, target_search,
            #                    lr, optimizer, unrolled=args.unrolled)
            # dist.broadcast_object_list(model.state_dict(), src=0)

            architect.step(input, target, input_search, target_search,
                           lr, optimizer, unrolled=args.unrolled)

            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm(get_model_params(model), args.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('train %03d %e %f %f', step,
                             objs.avg, top1.avg, top5.avg)

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
                    logging.info('valid %03d %e %f %f', step,
                                 objs.avg, top1.avg, top5.avg)

            return top1.avg, objs.avg

    print('MY RANK ', rank, ' WORLD SIZE ', world_size)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    start_time = torch.Tensor([time.time()]).cuda()
    dist.broadcast(start_time, 0)
    start_time = start_time.cpu()
    start_time = start_time.item()
    args.save = 'search-{}-{}'.format(args.save,
                                      time.strftime("%Y%m%d-%H%M%S", time.gmtime(start_time)))
    if local_rank == 0:
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    dist.barrier()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info('gpu device = %d' % my_gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = DistributedDARTSNet(args.init_channels, CIFAR_CLASSES,
                                args.layers, criterion, data_channels=1)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        get_model_params(model),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    trainloader, testloader, _ = get_fashion_mnist()
    train_data = trainloader.dataset

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    np.random.shuffle(indices)

    validation_data = torch.utils.data.Subset(
        train_data, indices[split:num_train])
    train_data = torch.utils.data.Subset(train_data, indices[:split])

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data,
        num_replicas=args.world_size,
        rank=rank
    )

    validation_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data,
        num_replicas=args.world_size,
        rank=rank
    )

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=validation_sampler,
        pin_memory=True,
        num_workers=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.arch_alphas_normal, dim=-1))
        print(F.softmax(model.arch_alphas_reduce, dim=-1))

        # training
        # train_acc, train_obj = train(
        #     train_queue, valid_queue, model, architect, criterion, optimizer, lr)
        # logging.info('train_acc %f', train_acc)

        reduce_parameters(model)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def reduce_parameters(model):
    with torch.no_grad():
        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param, op=dist.ReduceOp.SUM, async_op=True)
        for arch_p in model.arch_parameters():
            dist.all_reduce(arch_p, op=dist.ReduceOp.SUM, async_op=True)

        for param in model.parameters():
            param.data /= size
        for arch_p in model.arch_parameters():
            arch_p.data /= size

    dist.barrier()


def run_demo(demo_fn, args, world_size, procs_number):
    mp.spawn(demo_fn,
             args=(args, world_size,),
             nprocs=procs_number,
             join=True)


def mymain(parser):
    args = parser.parse_args()
    run_demo(main, args, args.world_size, torch.cuda.device_count())

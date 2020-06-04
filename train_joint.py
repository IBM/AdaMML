import os
import shutil
import time
import numpy as np
import sys
import warnings
import platform

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.optim import lr_scheduler

from models import build_model
from utils.utils import (train, validate, build_dataflow, get_augmentor,
                         save_checkpoint, accuracy, actnet_acc)
from utils.video_dataset import MultiVideoDataSet
from utils.dataset_config import get_dataset_config
from opts import arg_parser


warnings.filterwarnings("ignore", category=UserWarning)


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()

    if args.hostfile != '':
        curr_node_name = platform.node().split(".")[0]
        with open(args.hostfile) as f:
            nodes = [x.strip() for x in f.readlines() if x.strip() != '']
            master_node = nodes[0].split(" ")[0]
        for idx, node in enumerate(nodes):
            if curr_node_name in node:
                args.rank = idx
                break
        args.world_size = len(nodes)
        args.dist_url = "tcp://{}:10598".format(master_node)


    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = args.cudnn_benchmark
    args.gpu = gpu

    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file, multilabel = get_dataset_config(args.dataset)
    args.num_classes = num_classes

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    args.input_channels = []
    for modality in args.modality:
        if modality == 'rgb':
            args.input_channels.append(3)
        elif modality == 'flow':
            args.input_channels.append(2 * 5)
        elif modality == 'rgbdiff':
            args.input_channels.append(3 * 5)
        elif modality == 'sound':
            args.input_channels.append(1)

    model, arch_name = build_model(args)

    mean = [model.mean(x) for x in args.modality]
    std = [model.std(x) for x in args.modality]
    model = model.cuda(args.gpu)
    model.eval()

    if args.rank == 0:
        torch.cuda.empty_cache()

    if args.show_model and args.rank == 0:
        print(model)
        return 0

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # the batch size should be divided by number of nodes as well
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int(args.workers / ngpus_per_node)

            if args.sync_bn:
                process_group = torch.distributed.new_group(list(range(args.world_size)))
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        # assign rank to 0
        model = torch.nn.DataParallel(model).cuda()
        args.rank = 0

    if args.pretrained is not None:
        if args.rank == 0:
            print("=> using pre-trained model '{}'".format(arch_name))
        if args.gpu is None:
            checkpoint = torch.load(args.pretrained, map_location='cpu')
        else:
            checkpoint = torch.load(args.pretrained, map_location='cuda:{}'.format(args.gpu))
        new_dict = checkpoint['state_dict']
        model.load_state_dict(new_dict, strict=False)
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        if args.rank == 0:
            print("=> creating model '{}'".format(arch_name))

    # define loss function (criterion) and optimizer
    train_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    val_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    eval_criterion = accuracy

    # Data loading code
    # val_list = os.path.join(args.datadir, val_list_name)
    val_augmentors = []
    for idx, modality in enumerate(args.modality):
        val_augmentor = get_augmentor(False, args.input_size, scale_range=args.scale_range,
                                      mean=mean[idx], std=std[idx], disable_scaleup=args.disable_scaleup,
                                      threed_data=args.threed_data,
                                      modality=args.modality[idx],
                                      version=args.augmentor_ver)
        val_augmentors.append(val_augmentor)
    video_data_cls = MultiVideoDataSet
    val_dataset = video_data_cls(args.datadir, val_list_name, args.groups * args.num_segments, args.frames_per_group,
                                 num_clips=args.num_clips,
                                 num_classes=args.num_classes,
                                 modality=args.modality, image_tmpl=image_tmpl,
                                 dense_sampling=args.dense_sampling,
                                 transform=val_augmentors, is_train=False, test_mode=False,
                                 seperator=filename_seperator, filter_video=filter_video,
                                 fps=args.fps, audio_length=args.audio_length * args.num_segments,
                                 resampling_rate=args.resampling_rate)

    val_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                workers=args.workers,
                                is_distributed=args.distributed)

    log_folder = os.path.join(args.logdir, arch_name)
    if args.rank == 0:
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

    if args.evaluate:
        val_top1, val_top5, val_losses, val_speed = validate(val_loader, model, val_criterion, gpu_id=args.gpu)
        if args.rank == 0:
            logfile = open(os.path.join(log_folder, 'evaluate_log.log'), 'a')
            # flops, params = extract_total_flops_params(model_summary)
            flops, params = 'N/A', 'N/A'
            print(
                'Val@{}: \tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch\tFlops: {}\tParams: {}'.format(
                    args.input_size, val_losses, val_top1, val_top5, val_speed * 1000.0, flops, params), flush=True)
            print(
                'Val@{}: \tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch\tFlops: {}\tParams: {}'.format(
                    args.input_size, val_losses, val_top1, val_top5, val_speed * 1000.0, flops, params), flush=True,
                file=logfile)
        return

    train_augmentors = []
    for idx, modality in enumerate(args.modality):
        train_augmentor = get_augmentor(True, args.input_size, scale_range=args.scale_range,
                                        mean=mean[idx], std=std[idx],
                                        disable_scaleup=args.disable_scaleup,
                                        threed_data=args.threed_data, modality=args.modality[idx],
                                        version=args.augmentor_ver)

        train_augmentors.append(train_augmentor)

    train_dataset = video_data_cls(args.datadir, train_list_name, args.groups * args.num_segments, args.frames_per_group,
                                   num_clips=args.num_clips,
                                   modality=args.modality, image_tmpl=image_tmpl,
                                   num_classes=args.num_classes,
                                   dense_sampling=args.dense_sampling,
                                   transform=train_augmentors, is_train=True, test_mode=False,
                                   seperator=filename_seperator, filter_video=filter_video,
                                   fps=args.fps, audio_length=args.audio_length * args.num_segments,
                                   resampling_rate=args.resampling_rate)

    train_loader = build_dataflow(train_dataset, is_train=True, batch_size=args.batch_size,
                                  workers=args.workers, is_distributed=args.distributed)

    sgd_polices = model.parameters()
    optimizer = torch.optim.SGD(sgd_polices, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, args.lr_steps[0], gamma=0.1)
    elif args.lr_scheduler == 'multisteps':
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_steps, gamma=0.1)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    elif args.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    best_top1 = 0.0
    # optionally resume from a checkpoint
    if args.auto_resume:
        checkpoint_path = os.path.join(log_folder, 'checkpoint.pth.tar')
        if os.path.exists(checkpoint_path):
            args.resume = checkpoint_path
            print("Found the checkpoint in the log folder, will resume from there.")

    if args.resume:
        if args.rank == 0:
            logfile = open(os.path.join(log_folder, 'log.log'), 'a')
        if os.path.isfile(args.resume):
            if args.rank == 0:
                print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_top1 = checkpoint['best_top1']
            if args.gpu is not None:
                if not isinstance(best_top1, float):
                    best_top1 = best_top1.to(args.gpu)
                else:
                    best_top1 = best_top1.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except:
                pass
            if args.rank == 0:
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            del checkpoint  # dereference seems crucial
            torch.cuda.empty_cache()
        else:
            raise ValueError("Checkpoint is not found: {}".format(args.resume))
    else:
        if os.path.exists(os.path.join(log_folder, 'log.log')) and args.rank == 0:
            shutil.copyfile(os.path.join(log_folder, 'log.log'), os.path.join(
                log_folder, 'log.log.{}'.format(int(time.time()))))
        if args.rank == 0:
            logfile = open(os.path.join(log_folder, 'log.log'), 'w')

    if args.rank == 0:
        command = " ".join(sys.argv)
        print(command, flush=True)
        print(args, flush=True)
        print(model, flush=True)
        print(command, file=logfile, flush=True)
        print(args, file=logfile, flush=True)

    if args.resume == '' and args.rank == 0:
        print(model, file=logfile, flush=True)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_top1, train_top5, train_losses, train_speed, speed_data_loader, train_steps = \
            train(train_loader, model, train_criterion, optimizer, epoch + 1,
                  display=args.print_freq,  clip_gradient=args.clip_gradient, gpu_id=args.gpu, rank=args.rank,
                  eval_criterion=eval_criterion)
        if args.distributed:
            dist.barrier()

        eval_this_epoch = True
        if args.lazy_eval:
            if (epoch + 1) % 10 == 0 or (epoch + 1) >= args.epochs * 0.9:
                eval_this_epoch = True
            else:
                eval_this_epoch = False

        if eval_this_epoch:
            # evaluate on validation set
            val_top1, val_top5, val_losses, val_speed = validate(
                val_loader, model, val_criterion, gpu_id=args.gpu, eval_criterion=eval_criterion)
        else:
            val_top1, val_top5, val_losses, val_speed = 0.0, 0.0, 0.0, 0.0

        # update current learning rate
        if args.lr_scheduler == 'plateau':
            scheduler.step(val_losses)
        else:
            scheduler.step(epoch+1)

        if args.distributed:
            dist.barrier()

        # only logging at rank 0
        if args.rank == 0:
            print(
                'Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch\tData loading: {:.2f} ms/batch'.format(
                    epoch + 1, args.epochs, train_losses, train_top1, train_top5, train_speed * 1000.0,
                    speed_data_loader * 1000.0), file=logfile, flush=True)
            print(
                'Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch\tData loading: {:.2f} ms/batch'.format(
                    epoch + 1, args.epochs, train_losses, train_top1, train_top5, train_speed * 1000.0,
                    speed_data_loader * 1000.0), flush=True)
            if eval_this_epoch:
                print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch'.format(
                    epoch + 1, args.epochs, val_losses, val_top1, val_top5, val_speed * 1000.0), file=logfile, flush=True)
                print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch'.format(
                    epoch + 1, args.epochs, val_losses, val_top1, val_top5, val_speed * 1000.0), flush=True)

            # remember best prec@1 and save checkpoint
            is_best = val_top1 > best_top1
            best_top1 = max(val_top1, best_top1)


            save_dict = {'epoch': epoch + 1,
                         'arch': arch_name,
                         'state_dict': model.state_dict(),
                         'best_top1': best_top1,
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict()
                         }

            save_checkpoint(save_dict, is_best, filepath=log_folder)

        if args.distributed:
            dist.barrier()

    if args.rank == 0:
        logfile.close()


if __name__ == '__main__':
    main()

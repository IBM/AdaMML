import shutil
import os
import time
import multiprocessing

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm

from .video_transforms import (GroupRandomHorizontalFlip,
                               GroupMultiScaleCrop, GroupScale, GroupCenterCrop, GroupRandomCrop,
                               GroupNormalize, Stack, ToTorchFormatTensor, GroupRandomScale)

from torch.utils.data import DataLoader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def actnet_acc(logits, test_y, topk=None, have_softmaxed=False):
    from torchnet import meter

    """

    :param logits: (NxK)
    :param test_y: (Nx1)
    :param topk (tuple(int)):
    :return:
        - list[float]: topk acc
        - float: mAP
    """
    num_classes = logits.shape[1]
    topk = [1, min(5, num_classes)] if topk is None else topk
    single_label = True if len(test_y.shape) == 1 else False
    probs = F.softmax(logits, dim=1) if not have_softmaxed else logits
    if single_label:
        acc_meter = meter.ClassErrorMeter(topk=topk, accuracy=True)
        acc_meter.add(logits, test_y)
        acc = acc_meter.value()
        gt = torch.zeros_like(logits)
        gt[torch.LongTensor(range(gt.size(0))), test_y.view(-1)] = 1
    else:
        gt = test_y
        acc = [0] * len(topk)
    map_meter = meter.mAPMeter()
    map_meter.add(probs, gt)
    ap = map_meter.value() * 100.0
    return acc, ap.item()


def save_checkpoint(state, is_best, filepath='', epoch=None, suffix=''):
    curr_checkpoint_path = os.path.join(filepath, 'checkpoint.pth.tar')
    torch.save(state, curr_checkpoint_path)

    if epoch:
        shutil.copyfile(curr_checkpoint_path, os.path.join(filepath, 'checkpoint{}_{:02d}.pth.tar'.format(suffix, epoch)))
    if is_best:
        shutil.copyfile(curr_checkpoint_path, os.path.join(filepath, 'model_best.pth.tar'))

def extract_total_flops_params(summary):
    for line in summary.split("\n"):
        line = line.strip()
        if line == "":
            continue
        if "Total flops" in line:
            total_flops = line.split(":")[-1].strip()
        elif "Total params" in line:
            total_params = line.split(":")[-1].strip()

    return total_flops, total_params

def get_augmentor(is_train, image_size, mean=None,
                  std=None, disable_scaleup=False,
                  threed_data=False, version='v1', scale_range=None,
                  modality='rgb', num_clips=1, num_crops=1):

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    scale_range = [256, 320] if scale_range is None else scale_range

    if modality == 'sound':
        augments = [
            Stack(threed_data=threed_data),
            ToTorchFormatTensor(div=False, num_clips_crops=num_clips * num_crops)
        ]
    else:
        augments = []
        if is_train:
            if version == 'v1':
                augments += [
                    GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
                ]
            elif version == 'v2':
                augments += [
                    GroupRandomScale(scale_range),
                    GroupRandomCrop(image_size),
                ]
            augments += [GroupRandomHorizontalFlip(is_flow=(modality == 'flow'))]
        else:
            scaled_size = image_size if disable_scaleup else int(image_size / 0.875 + 0.5)
            augments += [
                GroupScale(scaled_size),
                GroupCenterCrop(image_size)
            ]
        augments += [
            Stack(threed_data=threed_data),
            ToTorchFormatTensor(num_clips_crops=num_clips * num_crops),
            GroupNormalize(mean=mean, std=std, threed_data=threed_data)
        ]

    augmentor = transforms.Compose(augments)
    return augmentor


def build_dataflow(dataset, is_train, batch_size, workers=36, is_distributed=False):
    workers = min(workers, multiprocessing.cpu_count())
    shuffle = False

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    if is_train:
        shuffle = sampler is None

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=workers, pin_memory=True, sampler=sampler)
    return data_loader


def compute_policy_loss(penalty_type, selection, cost_weights, gammas, cls_logits, cls_targets):
    num_modality = selection.shape[-1]
    policy_loss = torch.tensor(0.0, dtype=selection.dtype, device=selection.device)
    if penalty_type == 'mean':
        for w, pl in zip(cost_weights, selection.chunk(chunks=num_modality, dim=-1)):
            policy_loss = policy_loss + w * torch.mean(pl)

    elif penalty_type == 'blockdrop':
        top1_pred = torch.argmax(cls_logits.detach(), dim=-1)
        correctness = (top1_pred == cls_targets).type_as(cls_logits)

        selection = torch.mean(selection, dim=1)  # compute the selection per video per modality
        selection = selection * selection  # square it
        for w, pl in zip(cost_weights, selection.chunk(chunks=num_modality, dim=-1)):
            # pl: Nx1
            loss = w * torch.mean(correctness * pl)
            policy_loss = policy_loss + loss
        policy_loss = policy_loss + torch.mean((torch.ones_like(correctness) - correctness) * gammas)
    return policy_loss


def train(data_loader, model, criterion, optimizer, epoch, display=100,
          steps_per_epoch=99999999999, num_classes=None,
          clip_gradient=None, gpu_id=None, rank=0, eval_criterion=accuracy, **kwargs):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    end = time.time()
    num_batch = 0
    if gpu_id is None or gpu_id == 0:
        disable_status_bar = False
    else:
        disable_status_bar = True

    with tqdm(total=len(data_loader), disable=disable_status_bar) as t_bar:
        for i, (images, target) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
            if gpu_id is not None:
                if isinstance(images, list):
                    images = [x.cuda(gpu_id, non_blocking=True) for x in images]
                else:
                    images = images.cuda(gpu_id, non_blocking=True)
            output = model(images)
            target = target.cuda(gpu_id, non_blocking=True)
            # target = target.cuda(non_blocking=True)
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = eval_criterion(output, target)
            prec1 = prec1.to(device=loss.device)
            prec5 = prec5.to(device=loss.device)

            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size

            losses.update(loss.item(), target.size(0))
            top1.update(prec1[0], target.size(0))
            top5.update(prec5[0], target.size(0))
            # compute gradient and do SGD step
            loss.backward()

            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)

            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(data_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5), flush=True)
            num_batch += 1
            t_bar.update(1)

            if i > steps_per_epoch:
                break
    torch.cuda.empty_cache()
    return top1.avg, top5.avg, losses.avg, batch_time.avg, data_time.avg, num_batch


def validate(data_loader, model, criterion, gpu_id=None, eval_criterion=accuracy):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if gpu_id is None or gpu_id == 0:
        disable_status_bar = False
    else:
        disable_status_bar = True

    with torch.no_grad(), tqdm(total=len(data_loader), disable=disable_status_bar) as t_bar:
        end = time.time()
        for i, (images, target) in enumerate(data_loader):

            if gpu_id is not None:
                if isinstance(images, list):
                    images = [x.cuda(gpu_id, non_blocking=True) for x in images]
                else:
                    images = images.cuda(gpu_id, non_blocking=True)

            target = target.cuda(gpu_id, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = eval_criterion(output, target)
            prec1 = prec1.to(device=loss.device)
            prec5 = prec5.to(device=loss.device)

            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size
            losses.update(loss.item(), target.size(0))
            top1.update(prec1[0], target.size(0))
            top5.update(prec5[0], target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)
    torch.cuda.empty_cache()
    return top1.avg, top5.avg, losses.avg, batch_time.avg

def train_adamml(data_loader, model, criterion, optimizer, p_optimizer, epoch, modality, display=100,
                 steps_per_epoch=99999999999, clip_gradient=None, gpu_id=None,
                 rank=0, eval_criterion=accuracy, cost_weights=None,
                 gammas=None, penalty_type='blockdrop'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    selection_meter = { m:AverageMeter() for m in modality }

    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    model.zero_grad()
    end = time.time()
    num_batch = 0
    cost_weights = [0.0] * len(modality) if cost_weights is None else cost_weights
    cost_weights = torch.tensor(cost_weights).cuda()

    gammas = torch.tensor(gammas).cuda()
    if gpu_id is None or gpu_id == 0:
        disable_status_bar = False
    else:
        disable_status_bar = True

    with tqdm(total=len(data_loader), disable=disable_status_bar) as t_bar:
        for i, (images, target) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
            if gpu_id is not None:
                if isinstance(images, list):
                    images = [x.cuda(gpu_id, non_blocking=True) for x in images]
                else:
                    images = images.cuda(gpu_id, non_blocking=True)

            output, selection = model(images)
            # dim of selection: NxSxM
            target = target.cuda(gpu_id, non_blocking=True)
            policy_loss = compute_policy_loss(penalty_type, selection, cost_weights, gammas, output, target)
            selection_ratio = selection.detach().mean(0).mean(0)
            cls_loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = eval_criterion(output, target)
            prec1 = prec1.to(device=target.device)
            prec5 = prec5.to(device=target.device)
            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size

                dist.all_reduce(selection_ratio)
                selection_ratio /= world_size

            # classification is always considered but selection loss only used in training policy
            loss = cls_loss
            if model.module.update_policy_net:
                loss = loss + policy_loss

            losses.update(loss.item(), target.size(0))
            top1.update(prec1[0], target.size(0))
            top5.update(prec5[0], target.size(0))
            for ii, m in enumerate(modality):
                selection_meter[m].update(selection_ratio[ii].item())
            # compute gradient and do SGD step
            loss.backward()

            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)

            if model.module.update_policy_net:
                p_optimizer.step()
                p_optimizer.zero_grad()
            if model.module.update_main_net:
                optimizer.step()
                optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                selection_msg = "Selection: "
                for k, v in selection_meter.items():
                    selection_msg += "{}:{:.2f} ".format(k, v.avg * 100)
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      '{select}'.format(
                    epoch, i, len(data_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1, top5=top5, select=selection_msg), flush=True)
            num_batch += 1
            t_bar.update(1)

            if i > steps_per_epoch:
                break
    torch.cuda.empty_cache()
    return top1.avg, top5.avg, losses.avg, batch_time.avg, data_time.avg, num_batch, selection_meter


def validate_adamml(data_loader, model, criterion, num_segments, modality, gpu_id=None,
                    eval_criterion=accuracy, return_output=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    selection_meter = { m:AverageMeter() for m in modality }

    # switch to evaluate mode
    model.eval()
    if gpu_id is None or gpu_id == 0:
        disable_status_bar = False
    else:
        disable_status_bar = True

    outputs = None
    labels = None
    all_selections = None

    with torch.no_grad(), tqdm(total=len(data_loader), disable=disable_status_bar) as t_bar:
        end = time.time()
        for i, (images, target) in enumerate(data_loader):

            if gpu_id is not None:
                if isinstance(images, list):
                    images = [x.cuda(gpu_id, non_blocking=True) for x in images]
                else:
                    images = images.cuda(gpu_id, non_blocking=True)
            target = target.cuda(gpu_id, non_blocking=True)

            # compute output
            output, selection = model(images, num_segments)
            # dim of selection: NxSxM

            loss = criterion(output, target)

            selection_ratio = selection.detach().mean(0).mean(0)
            # measure accuracy and record loss
            prec1, prec5 = eval_criterion(output, target)
            prec1 = prec1.to(device=loss.device)
            prec5 = prec5.to(device=loss.device)

            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size
                dist.all_reduce(selection_ratio)
                selection_ratio /= world_size

            losses.update(loss.item(), target.size(0))
            top1.update(prec1[0], target.size(0))
            top5.update(prec5[0], target.size(0))
            for ii, m in enumerate(modality):
                selection_meter[m].update(selection_ratio[ii].item())            
            if outputs is None:
                outputs = concat_all_gather(output) if dist.is_initialized() else output
                labels = concat_all_gather(target) if dist.is_initialized() else target
                all_selections = concat_all_gather(selection) if dist.is_initialized() else selection
            else:
                outputs = torch.cat((outputs, concat_all_gather(output)), dim=0) if dist.is_initialized() else torch.cat((outputs, output), dim=0)
                labels = torch.cat((labels, concat_all_gather(target)), dim=0) if dist.is_initialized() else torch.cat((labels, target), dim=0)
                all_selections = torch.cat((all_selections, concat_all_gather(selection)), dim=0) if dist.is_initialized() else torch.cat((all_selections, selection), dim=0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)

    acc, mAP = actnet_acc(outputs, labels)
    top1, top5 = acc

    torch.cuda.empty_cache()

    flops = flops_computation(modality, selection_meter, num_segments)

    if return_output:
        return top1, top5, losses.avg, batch_time.avg, selection_meter, mAP, all_selections, flops, outputs
    else:
        return top1, top5, losses.avg, batch_time.avg, selection_meter, mAP, all_selections, flops


def flops_computation(modality, ratios, num_segments, net='resnet'):

    main_flops = {
        'rgb': 14135984128,
        'flow': 16338911232,
        'sound': 381739008,
    }

    policy_flops = {
        'rgb': 375446400,
        'sound': 381739008,
        'rgbdiff': 909283200,
        'lstm': 2359296
    }

    total_flops = 0

    for m in modality:
        if m == 'sound' or m == 'rgb':
            total_flops += (main_flops[m] * num_segments * ratios[m].avg) + (policy_flops[m] * num_segments)
        else:
            total_flops += (main_flops['flow'] * num_segments * ratios['flow'].avg) + (policy_flops['rgbdiff'] * num_segments)
    total_flops += policy_flops['lstm'] * num_segments
    total_flops /= 1e9
        
    return total_flops


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor.contiguous(), async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

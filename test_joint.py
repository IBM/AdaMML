import os
import time
import json
import warnings

import torch.nn as nn
from torch.nn import functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchsummary

from tqdm import tqdm

from models import build_model
from utils.utils import build_dataflow, extract_total_flops_params, AverageMeter, accuracy, actnet_acc
from utils.video_transforms import *
from utils.video_dataset import MultiVideoDataSetLMDB, MultiVideoDataSet, MultiVideoDataSetOnline
from utils.dataset_config import get_dataset_config
from opts import arg_parser


warnings.filterwarnings("ignore", category=UserWarning)


def load_categories(file_path):
    id_to_label = {}
    label_to_id = {}
    with open(file_path) as f:
        cls_id = 0
        for label in f.readlines():
            label = label.strip()
            if label == "":
                continue
            id_to_label[cls_id] = label
            label_to_id[label] = cls_id
            cls_id += 1
    return id_to_label, label_to_id


def eval_a_batch(data, model, in_channels, num_clips=1, num_crops=1, modality='rgb',
                 softmax=False, threed_data=False):
    with torch.no_grad():
        batch_size = data[0].shape[0]
        if threed_data:
            tmp = torch.chunk(data, num_clips * num_crops, dim=2)
            data = torch.cat(tmp, dim=0)
        else:
            new_data = []
            for x in data:
                xx = x.view((batch_size * num_crops * num_clips, -1) + x.size()[2:])
                new_data.append(xx)
            data = new_data
            #data = data.view((batch_size * num_crops * num_clips, -1) + data.size()[2:])
        result = model(data)

        if threed_data:
            tmp = torch.chunk(result, num_clips * num_crops, dim=0)
            result = None
            for i in range(len(tmp)):
                result = result + tmp[i] if result is not None else tmp[i]
            result /= (num_clips * num_crops)
        else:
            result = result.reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)

    return result


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    # TODO: for compatibility
    cudnn.benchmark = True

    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file, multilabel = get_dataset_config(args.dataset, args.use_lmdb)

    data_list_name = val_list_name if args.evaluate else test_list_name

    args.num_classes = num_classes
    if args.dataset == 'st2stv1' or args.dataset == 'activitynet':
        id_to_label, label_to_id = load_categories(os.path.join(args.datadir[0], label_file))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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

    model, arch_name = build_model(args, test_mode=True)
    mean = [model.mean(x) for x in args.modality]
    std = [model.std(x) for x in args.modality]


    model = model.cuda()
    model.eval()

    model = torch.nn.DataParallel(model).cuda()
    if args.pretrained is not None:
        print("=> using pre-trained model '{}'".format(arch_name))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> creating model '{}'".format(arch_name))

    # augmentor
    if args.disable_scaleup:
        scale_size = args.input_size
    else:
        scale_size = int(args.input_size / 0.875 + 0.5)

    val_augmentors = []
    for idx, modality in enumerate(args.modality):
        if modality == 'sound':
            augments = [
                Stack(threed_data=False),
                ToTorchFormatTensor(div=False, num_clips_crops=args.num_clips * args.num_crops)
            ]
        else:
            augments = []
            if args.num_crops == 1:
                augments += [
                    GroupScale(scale_size),
                    GroupCenterCrop(args.input_size)
                ]
            else:
                flip = True if args.num_crops == 10 else False
                augments += [
                    GroupOverSample(args.input_size, scale_size, num_crops=args.num_crops, flip=flip),
                ]
            augments += [
                Stack(threed_data=args.threed_data),
                ToTorchFormatTensor(num_clips_crops=args.num_clips * args.num_crops),
                GroupNormalize(mean=mean[idx], std=std[idx], threed_data=args.threed_data)
            ]

        augmentor = transforms.Compose(augments)
        val_augmentors.append(augmentor)
    # Data loading code
    #data_list = os.path.join(args.datadir, data_list_name)
    sample_offsets = list(range(-args.num_clips // 2 + 1, args.num_clips // 2 + 1))
    print("Image is scaled to {} and crop {}".format(scale_size, args.input_size))
    print("Number of crops: {}".format(args.num_crops))
    print("Number of clips: {}, offset from center with {}".format(args.num_clips, sample_offsets))

    if args.use_lmdb:
        video_data_cls = MultiVideoDataSetLMDB
    elif args.use_pyav:
        video_data_cls = MultiVideoDataSetOnline
    else:
        video_data_cls = MultiVideoDataSet
    val_dataset = video_data_cls(args.datadir, data_list_name, args.groups * args.num_segments, args.frames_per_group,
                                 num_clips=args.num_clips, modality=args.modality,
                                 image_tmpl=image_tmpl, dense_sampling=args.dense_sampling,
                                 fixed_offset=not args.random_sampling,
                                 transform=val_augmentors, is_train=False, test_mode=not args.evaluate,
                                 seperator=filename_seperator, filter_video=filter_video,
                                 fps=args.fps, audio_length=args.audio_length * args.num_segments,
                                 resampling_rate=args.resampling_rate)

    data_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                 workers=args.workers)

    log_folder = os.path.join(args.logdir, arch_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    batch_time = AverageMeter()
    if args.evaluate:
        logfile = open(os.path.join(log_folder, 'evaluate_log.log'), 'a')
    else:
        logfile = open(os.path.join(log_folder,
                                    'test_{}crops_{}clips_{}.csv'.format(args.num_crops,
                                                                         args.num_clips,
                                                                         args.input_size))
                       , 'w')
    total_outputs = 0
    outputs = torch.zeros((len(data_loader) * args.batch_size, num_classes))
    if args.evaluate:
        if multilabel:
            labels = torch.zeros((len(data_loader) * args.batch_size, num_classes),
                                 dtype=torch.long)
        else:
            labels = torch.zeros((len(data_loader) * args.batch_size), dtype=torch.long)
    else:
        labels = [None] * len(data_loader) * args.batch_size
    # switch to evaluate mode
    model.eval()

    total_batches = len(data_loader)
    with torch.no_grad(), tqdm(total=total_batches) as t_bar:
        end = time.time()
        for i, (video, label) in enumerate(data_loader):
            output = eval_a_batch(video, model, args.input_channels, num_clips=args.num_clips,
                                  num_crops=args.num_crops,
                                  modality=args.modality, softmax=True, threed_data=args.threed_data)
            batch_size = output.shape[0]
            outputs[total_outputs:total_outputs + batch_size, :] = output
            if multilabel:
                labels[total_outputs:total_outputs + batch_size, :] = label
            else:
                labels[total_outputs:total_outputs + batch_size] = label

            total_outputs += batch_size
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)

        outputs = outputs[:total_outputs]
        labels = labels[:total_outputs]
        print("Predicted {} videos.".format(total_outputs), flush=True)
        npy_prefix = os.path.basename(args.pretrained).split(".")[0]
        np.save(os.path.join(log_folder, '{}_{}crops_{}clips_{}_details_{}.npy'.format(
            "val" if args.evaluate else "test", args.num_crops, args.num_clips, args.input_size, npy_prefix)),
                outputs)

        if not args.evaluate:
            json_output = {
                'version': "VERSION 1.3",
                'results': {},
                'external_data': {'used': False, 'details': 'none'}
            }
            prob = F.softmax(outputs, dim=1).data.cpu().numpy().copy()
            predictions = np.argsort(prob, axis=1)
            for ii in range(len(predictions)):
                temp = predictions[ii][::-1][:5]
                preds = [str(pred) for pred in temp]
                if args.dataset == 'st2stv1':
                    print("{};{}".format(labels[ii], id_to_label[int(preds[0])]), file=logfile)
                elif args.dataset == 'activitynet':
                    video_id = labels[ii].replace("v_", "")
                    if video_id not in json_output['results']:
                        json_output['results'][video_id] = []
                    for jj in range(num_classes):
                        tmp = {'label': id_to_label[predictions[ii][::-1][jj]],
                               'score': prob[ii, predictions[ii][::-1][jj]].item()}
                        json_output['results'][video_id].append(tmp)
                else:
                    print("{};{}".format(labels[ii], ";".join(preds)), file=logfile)
            if args.dataset == 'activitynet':
                json.dump(json_output, logfile, indent=4)
        else:
            acc, mAP = actnet_acc(outputs, labels)
            top1, top5 = acc
            print(args.pretrained, file=logfile)
            print('Val@{}({}) (# crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@5: {:.4f}\tmAP: {:.4f}'.format(
                    args.input_size, scale_size, args.num_crops, args.num_clips, top1, top5, mAP), flush=True)
            print('Val@{}({}) (# crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@5: {:.4f}\tmAP: {:.4f}'.format(
                    args.input_size, scale_size, args.num_crops, args.num_clips, top1, top5, mAP,), flush=True, file=logfile)

    logfile.close()


if __name__ == '__main__':
    main()

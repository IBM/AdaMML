import os
import json
import numpy as np
import warnings

import torch
from torch.nn import functional as F

from models import build_model
from utils.utils import actnet_acc
from utils.dataset_config import get_dataset_config
from opts import arg_parser

from utils.activitynet.eval_classification import compute_map as activitynet_map

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


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    args.datadir = args.datadir[0]
    args.modality = args.modality[0]
    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file, multilabel = get_dataset_config(args.dataset, args.use_lmdb)
    data_list_name = val_list_name if args.evaluate else test_list_name

    args.num_classes = num_classes
    if args.dataset == 'st2stv1' or args.dataset == 'activitynet':
        id_to_label, label_to_id = load_categories(os.path.join(args.datadir, label_file))

    if args.dataset == 'activitynet':
        gt_json = os.path.join(args.datadir, 'activity_net.v1-3.min.json')

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5
    elif args.modality == 'rgbdiff':
        args.input_channels = 3 * 5
    elif args.modality == 'sound':
        args.input_channels = 1


    #model, arch_name = build_model(args, test_mode=True)

    if args.pred_weights == [] or args.pred_weights is None:
        args.pred_weights = [1.0 / len(args.pred_files)] * len(args.pred_files)

    predictions = None
    for pred_file, weight in zip(args.pred_files, args.pred_weights):
        tmp = torch.from_numpy(np.load(pred_file)).type(torch.float)
        tmp = F.softmax(tmp) if args.after_softmax else tmp
        tmp = tmp * weight
        predictions = tmp if predictions is None else predictions + tmp

    data_list = os.path.join(args.datadir, data_list_name)
    labels = []
    with open(data_list) as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            if args.evaluate:
                label = int(line.split(filename_seperator)[-1])
            else:  # get video id as label
                label = os.path.basename(line.split(filename_seperator)[0])
            labels.append(label)
    labels = np.asarray(labels)

    log_folder = os.path.join(args.logdir, args.dataset)
    print(log_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    predictions = predictions.data.numpy()
    # save the combined npy file
    np.save(os.path.join(log_folder, '{}_{}crops_{}clips_ensemble{}_details.npy'.format(
        "val" if args.evaluate else "test", args.num_crops, args.num_clips, len(args.pred_files))),
            predictions)

    outputs = torch.from_numpy(predictions)

    if args.evaluate:
        labels = torch.from_numpy(labels)
        logfile = open(os.path.join(log_folder, 'evaluate_log.log'), 'a')
        acc, mAP = actnet_acc(outputs, labels, have_softmaxed=args.after_softmax)
        top1, top5 = acc
        for i in args.pred_files:
            print(i, file=logfile)
        print(
            'Val (# crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@5: {:.4f}\tmAP: {:.4f}'.format(
                args.num_crops, args.num_clips, top1, top5, mAP), flush=True)
        print(
            'Val (# crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@5: {:.4f}\tmAP: {:.4f}'.format(
                args.num_crops, args.num_clips, top1, top5, mAP), flush=True, file=logfile)

    else:
        logfile = open(os.path.join(log_folder, 'test_{}crops_{}clips_ensemble{}.csv'.format(
            args.num_crops, args.num_clips, len(args.pred_files))), 'w')

        json_output_1 = {
            'version': "VERSION 1.3",
            'results': {},
            'external_data': {'used': False, 'details': 'none'}
        }
        json_output_2 = {
            'version': "VERSION 1.3",
            'results': {},
            'external_data': {'used': False, 'details': 'none'}
        }
        prob = outputs.data.numpy() if args.after_softmax else F.softmax(outputs, dim=1).data.numpy()
        predictions = np.argsort(prob, axis=1)
        for ii in range(len(predictions)):
            temp = predictions[ii][::-1][:5]
            preds = [str(pred) for pred in temp]
            if args.dataset == 'st2stv1':
                print("{};{}".format(labels[ii], id_to_label[int(preds[0])]), file=logfile)
            elif args.dataset == 'activitynet':
                video_id = labels[ii].replace("v_", "")
                if video_id not in json_output_1['results']:
                    json_output_1['results'][video_id] = []
                    json_output_2['results'][video_id] = []
                for jj in range(num_classes):
                    tmp = {'label': id_to_label[predictions[ii][::-1][jj]],
                           'score': prob[ii, predictions[ii][::-1][jj]].item()}
                    json_output_1['results'][video_id].append(tmp)
                    if jj == 0:
                        json_output_2['results'][video_id].append(tmp)
            else:
                print("{};{}".format(labels[ii], ";".join(preds)), file=logfile)
        if args.dataset == 'activitynet':
            json.dump(json_output_1, logfile, indent=4)
            act_map = activitynet_map(gt_json, json_output_1, 'validation', False, False, top_k=1, map_only=True)
            _, act_acc1 = activitynet_map(gt_json, json_output_2, 'validation', False, False, top_k=1, map_only=False)
            model_name = "_".join(args.pred_files)
            print("Model: {}\nTop@1: {:.4f}\tmAP: {:.4f}".format(model_name, act_acc1 * 100.0, act_map * 100.0), flush=True)
            tmp_file = os.path.join(log_folder,
                                    'test_{}crops_{}clips_{}.log'.format(args.num_crops,
                                                                         args.num_clips,
                                                                         args.input_size))
            with open(tmp_file, 'a') as f:
                print(model_name, file=f)
                print("Top@1: {:.4f}\tmAP: {:.4f}".format(act_acc1 * 100.0, act_map * 100.0), flush=True, file=f)

    logfile.close()


if __name__ == '__main__':
    main()

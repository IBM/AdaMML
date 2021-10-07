import argparse
from models.model_builder import MODEL_TABLE
from utils.dataset_config import DATASET_CONFIG

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Action recognition Training')

    # model definition
    parser.add_argument('--backbone_net', default='s3d', type=str, help='backbone network',
                        choices=list(MODEL_TABLE.keys()))
    parser.add_argument('-d', '--depth', default=18, type=int, metavar='N',
                        help='depth of resnet (default: 18)', choices=[18, 34, 50, 101, 152])
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='dropout ratio before the final layer')
    parser.add_argument('--groups', default=8, type=int, help='number of frames')
    parser.add_argument('--num_segments', default=1, type=int, help='number of consecutvie segments for adamml')
    parser.add_argument('--frames_per_group', default=1, type=int,
                        help='[uniform sampling] number of frames per group; '
                             '[dense sampling]: sampling frequency')
    parser.add_argument('--without_t_stride', dest='without_t_stride', action='store_true',
                        help='skip the temporal stride in the model')
    parser.add_argument('--pooling_method', default='max',
                        choices=['avg', 'max'], help='method for temporal pooling method or '
                                                     'which pool3d module')
    parser.add_argument('--fusion_point', default='logits', type=str, help='where to combine the features',
                        choices=['fc2', 'logits'])
    parser.add_argument('--prefix', default='', type=str, help='model prefix')
    parser.add_argument('--learnable_lf_weights', action='store_true')
    parser.add_argument('--causality_modeling', default=None, type=str,
                        help='causality modeling in policy net', choices=[None, 'lstm'])
    parser.add_argument('--cost_weights', default=None, type=float, nargs="+")
    parser.add_argument('--rng_policy', action='store_true', help='use rng as policy, baseline')
    parser.add_argument('--rng_threshold', type=float, default=0.5, help='rng threshold')
    parser.add_argument('--gammas', default=10.0, type=float)
    parser.add_argument('--penalty_type', default='blockdrop', type=str, choices=['mean', 'blockdrop'])

    # training setting
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--gpu_id', help='comma separated list of GPU(s) to use.', default=None)
    parser.add_argument('--disable_cudnn_benchmark', dest='cudnn_benchmark', action='store_false',
                        help='Disable cudnn to search the best mode (avoid OOM)')
    parser.add_argument('-b', '--batch-size', default=72, type=int,
                        metavar='N', help='mini-batch size (default: 72)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--p_lr', '--p_learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate for policy net')
    parser.add_argument('--lr_scheduler', default='cosine', type=str,
                        help='learning rate scheduler',
                        choices=['step', 'multisteps', 'cosine', 'plateau'])
    parser.add_argument('--lr_steps', default=[15, 30, 45], type=float, nargs="+",
                        metavar='LRSteps', help='[step]: use a single value: the periodto decay '
                                                'learning rate by 10. '
                                                '[multisteps] epochs to decay learning rate by 10')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--nesterov', action='store_true',
                        help='enable nesterov momentum optimizer')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warmup_epochs', default=5, type=int, metavar='N',
                        help='number of total epochs for warmup')
    parser.add_argument('--finetune_epochs', default=10, type=int, metavar='N',
                        help='number of total epochs for post finetune')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--auto_resume', action='store_true', help='if the log folder includes a checkpoint, automatically resume')
    parser.add_argument('--pretrained', dest='pretrained', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--unimodality_pretrained', type=str, nargs="+",
                        help='use pre-trained unimodality model', default=[])
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--clip_gradient', '--cg', default=None, type=float,
                        help='clip the total norm of gradient before update parameter')
    parser.add_argument('--curr_stage', type=str, help='set stage for staging training',
                        default='warmup', choices=['warmup', 'alternative_training', 'finetune'])
    # data-related
    parser.add_argument('-j', '--workers', default=18, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--datadir', metavar='DIR', help='path to dataset file list',
                        nargs="+", type=str)
    parser.add_argument('--dataset', default='activitynet',
                        choices=list(DATASET_CONFIG.keys()), help='path to dataset file list')
    parser.add_argument('--threed_data', action='store_true',
                        help='load data in the layout for 3D conv')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='input image size')
    parser.add_argument('--disable_scaleup', action='store_true',
                        help='do not scale up and then crop a small region, directly crop the input_size')
    parser.add_argument('--random_sampling', action='store_true',
                        help='perform determinstic sampling for data loader')
    parser.add_argument('--dense_sampling', action='store_true',
                        help='perform dense sampling for data loader')
    parser.add_argument('--augmentor_ver', default='v2', type=str, choices=['v1', 'v2'],
                        help='[v1] TSN data argmentation, [v2] resize the shorter side to `scale_range`')
    parser.add_argument('--scale_range', default=[256, 320], type=int, nargs="+",
                        metavar='scale_range', help='scale range for augmentor v2')
    parser.add_argument('--modality', default=['rgb'], type=str, help='rgb or flow or rgbdiff',
                        choices=['rgb', 'flow', 'rgbdiff', 'sound'], nargs="+")
    parser.add_argument('--mean', type=float, nargs="+",
                        metavar='MEAN', help='mean, dimension should be 3 for RGB and RGBdiff, 1 for flow')
    parser.add_argument('--std', type=float, nargs="+",
                        metavar='STD', help='std, dimension should be 3 for RGB and RGBdiff, 1 for flow')
    parser.add_argument('--skip_normalization', action='store_true',
                        help='skip mean and std normalization, default use imagenet`s mean and std.')
    parser.add_argument('--fps', type=float, metavar='FPS', default=29.97, help='fps of the video')
    parser.add_argument('--audio_length', type=float, default=1.28, help='length of audio segment')
    parser.add_argument('--resampling_rate', type=float, default=24000,
                        help='resampling rate of audio data')
    # logging
    parser.add_argument('--logdir', default='', type=str, help='log path')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='frequency to print the log during the training')
    parser.add_argument('--show_model', action='store_true', help='show model summary')
    
    # for testing and validation
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--num_crops', default=1, type=int, choices=[1, 3, 5, 10])
    parser.add_argument('--num_clips', default=1, type=int)
    parser.add_argument('--val_num_clips', default=10, type=int)
    parser.add_argument('--pred_files', type=str, nargs="+",
                         help='scale range for augmentor v2')
    parser.add_argument('--pred_weights', type=float, nargs="+",
                         help='scale range for augmentor v2')
    parser.add_argument('--after_softmax', action='store_true', help="perform softmax before ensumble")
    parser.add_argument('--lazy_eval', action='store_true', help="evaluate every 10 epochs and last 10 percentage of epochs")

    # for distributed learning, not supported yet
    parser.add_argument('--sync-bn', action='store_true',
                        help='sync BN across GPUs')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--hostfile', default='', type=str,
                        help='hostfile distributed learning')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    return parser

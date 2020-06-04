import os
import cv2
from PIL import Image
from multiprocessing import Pool
import argparse
import scipy.misc
import concurrent.futures
import time

def ToImg(raw_flow, bound):
    """
    this function scale the input pixels to 0-255 with bi-bound

    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    """
    flow = raw_flow
    flow[flow > bound] = bound
    flow[flow < -bound] = -bound
    flow -= (-bound)
    flow *= (255 / float(2 * bound))
    return flow


def save_flows(flows, flow_dir, num, bound):
    """
    To save the optical flow images and raw images
    :param flows: contains flow_x and flow_y
    :param save_dir: save_dir name (always equal to the video id)
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bi-bound to flow images
    :return: return 0
    """
    # rescale to 0~255 with the bound setting
    flow_x = ToImg(flows[..., 0], bound)
    flow_y = ToImg(flows[..., 1], bound)
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)

    # save the flows
    save_x = os.path.join(flow_dir, 'image_{:05d}_x.jpg'.format(num))
    save_y = os.path.join(flow_dir, 'image_{:05d}_y.jpg'.format(num))
    #flow_x_img = Image.fromarray(flow_x)
    #flow_y_img = Image.fromarray(flow_y)
    cv2.imwrite(save_x, flow_x)
    cv2.imwrite(save_y, flow_y)
    #scipy.misc.imsave(save_x, flow_x_img)
    #scipy.misc.imsave(save_y, flow_y_img)
    return 0


def dense_flow(augs):
    """
    To extract dense_flow images
    :param augs:the detailed augments:
        video_name: the video name which is like: 'v_xxxxxxx',if different ,please have a modify.
        save_dir: the destination path's final direction name.
        step: num of frames between each two extracted frames
        bound: bi-bound parameter
    :return: no returns
    """
    start_time = time.time()
    video_name, start_frame, end_frame, cls_id, bound, rgb_dir, flow_base_dir, image_fmt = augs
    if end_frame - start_frame + 1 <= 0:
        print("OMG {}".format(end_frame - start_frame + 1), flush=True)
        return "", 0, cls_id
    flow_dir = os.path.join(flow_base_dir, video_name.split('/')[-1])

    dtvl1 = cv2.createOptFlow_DualTVL1()

    file_path = os.path.join(rgb_dir, video_name, image_fmt % start_frame)
    prev_img = cv2.imread(file_path)
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    for k in range(start_frame + 1, end_frame + 1):
        cur_img = cv2.imread(os.path.join(rgb_dir, video_name, image_fmt % k))
        cur_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        flowDTVL1 = dtvl1.calc(prev_gray, cur_gray, None)
        save_flows(flowDTVL1, flow_dir, k - 1, bound)  # this is to save flows and img.
        prev_gray = cur_gray
        #continue

    end_time = time.time()

    #print("{} frames, spending {:.2f} sec".format(k-1, end_time-start_time), flush=True)
    return video_name, k -1, cls_id


def get_video_list(video_file, seperator=''):
    tmp = [x.strip().split(seperator) for x in open(video_file)]

    if not tmp:
        return []

    if len(tmp[0]) == 4:
        video_list = [(item[0], int(item[1]), int(item[2]), int(item[3])) for item in tmp]
    elif len(tmp[0]) == 3:
        video_list = [(item[0], int(item[1]), int(item[2]), -1) for item in tmp]
    else:
        video_list = [(item[0], int(item[1]), int(item[2]), int(item[3])) for item in tmp]
        raise ValueError('check the data file %s' % video_file)

    return video_list


def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--dataset', default='ucf101', type=str,
                        help='set the dataset name, to find the data path')
    parser.add_argument('--subset', default='train', type=str,
                        #choices=['train', 'val', 'test'],
                        help='set the sub-dataset name, to find the data path')
    parser.add_argument('--data_root', default='/n/zqj/video_classification/data', type=str)
    parser.add_argument('-o', '--output_root', default='/n/zqj/video_classification/data', type=str)
    parser.add_argument('--num_workers', default=4, type=int,
                        help='num of workers to act multi-process')
    #    parser.add_argument('--step',default=1,type=int,help='gap frames')
    parser.add_argument('--bound', default=20, type=int, help='set the maximum of optical flow')
    parser.add_argument('--s_', default=0, type=int, help='start id')
    parser.add_argument('--e_', default=9999999, type=int, help='end id')
    parser.add_argument('--mode', default='run', type=str,
                        help='set \'run\' if debug done, otherwise, set debug')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # example: if the data path not setted from args,just manually set them as belows.
    args = parse_args()
    # if args.dataset == 'st2stv2':
    #     data_separator = ' '
    #     image_format = '%05d.jpg'
    #     if args.subset == 'train':
    #         img_dir = os.path.join(args.data_root, 'training_256')
    #         flow_base_dir = os.path.join(args.data_root, 'training_256_flow')
    #         video_file = os.path.join(args.data_root, 'training_256.txt')
    #     elif args.subset == 'validation':
    #         img_dir = os.path.join(args.data_root, 'validation_256')
    #         flow_base_dir = os.path.join(args.data_root, 'validation_256_flow')
    #         video_file = os.path.join(args.data_root, 'validation_256.txt')
    #     elif args.subset == 'test':
    #         img_dir = os.path.join(args.data_root, 'test_256')
    #         flow_base_dir = os.path.join(args.data_root, 'test_256_flow')
    #         video_file = os.path.join(args.data_root, 'test_256.txt')
    # elif args.dataset == 'Kinetics':
    #     data_separator = ','
    if args.dataset == 'ucf101':
        data_separator = ' '
        image_format = '%05d.jpg'
    elif args.dataset == 'hmdb51':
        data_separator = ' '
        image_format = '%05d.jpg'
    elif args.dataset == 'actnet':
        data_separator = ' '
        image_format = 'image_%05d.jpg'
    file_list = os.path.join(args.data_root, '{}.txt'.format(args.subset))
    flow_folder = os.path.join(args.output_root, args.subset)
    flow_file_list = os.path.join(args.output_root, '{}.txt'.format(args.subset))
        
    if not os.path.exists(flow_folder):
        os.makedirs(flow_folder)

    video_list = get_video_list(file_list, data_separator)
    num_videos = len(video_list)

    # specify the augments
    #   step=args.step
    # get video list
    s_ = max(args.s_, 0)
    e_ = min(args.e_, num_videos)
    video_list = video_list[s_:e_]
    num_videos = len(video_list)

    # len_video = e_ - s_ + 1
    print('find {} videos.'.format(num_videos))

    x = [(item[0], item[1], item[2], item[3], args.bound, args.data_root, flow_folder, image_format) for item in video_list]
    cv2.setNumThreads(1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor, open(flow_file_list, 'a') as f_w:
        futures = [executor.submit(dense_flow, xx) for xx in x]
        for i, future in enumerate(futures):
            video_name, num_frames, cls_id = future.result()
            print("{} 1 {} {}".format(video_name, num_frames, cls_id), flush=True, file=f_w)
            if (i + 1) % 20 == 0:
                print("{}/{}".format(i, num_videos), flush=True)

    # pool = Pool(num_workers)
    # if mode == 'run':
    #     pool.map(dense_flow, x)
    # else:  # mode=='debug
    #     dense_flow((video_list[0][0], video_list[0][1], video_list[0][2], bound, flow_base_dir,
    #                 image_format))

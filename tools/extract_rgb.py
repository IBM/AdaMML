#!/usr/bin/env python3

import argparse
import os
import skvideo.io
import concurrent.futures
import subprocess
import glob


def video_to_images(video, targetdir, short_side=256):
    filename = video
    output_foldername = os.path.join(targetdir, os.path.basename(video).split(".")[0])
    if not os.path.exists(filename):
        print(f"{filename} is not existed.")
        return video, False
    else:
        try:
            video_meta = skvideo.io.ffprobe(filename)
            height = int(video_meta['video']['@height'])
            width = int(video_meta['video']['@width'])
        except Exception as e:
            print(f"Can not get video info: {filename}, error {e}")
            return video, False

        if width > height:
            scale = "scale=-1:{}".format(short_side)
        else:
            scale = "scale={}:-1".format(short_side)
        if not os.path.exists(output_foldername):
            os.makedirs(output_foldername)

        command = ['ffmpeg',
                   '-i', '"%s"' % filename,
                   '-vf', scale,
                   '-threads', '1',
                   '-loglevel', 'panic',
                   '-q:v', '2',
                   '{}/'.format(output_foldername) + '"%05d.jpg"']
        command = ' '.join(command)
        try:
            subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except Exception as e:
            print(f"fail to convert {filename}, error: {e}")
            return video, False

        return video, True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('videos_dir', help='Input directory of videos with audio')
    parser.add_argument('output_dir', help='Output directory to store JPEG files')
    parser.add_argument('--num_workers', help='Number of workers', default=8, type=int)
    args = parser.parse_args()

    video_list = glob.glob(args.videos_dir + '/**/*.*', recursive=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(video_to_images, video, args.output_dir, 256)
                   for video in video_list]
        total_videos = len(futures)
        for future in concurrent.futures.as_completed(futures):
            video_id, success = future.result()
            if not success:
                print(f"Something wrong for {video_id}")
    print("Completed")

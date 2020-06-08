import argparse
import subprocess
import os


def ffmpeg_extraction(input_video, output_sound, sample_rate):
    ffmpeg_command = ['ffmpeg', '-i', input_video,
                      '-vn', '-acodec', 'pcm_s16le',
                      '-ac', '1', '-ar', sample_rate,
                      output_sound]

    subprocess.call(ffmpeg_command)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('videos_dir', help='Input directory of videos with audio')
    parser.add_argument('output_dir', help='Output directory to store .wav files')
    parser.add_argument('--dataset', required=True, help='dataset', choices=['activitynet', 'fcvid', 'kinetics-sounds', 'moments'])
    parser.add_argument('--sample_rate', default='24000', help='Rate to resample audio')

    args = parser.parse_args()

    if args.dataset == 'activitynet':
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        for root, dirs, files in os.walk(args.videos_dir):
            for f in files:
                if f.endswith('.webm'):
                    ffmpeg_extraction(os.path.join(root, f),
                                      os.path.join(args.output_dir,
                                                   os.path.splitext(f)[0] + '.wav'),
                                      args.sample_rate)
    elif args.dataset == 'fcvid':
        for root, dirs, files in os.walk(args.videos_dir):
            for f in files:
                if not os.path.exists(root.replace('fcvid/videos', 'audios/fcvid1')):
                    os.mkdir(root.replace('fcvid/videos', 'audios/fcvid1'))
                ffmpeg_extraction(os.path.join(root, f), os.path.join(root.replace('fcvid/videos', 'audios/fcvid1'), os.path.splitext(f)[0] + '.wav'), args.sample_rate)

    elif args.dataset == 'kinetics-sounds':
        for root, dirs, files in os.walk(args.videos_dir):
            for f in files:
                if not os.path.exists(os.path.join(args.output_dir, os.path.basename(os.path.normpath(root)))):
                    os.mkdir(os.path.join(args.output_dir, os.path.basename(os.path.normpath(root))))
                ffmpeg_extraction(os.path.join(root, f), os.path.join(args.output_dir, os.path.basename(os.path.normpath(root)), os.path.splitext(f)[0] + '.wav'), args.sample_rate)
    else:
        raise ValueError(args.dataset)

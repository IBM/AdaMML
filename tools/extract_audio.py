import argparse
import subprocess
import os
import glob


def ffmpeg_extraction(input_video, output_sound, sample_rate):
    ffmpeg_command = ['ffmpeg', '-i', input_video,
                      '-vn', '-acodec', 'pcm_s16le',
                      '-loglevel', 'panic',
                      '-ac', '1', '-ar', sample_rate,
                      output_sound]

    subprocess.call(ffmpeg_command)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('videos_dir', help='Input directory of videos with audio')
    parser.add_argument('output_dir', help='Output directory to store .wav files')
    parser.add_argument('--sample_rate', default='24000', help='Rate to resample audio')
    parser.add_argument('--ext', default=['.mp4'], nargs='+', help='The extension of videos')

    args = parser.parse_args()

    video_list = glob.glob(args.videos_dir + '/**/*.*', recursive=True)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for video in video_list:
        ffmpeg_extraction(video,
                          os.path.join(args.output_dir,
                                       os.path.basename(video).split(".")[0] + ".wav"),
                          args.sample_rate)

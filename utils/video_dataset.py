import os
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data

def random_clip(video_frames, sampling_rate, frames_per_clip, fixed_offset=False, start_frame_idx=0, end_frame_idx=None):
    """

    Args:
        video_frames (int): total frame number of a video
        sampling_rate (int): sampling rate for clip, pick one every k frames
        frames_per_clip (int): number of frames of a clip
        fixed_offset (bool): used with sample offset to decide the offset value deterministically.

    Returns:
        list[int]: frame indices (started from zero)
    """
    new_sampling_rate = sampling_rate
    highest_idx = video_frames - new_sampling_rate * frames_per_clip if end_frame_idx is None else end_frame_idx
    if highest_idx <= 0:
        random_offset = 0
    else:
        if fixed_offset:
            random_offset = (video_frames - new_sampling_rate * frames_per_clip) // 2
        else:
            random_offset = int(np.random.randint(start_frame_idx, highest_idx, 1))
    frame_idx = [int(random_offset + i * sampling_rate) % video_frames for i in range(frames_per_clip)]
    return frame_idx


def compute_img_diff(image_1, image_2, bound=255.0):
    image_diff = np.asarray(image_1, dtype=np.float) - np.asarray(image_2, dtype=np.float)
    image_diff += bound
    image_diff *= (255.0 / float(2 * bound))
    image_diff = image_diff.astype(np.uint8)
    image_diff = Image.fromarray(image_diff)
    return image_diff


def load_image(root_path, directory, image_tmpl, idx, modality):
    """

    :param root_path:
    :param directory:
    :param image_tmpl:
    :param idx: if it is a list, load a batch of images
    :param modality:
    :return:
    """
    def _safe_load_image(img_path):
        img = None
        num_try = 0
        while num_try < 10:
            try:
                img_tmp = Image.open(img_path)
                img = img_tmp.copy()
                img_tmp.close()
                break
            except Exception as e:
                print('[Will try load again] error loading image: {}, '
                      'error: {}'.format(img_path, str(e)))
                num_try += 1
        if img is None:
            raise ValueError('[Fail 10 times] error loading image: {}'.format(img_path))
        return img

    if not isinstance(idx, list):
        idx = [idx]
    out = []
    if modality == 'rgb':
        for i in idx:
            image_path_file = os.path.join(root_path, directory, image_tmpl.format(i))
            out.append(_safe_load_image(image_path_file))
    elif modality == 'rgbdiff':
        tmp = {}
        new_idx = np.unique(np.concatenate((np.asarray(idx), np.asarray(idx) + 1)))
        for i in new_idx:
            image_path_file = os.path.join(root_path, directory, image_tmpl.format(i))
            tmp[i] = _safe_load_image(image_path_file)
        for k in idx:
            img_ = compute_img_diff(tmp[k+1], tmp[k])
            out.append(img_)
        del tmp
    elif modality == 'flow':
        for i in idx:
            flow_x_name = os.path.join(root_path, directory, "x_" + image_tmpl.format(i))
            flow_y_name = os.path.join(root_path, directory, "y_" + image_tmpl.format(i))
            out.extend([_safe_load_image(flow_x_name), _safe_load_image(flow_y_name)])
            
    return out


def load_sound(data_dir, record, idx, fps, audio_length, resampling_rate,
               window_size=10, step_size=5, eps=1e-6):
    import librosa
    """idx must be the center frame of a clip"""
    centre_sec = (record.start_frame + idx) / fps
    left_sec = centre_sec - (audio_length / 2.0)
    right_sec = centre_sec + (audio_length / 2.0)
    audio_fname = os.path.join(data_dir, record.path)
    if not os.path.exists(audio_fname):
        return [Image.fromarray(np.zeros((256, 256 * int(audio_length / 1.28))))]
    samples, sr = librosa.core.load(audio_fname, sr=None, mono=True)
    duration = samples.shape[0] / float(resampling_rate)

    left_sample = int(round(left_sec * resampling_rate))
    right_sample = int(round(right_sec * resampling_rate))

    required_samples = int(round(resampling_rate * audio_length))

    if left_sec < 0:
        samples = samples[:required_samples]
    elif right_sec > duration:
        samples = samples[-required_samples:]
    else:
        samples = samples[left_sample:right_sample]

    # if the samples is not long enough, repeat the waveform
    if len(samples) < required_samples:
        multiplies = required_samples / len(samples)
        samples = np.tile(samples, int(multiplies + 0.5) + 1)
        samples = samples[:required_samples]

    # log sepcgram
    nperseg = int(round(window_size * resampling_rate / 1e3))
    noverlap = int(round(step_size * resampling_rate / 1e3))
    spec = librosa.stft(samples, n_fft=511, window='hann', hop_length=noverlap,
                        win_length=nperseg, pad_mode='constant')
    spec = np.log(np.real(spec * np.conj(spec)) + eps)
    img = Image.fromarray(spec)    
    return [img]


def sample_train_clip(video_length, num_consecutive_frames, num_frames, sample_freq, dense_sampling, num_clips=1):

    max_frame_idx = max(1, video_length - num_consecutive_frames + 1)
    if dense_sampling:                
        frame_idx = np.zeros((num_clips, num_frames), dtype=int)
        if num_clips == 1:  # backward compatibility
            frame_idx[0] = np.asarray(random_clip(max_frame_idx, sample_freq, num_frames, False))
        else:
            max_start_frame_idx = max_frame_idx - sample_freq * num_frames
            frames_per_segment = max_start_frame_idx // num_clips
            for i in range(num_clips):
                if frames_per_segment <= 0:
                    frame_idx[i] = np.asarray(random_clip(max_frame_idx, sample_freq, num_frames, False))
                else:
                    frame_idx[i] = np.asarray(random_clip(max_frame_idx, sample_freq, num_frames, False, i * frames_per_segment, (i+1) * frames_per_segment))

        frame_idx = frame_idx.flatten()
    else:  # uniform sampling
        total_frames = num_frames * sample_freq
        ave_frames_per_group = max_frame_idx // num_frames
        if ave_frames_per_group >= sample_freq:
            # randomly sample f images per segement
            frame_idx = np.arange(0, num_frames) * ave_frames_per_group
            frame_idx = np.repeat(frame_idx, repeats=sample_freq)
            offsets = np.random.choice(ave_frames_per_group, sample_freq, replace=False)
            offsets = np.tile(offsets, num_frames)
            frame_idx = frame_idx + offsets
        elif max_frame_idx < total_frames:
            # need to sample the same images
            frame_idx = np.random.choice(max_frame_idx, total_frames)
        else:
            # sample cross all images
            frame_idx = np.random.choice(max_frame_idx, total_frames, replace=False)
        frame_idx = np.sort(frame_idx)
    frame_idx = frame_idx + 1
    return frame_idx


def sample_val_test_clip(video_length, num_consecutive_frames, num_frames, sample_freq, dense_sampling,
                         fixed_offset, num_clips, whole_video):
    max_frame_idx = max(1, video_length - num_consecutive_frames + 1)
    if whole_video:
        return np.arange(1, max_frame_idx, step=sample_freq, dtype=int)
    if dense_sampling:
        if fixed_offset:
            sample_pos = max(1, 1 + max_frame_idx - sample_freq * num_frames)
            t_stride = sample_freq
            start_list = np.linspace(0, sample_pos - 1, num=num_clips, dtype=int)
            frame_idx = []
            for start_idx in start_list.tolist():
                frame_idx += [(idx * t_stride + start_idx) % max_frame_idx for idx in
                              range(num_frames)]
        else:
            frame_idx = []
            for i in range(num_clips):
                frame_idx.extend(random_clip(max_frame_idx, sample_freq, num_frames))
        frame_idx = np.asarray(frame_idx) + 1
    else:  # uniform sampling
        if fixed_offset:
            frame_idices = []
            sample_offsets = list(range(-num_clips // 2 + 1, num_clips // 2 + 1))
            for sample_offset in sample_offsets:
                if max_frame_idx > num_frames:
                    tick = max_frame_idx / float(num_frames)
                    curr_sample_offset = sample_offset
                    if curr_sample_offset >= tick / 2.0:
                        curr_sample_offset = tick / 2.0 - 1e-4
                    elif curr_sample_offset < -tick / 2.0:
                        curr_sample_offset = -tick / 2.0
                    frame_idx = np.array([int(tick / 2.0 + curr_sample_offset + tick * x) for x in
                                          range(num_frames)])
                else:
                    np.random.seed(sample_offset - (-num_clips // 2 + 1))
                    frame_idx = np.random.choice(max_frame_idx, num_frames)
                frame_idx = np.sort(frame_idx)
                frame_idices.extend(frame_idx.tolist())
        else:
            frame_idices = []
            for i in range(num_clips):
                total_frames = num_frames * sample_freq
                ave_frames_per_group = max_frame_idx // num_frames
                if ave_frames_per_group >= sample_freq:
                    # randomly sample f images per segment
                    frame_idx = np.arange(0, num_frames) * ave_frames_per_group
                    frame_idx = np.repeat(frame_idx, repeats=sample_freq)
                    offsets = np.random.choice(ave_frames_per_group, sample_freq,
                                               replace=False)
                    offsets = np.tile(offsets, num_frames)
                    frame_idx = frame_idx + offsets
                elif max_frame_idx < total_frames:
                    # need to sample the same images
                    np.random.seed(i)
                    frame_idx = np.random.choice(max_frame_idx, total_frames)
                else:
                    # sample cross all images
                    np.random.seed(i)
                    frame_idx = np.random.choice(max_frame_idx, total_frames, replace=False)
                frame_idx = np.sort(frame_idx)
                frame_idices.extend(frame_idx.tolist())
        frame_idx = np.asarray(frame_idices) + 1
    return frame_idx


class VideoRecord(object):
    def __init__(self, path, start_frame, end_frame, label, reverse=False):
        self.path = path
        self.video_id = os.path.basename(path)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.label = label
        self.reverse = reverse

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame + 1

    def __str__(self):
        return self.path


class VideoDataSet(data.Dataset):

    def __init__(self, root_path, list_file, num_groups=64, frames_per_group=1, sample_offset=0, num_clips=1,
                 modality='rgb', dense_sampling=False, fixed_offset=True,
                 image_tmpl='{:05d}.jpg', transform=None, is_train=True, test_mode=False, seperator=' ',
                 filter_video=0, num_classes=None, whole_video=False,
                 fps=29.97, audio_length=1.28, resampling_rate=24000):
        """

        Arguments have different meaning when dense_sampling is True:
            - num_groups ==> number of frames
            - frames_per_group ==> sample every K frame
            - sample_offset ==> number of clips used in validation or test mode

        Args:
            root_path (str): the file path to the root of video folder
            list_file (str): the file list, each line with folder_path, start_frame, end_frame, label_id
            num_groups (int): number of frames per data sample
            frames_per_group (int): number of frames within one group
            sample_offset (int): used in validation/test, the offset when sampling frames from a group
            modality (str): rgb or flow
            dense_sampling (bool): dense sampling in I3D
            fixed_offset (bool): used for generating the same videos used in TSM
            image_tmpl (str): template of image ids
            transform: the transformer for preprocessing
            is_train (bool): shuffle the video but keep the causality
            test_mode (bool): testing mode, no label
            whole_video (bool): take whole video
            fps (float): frame rate per second, used to localize sound when frame idx is selected.
            audio_length (float): the time window to extract audio feature.
            resampling_rate (int): used to resampling audio extracted from wav
        """
        if modality not in ['flow', 'rgb', 'rgbdiff', 'sound']:
            raise ValueError("modality should be 'flow' or 'rgb' or 'rgbdiff' or 'sound'.")

        self.root_path = root_path
        self.list_file = os.path.join(root_path, list_file)
        self.num_groups = num_groups
        self.num_frames = num_groups
        self.frames_per_group = frames_per_group
        self.sample_freq = frames_per_group
        self.num_clips = num_clips
        self.sample_offset = sample_offset
        self.fixed_offset = fixed_offset
        self.dense_sampling = dense_sampling
        self.modality = modality.lower()
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.is_train = is_train
        self.test_mode = test_mode
        self.separator = seperator
        self.filter_video = filter_video
        self.whole_video = whole_video
        self.fps = fps
        self.audio_length = audio_length
        self.resampling_rate = resampling_rate
        self.video_length = (self.num_frames * self.sample_freq) / self.fps


        if self.modality in ['flow', 'rgbdiff']:
            self.num_consecutive_frames = 5
        else:
            self.num_consecutive_frames = 1

        self.video_list, self.multi_label = self._parse_list()
        self.num_classes = num_classes

    def _parse_list(self):
        # usually it is [video_id, num_frames, class_idx]
        # or [video_id, start_frame, end_frame, list of class_idx]
        tmp = []
        original_video_numbers = 0
        for x in open(self.list_file):
            elements = x.strip().split(self.separator)
            start_frame = int(elements[1])
            end_frame = int(elements[2])
            total_frame = end_frame - start_frame + 1
            original_video_numbers += 1
            if self.test_mode:
                tmp.append(elements)
            else:
                if total_frame >= self.filter_video:
                    tmp.append(elements)

        num = len(tmp)
        print("The number of videos is {} (with more than {} frames) "
              "(original: {})".format(num, self.filter_video, original_video_numbers), flush=True)
        assert (num > 0)
        # TODO: a better way to check if multi-label or not
        multi_label = np.mean(np.asarray([len(x) for x in tmp])) > 4.0
        file_list = []
        for item in tmp:
            if self.test_mode:
                file_list.append([item[0], int(item[1]), int(item[2]), -1])
            else:
                labels = []
                for i in range(3, len(item)):
                    labels.append(float(item[i]))
                if not multi_label:
                    labels = labels[0] if len(labels) == 1 else labels
                file_list.append([item[0], int(item[1]), int(item[2]), labels])

        video_list = [VideoRecord(item[0], item[1], item[2], item[3]) for item in file_list]
        # flow model has one frame less
        if self.modality in ['rgbdiff']:
            for i in range(len(video_list)):
                video_list[i].end_frame -= 1

        return video_list, multi_label

    def remove_data(self, idx):
        original_video_num = len(self.video_list)
        self.video_list = [v for i, v in enumerate(self.video_list) if i not in idx]
        print("Original videos: {}\t remove {} videos, remaining {} videos".format(original_video_num, len(idx), len(self.video_list)))

    def _sample_indices(self, record):
        return sample_train_clip(record.num_frames, self.num_consecutive_frames, self.num_frames,
                                 self.sample_freq, self.dense_sampling, self.num_clips)

    def _get_val_indices(self, record):
        return sample_val_test_clip(record.num_frames, self.num_consecutive_frames, self.num_frames,
                                    self.sample_freq, self.dense_sampling, self.fixed_offset,
                                    self.num_clips, self.whole_video)

    def __getitem__(self, index):
        """
        Returns:
            torch.FloatTensor: (3xgxf)xHxW dimension, g is number of groups and f is the frames per group.
            torch.FloatTensor: the label
        """
        record = self.video_list[index]
        # check this is a legit video folder
        indices = self._sample_indices(record) if self.is_train else self._get_val_indices(record)
        images = self.get_data(record, indices)
        images = self.transform(images)
        label = self.get_label(record)

        # re-order data to targeted format.
        return images, label

    def get_data(self, record, indices):
        images = []
        if self.whole_video:
            tmp = len(indices) % self.num_frames
            if tmp != 0:
                indices = indices[:-tmp]
            num_clips = len(indices) // self.num_frames
            #print(tmp, indices, self.num_frames, num_clips)
        else:
            num_clips = self.num_clips
        if self.modality == 'sound':                
            new_indices = [indices[i * self.num_frames: (i + 1) * self.num_frames]
                           for i in range(num_clips)]
            for curr_indiecs in new_indices:
                center_idx = (curr_indiecs[self.num_frames // 2 - 1] + curr_indiecs[self.num_frames // 2]) // 2 \
                    if self.num_frames % 2 == 0 else curr_indiecs[self.num_frames // 2]
                center_idx = min(record.num_frames, center_idx)
                seg_imgs = load_sound(self.root_path, record, center_idx,
                                      self.fps, self.audio_length, self.resampling_rate)
                images.extend(seg_imgs)
        else:
            images = []
            for seg_ind in indices:
                new_seg_ind = [min(seg_ind + record.start_frame - 1 + i, record.num_frames)
                               for i in range(self.num_consecutive_frames)]
                seg_imgs = load_image(self.root_path, record.path, self.image_tmpl,
                                      new_seg_ind, self.modality)
                images.extend(seg_imgs)
        return images

    def get_label(self, record):
        if self.test_mode:
            # in test mode, return the video id as label
            label = record.video_id
        else:
            if not self.multi_label:
                label = int(record.label)
            else:
                # create a binary vector.
                label = torch.zeros(self.num_classes, dtype=torch.float)
                for x in record.label:
                    label[int(x)] = 1.0
        return label

    def __len__(self):
        return len(self.video_list)


class MultiVideoDataSet(data.Dataset):

    def __init__(self, root_path, list_file, num_groups=64, frames_per_group=1, sample_offset=0, num_clips=1,
                 modality='rgb', dense_sampling=False, fixed_offset=True,
                 image_tmpl='{:05d}.jpg', transform=None, is_train=True, test_mode=False, seperator=' ',
                 filter_video=0, num_classes=None, whole_video=False,
                 fps=29.97, audio_length=1.28, resampling_rate=24000):
        """
        # root_path, modality and transform become list, each for one modality

        Argments have different meaning when dense_sampling is True:
            - num_groups ==> number of frames
            - frames_per_group ==> sample every K frame
            - sample_offset ==> number of clips used in validation or test mode

        Args:
            root_path (str): the file path to the root of video folder
            list_file (str): the file list, each line with folder_path, start_frame, end_frame, label_id
            num_groups (int): number of frames per data sample
            frames_per_group (int): number of frames within one group
            sample_offset (int): used in validation/test, the offset when sampling frames from a group
            modality (str): rgb or flow
            dense_sampling (bool): dense sampling in I3D
            fixed_offset (bool): used for generating the same videos used in TSM
            image_tmpl (str): template of image ids
            transform: the transformer for preprocessing
            is_train (bool): shuffle the video but keep the causality
            test_mode (bool): testing mode, no label
        """

        video_datasets = []
        for i in range(len(modality)):
            tmp = VideoDataSet(root_path[i], os.path.join(root_path[i], list_file),
                               num_groups, frames_per_group, sample_offset,
                               num_clips, modality[i], dense_sampling, fixed_offset,
                               image_tmpl, transform[i], is_train, test_mode, seperator,
                               filter_video, num_classes, whole_video, fps, audio_length, resampling_rate)
            video_datasets.append(tmp)

        self.video_datasets = video_datasets
        self.is_train = is_train
        self.test_mode = test_mode
        self.num_frames = num_groups
        self.sample_freq = frames_per_group
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.fixed_offset = fixed_offset
        self.modality = modality
        self.num_classes = num_classes
        self.whole_video = whole_video

        self.video_list = video_datasets[0].video_list
        self.num_consecutive_frames = max([x.num_consecutive_frames for x in self.video_datasets])

    def _sample_indices(self, record):
        return sample_train_clip(record.num_frames, self.num_consecutive_frames, self.num_frames,
                                 self.sample_freq, self.dense_sampling, self.num_clips)

    def _get_val_indices(self, record):
        return sample_val_test_clip(record.num_frames, self.num_consecutive_frames, self.num_frames,
                                    self.sample_freq, self.dense_sampling, self.fixed_offset,
                                    self.num_clips, self.whole_video)

    def remove_data(self, idx):
        for i in range(len(self.video_datasets)):
            self.video_datasets[i].remove_data(idx)
        self.video_list = self.video_datasets[0].video_list

    def __getitem__(self, index):
        """
        Returns:
            torch.FloatTensor: (3xgxf)xHxW dimension, g is number of groups and f is the frames per group.
            torch.FloatTensor: the label
        """

        record = self.video_list[index]
        if self.is_train:
            indices = self._sample_indices(record)
        else:
            indices = self._get_val_indices(record)

        multi_modalities = []
        for modality, video_dataset in zip(self.modality, self.video_datasets):
            record = video_dataset.video_list[index]
            images = video_dataset.get_data(record, indices)
            images = video_dataset.transform(images)
            label = video_dataset.get_label(record)
            multi_modalities.append((images, label))

        return [x for x, y in multi_modalities], multi_modalities[0][1]

    def __len__(self):
        return len(self.video_list)


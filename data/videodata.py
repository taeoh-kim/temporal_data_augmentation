import random
import cv2
import numpy as np
import torch.utils.data as data
from data.utils import color_jitter
from data.augment import RandAugment, AugMix

# General Video Data Loader
# Depends on OpenCV VideoCapture Class
# written by Taeoh Kim (kto@yonsei.ac.kr)

class VideoData(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.videos, self.labels = self._scan()
        self.clip_len = args.clip_len
        if self.train:
            if args.aug_mode == 'randaug':
                self.randaug = RandAugment(n=args.randaug_n, m=args.randaug_m, temp_degree=args.aug_degree, range=args.randaug_range)
            else:
                raise NotImplementedError('Invalid augmentation mode.')
        self.crop_size = args.crop_size
        self.frame_sample_rate = args.frame_sample_rate

    def _scan(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        # 1. Video Load: "buffer=[Clip_len, H, W, 3]" & 2. Temporal Crop
        buffer, filename = self._get_video(idx)

        # If num of frames is shorter than clip_len, do PAdding
        if buffer.shape[0] < self.clip_len:
            buffer = self._short_frame_padding(buffer)

        if self.train:
            # 3. Horizontal Flip
            buffer = self._flip(buffer)

        # 4. Spatial Crop "buffer=[clip_len, crop_size, crop_size, 3]"
        if self.train or self.args.test_view == 10:
            buffer = self._get_spatial_patch(buffer, self.crop_size)
        elif self.args.test_view == 30:
            buffer = self._get_spatial_patch30(buffer)

        # 5. Rand augmentation
        if self.train and self.args.rand_augmentation:
            buffer = self._rand_aug(buffer)

        # 6. Normalization
        buffer = self._normalize_and_make_tensor(buffer)

        return buffer, self.labels[idx], filename

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    # @profile
    def _get_video(self, idx):
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(self.videos[idx])
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        start_idx = 0
        end_idx = frame_count - 1
        frame_count_sample = frame_count // self.frame_sample_rate - 1

        if self.train:
            if frame_count > self.clip_len + 1:
                # ------------------------------------------------------------ Temporal Crop (Train Only)
                time_index = np.random.randint(start_idx, end_idx - self.clip_len)
                start_idx = time_index
                end_idx = time_index + self.clip_len - 1
                frame_count_sample = self.clip_len

        buffer = np.empty((frame_count_sample, frame_height, frame_width, 3), np.dtype('float32'))  # T H W C

        count = 0
        retaining = True
        sample_count = 0

        # read in each frame, one at a time into the numpy buffer array
        while (count <= end_idx and retaining):
            # retaining, frame = capture.read()
            retaining = capture.grab()
            if count < start_idx:
                count += 1
                continue
            if retaining is False or count > end_idx:
                break
            _, frame = capture.retrieve()
            if sample_count < frame_count_sample:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buffer[sample_count] = frame
                sample_count = sample_count + 1
            count += 1
        capture.release()

        return buffer, self.videos[idx]

    def _short_frame_padding(self, buffer):
        if buffer.shape[0] < (self.clip_len // 2):
            newbuffer = np.empty((buffer.shape[0] * 4, buffer.shape[1], buffer.shape[2], 3), np.dtype('float32'))
            for i in range(buffer.shape[0]):
                newbuffer[4 * i] = buffer[i]
                newbuffer[4 * i + 1] = buffer[i]
                newbuffer[4 * i + 2] = buffer[i]
                newbuffer[4 * i + 3] = buffer[i]
        else:
            newbuffer = np.empty((buffer.shape[0] * 2, buffer.shape[1], buffer.shape[2], 3), np.dtype('float32'))
            for i in range(buffer.shape[0]):
                newbuffer[2 * i] = buffer[i]
                newbuffer[2 * i + 1] = buffer[i]
        if newbuffer.shape[0] > self.clip_len:
            time_index = np.random.randint(newbuffer.shape[0] - self.clip_len)
            newbuffer = newbuffer[time_index:time_index + self.clip_len, :, :, :]

        return newbuffer

    def _flip(self, buffer):
        if random.random() < 0.5:
            buffer = np.flip(buffer, 2)
        return buffer

    def _get_spatial_patch(self, buffer, crop_size):
        if self.train: # Crop and Resize
            new_buffer = np.empty((buffer.shape[0], crop_size, crop_size, 3), np.dtype('float32'))

            crop_size0 = np.random.randint(self.args.rand_crop_size_min, self.args.rand_crop_size_max + 1)
            if crop_size0 > min(buffer.shape[1], buffer.shape[2]):
                crop_size0 = min(buffer.shape[1], buffer.shape[2])
            crop_size1 = crop_size0

            crop_size_max = np.maximum(crop_size0, crop_size1)
            height_index0 = np.random.randint(crop_size_max // 2, buffer.shape[1] - crop_size_max // 2 + 1)
            width_index0 = np.random.randint(crop_size_max // 2, buffer.shape[2] - crop_size_max // 2 + 1)
            height_index1 = height_index0
            width_index1 = width_index0

            t_weight = np.arange(buffer.shape[0]) / (buffer.shape[0] - 1)
            crop_size_t = (1 - t_weight) * crop_size0 + t_weight * crop_size1
            height_index_t = np.around(
                (1 - t_weight) * height_index0 + t_weight * height_index1 - (crop_size_t / 2) + 1e-6).astype(int)
            width_index_t = np.around(
                (1 - t_weight) * width_index0 + t_weight * width_index1 - (crop_size_t / 2) + 1e-6).astype(int)
            crop_size_t = np.around(crop_size_t).astype(int)

            for t in range(buffer.shape[0]):
                new_buffer[t, :, :, :] = cv2.resize(buffer[t, height_index_t[t]:height_index_t[t] + crop_size_t[t],
                                                    width_index_t[t]:width_index_t[t] + crop_size_t[t], :],
                                                    (crop_size, crop_size))
        else:  # Center Crop
            height_index = (buffer.shape[1] // 2) - (crop_size // 2)
            width_index = (buffer.shape[2] // 2) - (crop_size // 2)
            new_buffer = buffer[:, height_index:height_index + crop_size, width_index:width_index + crop_size, :]

        return new_buffer

    def _get_spatial_patch30(self, buffer):
        width_size = int(1.5 * self.crop_size)
        height_index = (buffer.shape[1] // 2) - (self.crop_size // 2)
        width_index = (buffer.shape[2] // 2) - (width_size // 2)

        buffer = buffer[:,
                 height_index:height_index + self.crop_size,
                 width_index:width_index + width_size, :]

        return buffer

    def _rand_aug(self, buffer):
        return self.randaug(buffer)

    def _normalize_and_make_tensor(self, buffer):
        buffer = (buffer - 128.) / 128
        return buffer.transpose((3, 0, 1, 2))  # [T H W C] --> [C T H W] for PyTorch

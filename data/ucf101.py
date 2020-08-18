import os
from data import videodata

# filesystem code for UCF101 dataset (VIPriors Challenge)
# written by Taeoh Kim (kto@yonsei.ac.kr)

class UCF101(videodata.VideoData):
    def __init__(self, args, train=True):
        super(UCF101, self).__init__(args, train)
        # If "batch size" increases, iteration per epoch decreases
        # args.sameiter option makes model with fixed iterations per epoch
        if args.sameiter:
            self.repeat = args.batch_size
        else:
            self.repeat = 1

    def _scan(self):
        # top directory
        self.apath = self.args.dir_data + '/mod-ucf101'

        if self.train:
            if self.args.split == 'challenge':
                self.path_to_datalist = os.path.join(self.apath, 'annotations/mod-ucf101-whole.txt')
            else:
                self.path_to_datalist = os.path.join(self.apath, 'annotations/mod-ucf101-train.txt')
        else:
            if self.args.is_validate:
                self.path_to_datalist = os.path.join(self.apath, 'annotations/mod-ucf101-validation.txt')
            else: # test annotation file does not contain labels
                self.path_to_datalist = os.path.join(self.apath, 'annotations/mod-ucf101-test.txt')

        list_video = []
        list_label = []
        self.vpath = os.path.join(self.apath, 'videos')

        # read filelist text file
        with open(self.path_to_datalist, 'r') as train_f:
            lines = train_f.readlines()

        # split every line into {video_file_name, label}
        for files in range(len(lines)):
            t = lines[files].strip().split(' ')
            list_video.append(os.path.join(self.vpath, t[0]))
            if self.args.is_validate:
                list_label.append(int(t[1]) - 1)
            else:
                list_label.append(0)

        return list_video, list_label

    def __len__(self):
        if self.train:
            return len(self.videos) * self.repeat
        else:
            return len(self.videos)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.videos)
        else:
            return idx

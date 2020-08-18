import argparse

parser = argparse.ArgumentParser(description='Action Recognition Challenge')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=0, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--gpu_id', type=int, default=0, help='specify gpu ids')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='.',
                    help='dataset directory')
parser.add_argument('--out_dir', type=str, default='./output',
                    help='ouput (ex. ckpt) directory name')

# Dataset specifications
parser.add_argument('--data_train', type=str, default='UCF101',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='UCF101',
                    help='test dataset name')
parser.add_argument('--split', type=str, default='paper', choices=('paper', 'challenge'),
                    help='train-val split to use (paper | challenge)')

# Test and Debugs
parser.add_argument('--load', type=str, default=None,
                    help='load checkpoint for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--restart', action='store_true', help='for restart based fine-tuning')

# Training specifications
parser.add_argument('--is_validate', action='store_true', help='validate vs test (false)')
parser.add_argument('--backbone', type=str, default='resnet50',
                    help='backbone architecture to use (resnet50 | resnet101 | resnet152 | resnet200)')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay_type', type=str, default='cosine_warmup',
                    help='learning rate decay type to use (step | step_10_20 | cosine | cosine_warmup)')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'ADAMax', 'RMSprop', 'RAdam'),
                    help='optimizer to use (SGD | ADAM | ADAMax | RMSprop | RAdam)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight decay')

# Other Training Things
parser.add_argument('--warmup_decay', type=int, default=0,
                    help='learning rate decay per N epochs')
parser.add_argument('--lr_decay', type=int, default=10,
                    help='learning rate decay per N epochs')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay factor for step decay')

# Log specifications
parser.add_argument('--print_every', type=int, default=100,
                    help='print every N batches')
parser.add_argument('--save_every', type=int, default=10,
                    help='save model every N epochs')
parser.add_argument('--test_every', type=int, default=10,
                    help='test model every N epochs')
parser.add_argument('--test_view', type=int, default=10,
                    help='Num of View in Test')

# Data Augmentation
parser.add_argument('--clip_len', type=int, default=64,
                    help='Temporal Width for Forward Pass')
parser.add_argument('--crop_size', type=int, default=160,
                    help='spatial crop size')
parser.add_argument('--rand_crop_size_min', type=int, default=160,
                    help='minimum random crop size. It will be resized to crop_size')
parser.add_argument('--rand_crop_size_max', type=int, default=200,
                    help='maximum random crop size. It will be resized to crop_size')
parser.add_argument('--frame_sample_rate', type=int, default=1,
                    help='default frame sampling rate')

# Spatio-Temporal Data Augmentation
parser.add_argument('--rand_augmentation', action='store_true',
                    help='use color jitter augmentation')
parser.add_argument('--aug_mode', type=str, default='randaug',
                    help='augmentation mode to use (augmix | randaug)')
parser.add_argument('--aug_degree', type=int, default=1,
                    help='temporal level change polynomial degree. (0: constant | 1: linear)')
# for RandAug
parser.add_argument('--randaug_n', type=int, default=2,
                    help='number of augment operations in randaug')
parser.add_argument('--randaug_m', type=float, default=5,
                    help='augment level in randaug')
parser.add_argument('--randaug_range', type=float, default=1.0,
                    help='augment level in randaug')

# Data Mix Augmentation
parser.add_argument('--mix_type', type=str, default='none',
                    help='mix mode to use (none | cutmix | framemix | cubemix | mixup | fademixup | cutmixup | framemixup | cubemixup)')
parser.add_argument('--prob_mix', type=float, default=1.0,
                    help='mixing probability')
parser.add_argument('--mix_beta', type=float, default=1.0,
                    help='beta distribution parameter')

# Unused
parser.add_argument('--sameiter', action='store_true', help='always perform same iterations per epoch')
parser.add_argument('--fast_validate', action='store_true', help='validate subset of validation set only')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

import os
import time
import datetime

import torch
import utility
from utility import AverageMeter, plot_learning_curves, rand_bbox
from models import slowfastnet
import numpy as np

from tqdm import tqdm

class Trainer:
    def __init__(self, args, loader, start_epoch=0):
        self.args = args
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model = slowfastnet.model_select(args.backbone, class_num=101)
        # Push model to GPU
        self.model = torch.nn.DataParallel(self.model).cuda()
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
        # self.model = self.model.cuda()

        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.warmup = args.warmup_decay

        if args.load is not None:
            checkpoint = torch.load(args.load)
            self.model.load_state_dict(checkpoint['state_dict'])
            if not args.restart:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch']

        self.loss = torch.nn.CrossEntropyLoss().cuda()
        self.current_epoch = start_epoch

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        self.ckpt_dir = args.out_dir + '/checkpoint'
        self.out_dir = args.out_dir

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.final_test = True if (args.test_only and (not args.is_validate)) else False

        self.metrics = {'train_loss' : [], 'train_acc' : [], 'val_acc' : [], 'val_acc_top5': []}
        self.load_epoch = 0
        
        with open(self.args.out_dir + '/config.txt', 'a') as f:
            f.write(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def train(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.train()
        end = time.time()

        for batch, (inputs, labels, _) in enumerate(tqdm(self.loader_train)):
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            labels = labels.cuda()

            r = np.random.rand(1)
            if r < self.args.prob_mix and self.args.mix_type != 'none':
                outputs, loss, labels = utility.mix_regularization(inputs, labels, self.model, self.loss, self.args.mix_type,
                                                           self.args.mix_beta)
            else:  # no mix no out
                # compute output
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels.long())

            prec1, prec5 = utility.accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            if (batch + 1) % self.args.print_every == 0:
                print('-------------------------------------------------------')
                print_string = 'Epoch: [{0}][{1}/{2}]'.format(self.current_epoch + 1, batch + 1, len(self.loader_train))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)
                print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
                print(print_string)
                print_string = '[Training] Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg,
                    top5_acc=top5.avg)
                print(print_string)

        self.current_epoch += 1
        self.load_epoch += 1
        if self.current_epoch > self.warmup:
            self.scheduler.step()

        self.metrics['train_loss'].append(losses.avg)
        self.metrics['train_acc'].append(top1.avg)

    def test(self):
        if self.current_epoch % self.args.test_every == 0:
            batch_time = AverageMeter()
            data_time = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            self.model.eval()

            end = time.time()

            with torch.no_grad():
                for batch, (inputs, labels, filename) in enumerate(tqdm(self.loader_test)):
                    data_time.update(time.time() - end)

                    _, _, len_of_frame, height, width = inputs.size()
                    spatial_stride = (width - self.args.crop_size) // 2
                    stride = len_of_frame / 10
                    if len_of_frame <= self.args.clip_len:
                        avail_number = 0
                        new_len = len_of_frame
                    else:
                        last_subclip_end_idx = round(min(len_of_frame, len_of_frame - (stride / 2) + (self.args.clip_len / 2)))
                        last_subclip_begin_idx = last_subclip_end_idx - self.args.clip_len

                        avail_number = min(last_subclip_begin_idx, 9)
                        new_stride = last_subclip_begin_idx / float(avail_number)
                        new_len = self.args.clip_len

                    # Per View Test
                    begin_idx = 0
                    for t in range(avail_number + 1):
                        end_idx = begin_idx + new_len

                        sub_inputs_t = inputs[:, :, begin_idx:end_idx, :, :]
                        if self.args.test_view == 30:
                            begin_spatial_idx = 0
                            for st in range(3):
                                end_spatial_idx = begin_spatial_idx + self.args.crop_size
                                sub_inputs_st = sub_inputs_t[:, :, :, :, begin_spatial_idx:end_spatial_idx]
                                begin_spatial_idx = begin_spatial_idx + spatial_stride

                                sub_inputs_st = sub_inputs_st.cuda()
                                if t == 0 and st == 0:
                                    outputs = torch.nn.Softmax(dim=1)(self.model(sub_inputs_st))
                                else:
                                    outputs = outputs + torch.nn.Softmax(dim=1)(self.model(sub_inputs_st))
                        else:
                            sub_inputs_t = sub_inputs_t.cuda()
                            if t == 0:
                                outputs = torch.nn.Softmax(dim=1)(self.model(sub_inputs_t))
                            else:
                                outputs = outputs + torch.nn.Softmax(dim=1)(self.model(sub_inputs_t))

                        # idx update
                        begin_idx = round(begin_idx + new_stride)

                    if self.args.test_view == 10:
                        outputs = outputs / (avail_number + 1)
                    else:
                        outputs = outputs / (3 * (avail_number + 1))
                    labels = labels.cuda()

                    if self.final_test:
                        # Write Prediction into Text File Here
                        final_array = utility.inference(outputs.data)

                        # write [filename final_array] and [newline]
                        self.logfile.write(filename[0][-22:])
                        for tops in range(5):
                            data_msg = ' {0}'.format(final_array[tops])
                            self.logfile.write(data_msg)
                        self.logfile.write('\n')
                    else:
                        # measure accuracy and record loss
                        prec1, prec5 = utility.accuracy(outputs.data, labels, topk=(1, 5))
                        top1.update(prec1.item(), inputs.size(0))
                        top5.update(prec5.item(), inputs.size(0))
                        batch_time.update(time.time() - end)
                        end = time.time()

            if self.args.is_validate:
                print('----Validation Results Summary----')
                print_string = 'Epoch: [{0}]'.format(self.current_epoch)
                print(print_string)
                print_string = '----------------------------- Top-1 accuracy: {top1_acc:.2f}%'.format(top1_acc=top1.avg)
                print(print_string)
                print_string = '----------------------------- Top-5 accuracy: {top5_acc:.2f}%'.format(top5_acc=top5.avg)
                print(print_string)

            # save model per epoch
            if not self.args.test_only:
                if self.current_epoch % self.args.save_every == 0:
                    torch.save({'epoch': self.current_epoch, 'state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict()},
                               self.ckpt_dir + '/model_epoch' + str(self.current_epoch).zfill(3) + '.pth')
            self.metrics['val_acc'].append(top1.avg)
            self.metrics['val_acc_top5'].append(top5.avg)
        else:
            self.metrics['val_acc'].append(0.)
            self.metrics['val_acc_top5'].append(0.)

        # Write logs
        if not self.args.test_only:
            with open(self.args.out_dir + '/log_epoch.csv', 'a') as epoch_log:
                if not self.args.load:
                    epoch_log.write('{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n'.format(
                            self.current_epoch, self.metrics['train_loss'][self.current_epoch-1], 
                            self.metrics['train_acc'][self.current_epoch-1], 
                            self.metrics['val_acc'][self.current_epoch-1], self.metrics['val_acc_top5'][self.current_epoch-1]))
                    plot_learning_curves(self.metrics, self.current_epoch, self.args)
                else:
                    epoch_log.write('{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n'.format(
                            self.current_epoch, self.metrics['train_loss'][self.load_epoch-1], 
                            self.metrics['train_acc'][self.load_epoch-1], 
                            self.metrics['val_acc'][self.load_epoch-1], self.metrics['val_acc_top5'][self.load_epoch-1]))
                    plot_learning_curves(self.metrics, self.load_epoch, self.args)
     
    def terminate(self):
        if self.args.test_only:
            if self.final_test:
                self.logfile = open(self.out_dir + '/submission.txt', 'w')
                print('----------------------------- Test Begin')
            self.test()
            if self.final_test:
                self.logfile.close()
                print('----------------------------- Test Complete')
            return True
        else:
            return self.current_epoch >= self.args.epochs

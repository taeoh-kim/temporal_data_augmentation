import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

def rand_bbox(size, lam, mix):
    T = size[2]
    W = size[3]
    H = size[4]

    if mix in ['cutmix', 'cutmixup']:
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbt1 = 0
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbt2 = T
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
    elif mix in ['framemix', 'framemixup']:
        cut_rat = 1. - lam
        cut_t = np.int(T * cut_rat)

        ct = np.random.randint(T)

        bbt1 = np.clip(ct - cut_t // 2, 0, T)
        bbx1 = 0
        bby1 = 0
        bbt2 = np.clip(ct + cut_t // 2, 0, T)
        bbx2 = W
        bby2 = H
    else: # spatio-temporal, cubemix
        cut_rat = np.power(1. - lam, 1./3.)
        cut_t = np.int(T * cut_rat)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        ct = np.random.randint(T)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbt1 = np.clip(ct - cut_t // 2, 0, T)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbt2 = np.clip(ct + cut_t // 2, 0, T)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbt1, bbx1, bby1, bbt2, bbx2, bby2

def mix_regularization(inputs, labels, model, loss, mix_type, beta=1.0):
    if mix_type in ['mixup', 'cutmix', 'framemix', 'cubemix', 'framemixup', 'fademixup', 'cutmixup', 'cubemixup']:
        # Sample Mix Ratio (Lambda)
        lam = np.random.beta(beta, beta)

        # Random Mix within Batch
        rand_index = torch.randperm(inputs.size()[0]).cuda()

        if mix_type in ['cutmix', 'framemix', 'cubemix', 'cutmixup', 'framemixup', 'cubemixup']:
            # Sample Mixing Coordinates
            bbt1, bbx1, bby1, bbt2, bbx2, bby2 = rand_bbox(inputs.size(), lam, mix_type)
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbt2 - bbt1) * (bbx2 - bbx1) * (bby2 - bby1) / (
                    inputs.size()[-1] * inputs.size()[-2] * inputs.size()[-3]))

            if mix_type in ['cutmixup', 'framemixup', 'cubemixup']:
                lamt = np.random.beta(2.0, 2.0)
                mix_tmp = inputs * lamt + inputs[rand_index] * (1. - lamt)
                fr = np.random.rand(1)
                if fr < 0.5:  # Basic MixUp, 0.5 Prob FrameMixUp
                    if lamt >= 0.5:
                        mix_tmp[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbt1:bbt2, bbx1:bbx2,
                                                                         bby1:bby2]
                        lam = lamt * lam
                    else:
                        mix_tmp[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = inputs[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2]
                        lam = lamt * lam + (1 - lam)
                else:
                    lam = lamt
                inputs = mix_tmp
            else:
                # Mix
                inputs[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbt1:bbt2, bbx1:bbx2,
                                                                bby1:bby2]
        else:  # mixup: blending two videos
            if mix_type in ['mixup']:
                inputs = inputs * lam + inputs[rand_index] * (1. - lam)
            elif mix_type in ['fademixup']:  # temporal-mix-up
                adj = np.random.choice([-1, 1]) * np.random.uniform(0, min(lam, 1.0 - lam))
                fade = np.linspace(lam - adj, lam + adj, num=inputs.size(2))
                for taxis in range(inputs.size(2)):
                    inputs[:, :, taxis, :, :] = inputs[:, :, taxis, :, :] * fade[taxis] + inputs[rand_index, :, taxis,
                                                                                          :, :] * (1. - fade[taxis])
        outputs = model(inputs)
        loss = loss(outputs, labels.long()) * lam + loss(outputs, labels[rand_index].long()) * (
                1. - lam)
        labels = labels if lam >= 0.5 else labels[rand_index]
    else:
        print('mixtype error')
        return

    return outputs, loss, labels



""" 
Gradually warm-up(increasing) learning rate in optimizer.
Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

Code Reference: https://github.com/ildoonet/pytorch-gradual-warmup-lr
"""

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

""" 
On the Variance of the Adaptive Learning Rate and Beyond (https://arxiv.org/abs/1908.03265)
Code Reference: https://github.com/LiyuanLucasLiu/RAdam
"""
class RAdam(optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss

def plot_learning_curves(metrics, cur_epoch, args):
    x = np.arange(1, cur_epoch+1)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ln1 = ax1.plot(x, metrics['train_loss'], color='tab:red')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ln3 = ax2.plot(x, metrics['train_acc'], color='tab:blue')
    ln4 = ax2.plot(x, metrics['val_acc'], color='tab:blue', linestyle='dashed')
    lns = ln1+ln3+ln4
    plt.legend(lns, ['Train loss','Train accuracy','Validation accuracy'])
    plt.tight_layout()
    plt.savefig(args.out_dir + '/learning_curve.png', bbox_inches='tight')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def inference(output, topk=(5,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    final_array = pred.squeeze(0).cpu().numpy() + 1

    return final_array

def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args.optimizer == 'ADAMax':
        optimizer_function = optim.Adamax
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': 1e-08}
    elif args.optimizer == 'RAdam':
        optimizer_function = RAdam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
    elif args.decay_type == 'cosine':
        scheduler = lrs.CosineAnnealingLR(
            my_optimizer,
            T_max=args.epochs
        )
    elif args.decay_type == 'cosine_warmup':
        cosine_scheduler = lrs.CosineAnnealingLR(
            my_optimizer,
            T_max=args.epochs
        )
        scheduler = GradualWarmupScheduler(
            my_optimizer,
            multiplier=1,
            total_epoch=args.epochs//10,
            after_scheduler=cosine_scheduler
        )
    else:
        raise Exception('unknown lr scheduler: {}'.format(args.decay_type))
    
    return scheduler
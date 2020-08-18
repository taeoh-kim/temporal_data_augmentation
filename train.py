import torch
from trainer import Trainer
from debugger import Debugger
from option import args
import data

torch.manual_seed(args.seed)


def main():
    torch.cuda.set_device(args.gpu_id)
    loader = data.Data(args)  # ex. UCF101
    start_epoch = 0

    my_trainer = Trainer(args, loader, start_epoch)

    print('trainer is ready')

    while not my_trainer.terminate():
        my_trainer.train()
        my_trainer.test()


if __name__ == "__main__":
    main()

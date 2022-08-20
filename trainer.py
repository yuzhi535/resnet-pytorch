import random
import torch
import os
import torchmetrics
from argparse import ArgumentParser
import torch.nn as nn
from tqdm import tqdm
from utils.dataloader import get_CIFAdataset_loader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from networks.resnet import Resnet, Restnet34
from networks.vgg import VGG16

# 初始化权重
def weights_init_normal(m, bias=0.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)


def weights_init_xavier(m, bias=0.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, bias)


def weights_init_kaiming(m, bias=0.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, bias)


def weights_init_orthogonal(m, bias=0.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, bias)
    elif isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, bias)


def init_weights(model, init_type='normal', bias=0.0):
    if init_type == 'normal':
        init_function = weights_init_normal
    elif init_type == 'xavier':
        init_function = weights_init_xavier
    elif init_type == 'kaiming':
        init_function = weights_init_kaiming
    elif init_type == 'orthogonal':
        init_function = weights_init_orthogonal
    else:
        raise NotImplementedError(
            'initialization method [%s] is not implemented' % init_type)

    if isinstance(model, list):
        for m in model:
            init_weights(m, init_type)
    else:
        for m in model.modules():
            init_function(m, bias=bias)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def arg_parser():
    parser = ArgumentParser()

    parser.add_argument('--batch-size', '-bs', type=int,
                        default=16, required=True, help='input batch size')
    parser.add_argument('--num-workers', '-nw',  type=int,
                        default=4, required=True, help='number of workers')
    # parser.add_argument('--resume', '-r', type=str,
    #                     required=False, help='resume a train')
    parser.add_argument('--device', type=str,
                        help='gpu or cpu', choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--num-classes', '-nc', type=int,
                        help='number of classes', required=True)
    parser.add_argument('--lr', '-lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int,
                        required=True,  help='num of epochs')

    args = parser.parse_args()
    return args


def train_fn(net, dataloader, opt, device, criterion, writer, epoch):
    net.train()
    train_loss = 0
    criterion.to(device)
    for idx, (input, target) in dataloader:
        input = input.to(device)
        target = target.to(device)

        opt.zero_grad()
        pred = net(input)

        loss = criterion(pred, target)

        train_loss += loss.item()
        loss.backward()
        opt.step()

        cur_loss = train_loss/(idx+1)

        dataloader.set_postfix(loss=cur_loss)
        writer.add_scalar('training loss',
                          cur_loss,
                          epoch*len(dataloader)+idx)


def val_fn(net, dataloader, device, num_classes, writer, epoch: int):
    net.eval()
    metric = torchmetrics.Accuracy(numClass=num_classes).to(device)
    with torch.no_grad():
        for idx, (input, target) in dataloader:
            input = input.to(device)
            target = target.to(device)
            pred = net(input)
            acc = metric.update(pred, target)
    acc = metric.compute()
    writer.add_scalar('val_acc', acc, epoch*len(dataloader)+idx)
    return acc


def train(net, opt, epochs, batch_size, num_workers, device, num_classes, model='Resnet', scheduler=None):

    train_dataloader, val_dataloader, _ = get_CIFAdataset_loader(
        root='./data/CIFA', batch_size=batch_size, num_workers=num_workers, pin_memory=True, valid_rate=0.2, shuffle=True)

    # 模型权重位置
    model_path = 'runs'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    save_path = os.path.join(model_path, model)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    log_dir = os.path.join(model_path,  model, 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    writer = SummaryWriter(log_dir)

    net.to(device)
    init_weights(net, 'kaiming')

    best = 0.0

    early_stop_step = 0
    early_stop_limit = 15

    for idx in range(epochs):
        train_loop = tqdm(enumerate(train_dataloader),
                          total=len(train_dataloader), leave=True)
        train_loop.set_description(f'epoch: {idx}/{epochs}')

        train_fn(net=net, opt=opt,
                 dataloader=train_loop, device=device,
                 criterion=nn.CrossEntropyLoss(),
                 writer=writer, epoch=idx,
                 )

        val_loop = tqdm(enumerate(val_dataloader),
                        total=len(val_dataloader), leave=True)

        score = val_fn(net=net, dataloader=val_loop,
                       device=device, num_classes=num_classes, writer=writer, epoch=idx)

        print(f'acc={score}, best acc is {max(score, best)}')

        if (score > best):
            torch.save({
                'epoch': idx,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }, os.path.join(save_path, f'epoch={idx}-miou={score:.4f}.pth'))
            best = score
            early_stop_step = 0
        else:
            if early_stop_step > early_stop_limit:
                print(f'因为已经有{early_stop_limit}轮没有提升，训练提前终止')
                writer.close()
                break
            early_stop_step += 1

        if scheduler:
            writer.add_scalar(
                "lr", scheduler.get_last_lr()[-1]
            )
            scheduler.step()
    writer.close()


if __name__ == '__main__':
    # args = arg_parser()
    bs = 32  # args.batch_size
    num_workers = 2  # args.num_workers
    device = 'cpu'  # args.device
    num_classes = 10  # args.num_classes
    lr = 0.001  # args.num_classes
    epochs = 10  # args.epochs
    # net = Resnet(num_classes, [3, 4, 6, 3], [16, 32, 64, 128])
    net = VGG16()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    seed_everything(42)
    train(net=net, epochs=epochs, batch_size=bs, num_workers=num_workers,
          device=device, num_classes=num_classes, opt=opt)


import argparse
from pydoc import Helper
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from dataloader import get_CIFAdataset_loader
from network import Restnet34
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint


def arg_parser():
    parser = argparse.ArgumentParser()

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


# 配置训练步骤
class MyResnet(pl.LightningModule):
    def __init__(self, net, train_dataloader, val_dataloader, test_dataloader,num_classes=10,
                 lr=1e-4, ) -> None:
        super(MyResnet, self).__init__()
        self.net = net
        # self.save_hyperparameters(ignore=['criterion', 'net'])
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.acc = torchmetrics.Accuracy(num_classes=num_classes)
        
        self.train_dataloader, self.val_dataloader, self.test_dataloader = train_dataloader, val_dataloader, test_dataloader

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y,  = batch

        x = x.to(self.device)
        y = y.to(self.device)
        x = self(x)
        loss = self.criterion(x, y)
        self.log('loss', loss,
                 logger=True, enable_graph=True)
        log_dict = {'train_loss': loss}
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y, = batch

        x = x.to(self.device)
        y = y.to(self.device)
        pred = self(x)
        self.acc.update(pred, y)
        loss = self.criterion(x, y)
        self.log('eval_loss', loss, enable_graph=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        acc = self.acc.compute()
        self.log('acc', acc, enable_graph=True)
        self.acc.reset()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(),
                               lr=self.lr)
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        # opt, T_max=self.epochs, eta_min=1e-5)
        # sch = torch.optim.lr_scheduler.StepLR(
        # opt, step_size=5, gamma=0.9, last_epoch=-1, verbose=False)
        # return [opt], [sch]
        return opt

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader
    
    def test_dataloader(self):
        return self.test_dataloader



def train(epochs, bs, nw, lr, num_classes, device):

    pl.seed_everything(42)

    net = Restnet34(num_classes)
    
    # 设置数据集加载，由于CIFA-10本身pytorch自带有，所以这里比较简单
    train_dataloader, val_dataloader, test_dataloader = \
        get_CIFAdataset_loader(
            root='./data/CIFA', batch_size=bs, num_workers=nw, pin_memory=True, valid_rate=0.2, shuffle=True)
    
    
    model = MyResnet(net=net, num_classes=num_classes,
                     lr=lr, train_dataloader=train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader)


    early_stop_callback = EarlyStopping(
        monitor="acc", min_delta=0.0, patience=30, verbose=True, mode="max")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = ModelCheckpoint(
        save_top_k=5, monitor='acc', mode='max', filename="{epoch:03d}-{eval_miou:.4f}", )

    trainer = pl.Trainer(max_epochs=epochs,
                         log_every_n_steps=10,
                         benchmark=True,
                         check_val_every_n_epoch=1,
                         gradient_clip_val=0.5,
                         devices=1,
                         accelerator=device,
                         callbacks=[
                             checkpoint_callback,
                             lr_monitor,
                             early_stop_callback,
                         ],
                         )
    
    trainer.logger._default_hp_metric = None

    resume_path = None
    
    trainer.fit(model=model, ckpt_path=resume_path)


if __name__ == '__main__':
    args = arg_parser()
    bs = args.batch_size
    num_workers = args.num_workers
    device = args.device
    num_classes = args.num_classes
    lr = args.num_classes
    epochs = args.epochs

    train(epochs, bs, num_workers, lr, num_classes, device=device)

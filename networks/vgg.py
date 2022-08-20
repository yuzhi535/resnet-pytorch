import torch.nn as nn
import torch
from einops.layers.torch import Rearrange


class Conv(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=1, stride=1, padding=1) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
    

class VGG16(nn.Module):
    
    '''
    dataset: cifar10
    input-size: batch_size x 3 x 32 x 32
    output-size: batch_size x 10
    '''
    
    def __init__(self) -> None:
        super().__init__()
        
        self.net1 = nn.Sequential(
            Conv(3, 64, kernel_size=3, stride=1, padding=1),
            Conv(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.net2 = nn.Sequential(
            Conv(64, 128, kernel_size=3, stride=1, padding=1),
            Conv(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.net3 = nn.Sequential(
            Conv(128, 256, kernel_size=3, stride=1, padding=1),
            Conv(256, 256, kernel_size=3, stride=1, padding=1),
            Conv(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.net4 = nn.Sequential(
            Conv(256, 512, kernel_size=3, stride=1, padding=1),
            Conv(512, 512, kernel_size=3, stride=1, padding=1),
            Conv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.net5 = nn.Sequential(
            Conv(512, 512, kernel_size=3, stride=1, padding=1),
            Conv(512, 512, kernel_size=3, stride=1, padding=1),
            Conv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifer = nn.Sequential(
            nn.AvgPool2d(kernel_size=1, stride=1),
            Rearrange('b c h w -> b(c h w)'),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        return self.classifer(self.net5(self.net4(self.net3(self.net2(self.net1(x))))))

if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    net = VGG16()
    print(net(x).shape)
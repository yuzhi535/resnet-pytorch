import torch.nn as nn
import torch
from einops.layers.torch import Rearrange

class Conv(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=1, stride=1, padding=1) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


'''
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)

'''


class Alexnet(nn.Module):
    '''
    input size: batch size x 3 x 224 x 224
    output size: batch size x num_classes
    '''

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            Conv(3, 64, kernel_size=7, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Sequential(
            Conv(64, 192, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Sequential(
            Conv(192, 384, kernel_size=3, padding=1),
            Conv(384, 256, kernel_size=3, padding=1),
            Conv(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        self.dense = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        out = self.conv1(x)
        print(out.shape)
        out = self.conv2(out)
        print(out.shape)
        out = self.conv3(out)

        out = self.dense(out)

        return out


if __name__ == '__main__':
    net = Alexnet()
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    print(out.shape)

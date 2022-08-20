import torch.nn as nn
import torch


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


class Alexnet(nn.Module):
    '''
    input size: batch size x 3 x 224 x 224
    output size: batch size x num_classes
    '''

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            Conv(3, 64, kernel_size=11, stride=4, padding=2),
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
            nn.Linear(256 * 6 * 6, 4096),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.view(out.size(0), -1)

        out = self.dense(out)

        return out


if __name__ == '__main__':
    net = Alexnet()
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    print(out.shape)

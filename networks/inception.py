import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    '''
    conv+bn+relu
    '''

    def __init__(self, in_chan, out_chan, kernel_size=1, stride=1, padding=0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size, stride, padding),
            nn.BatchNorm2d(out_chan, eps=0.001),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class InceptionBlockx1(nn.Module):
    def __init__(self, in_chan, out_chan_pooling) -> None:
        super().__init__()
        self.branch1 = Conv(in_chan, 64)
        self.branch2 = nn.Sequential(
            Conv(in_chan, 48, kernel_size=1),
            Conv(48, 64, kernel_size=5, padding=2),
        )

        self.branch3 = nn.Sequential(
            Conv(in_chan, 64, kernel_size=1),
            Conv(64, 96, kernel_size=3, padding=1),
            Conv(96, 96, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(in_chan, out_chan_pooling, kernel_size=1),
        )

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(
            x), self.branch3(x), self.branch4(x)], dim=1)
        return out


class InceptionBlockx2(nn.Module):
    def __init__(self, in_chan) -> None:
        super().__init__()
        self.branch1 = Conv(in_chan, 384, 3, 2)

        self.branch2 = nn.Sequential(
            Conv(in_chan, 64, kernel_size=1),
            Conv(64, 96, kernel_size=3, padding=1),
            Conv(96, 96, kernel_size=3, stride=2),
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class InceptionBlockx3(nn.Module):
    def __init__(self, in_chan, internal_chan) -> None:
        super().__init__()

        self.branch1 = Conv(in_chan, 192, 1)

        self.branch2 = nn.Sequential(
            Conv(in_chan, internal_chan, 1),
            Conv(internal_chan, internal_chan, [1, 7], padding=[0, 3]),
            Conv(internal_chan, 192, [7, 1], padding=[3, 0]),
        )

        self.branch3 = nn.Sequential(
            Conv(in_chan, internal_chan, 1),
            Conv(internal_chan, internal_chan, [7, 1], padding=[3, 0]),
            Conv(internal_chan, internal_chan, [1, 7], padding=[0, 3]),
            Conv(internal_chan, internal_chan, [7, 1], padding=[3, 0]),
            Conv(internal_chan, internal_chan, [1, 7], padding=[0, 3]),
            Conv(internal_chan, 192, [1, 7], padding=[0, 3]),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(in_chan, 192, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class InceptionBlockx4(nn.Module):
    def __init__(self, in_chan, ) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            Conv(in_chan, 192, kernel_size=1),
            Conv(192, 320, kernel_size=3, stride=2),
        )

        self.branch2 = nn.Sequential(
            Conv(in_chan, 192, kernel_size=1),
            Conv(192, 192, kernel_size=[1, 7], padding=[0, 3]),
            Conv(192, 192, kernel_size=[7, 1], padding=[3, 0]),
            Conv(192, 192, kernel_size=3, stride=2),
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class InceptionBlockx5(nn.Module):
    def __init__(self, in_chan) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            Conv(in_chan, 320, kernel_size=1),
        )

        self.branch2x1 = Conv(in_chan, 384, kernel_size=1)
        self.branch2x2 = Conv(384, 384, kernel_size=[1, 3], padding=[0, 1])
        self.branch2x3 = Conv(384, 384, kernel_size=[3, 1], padding=[1, 0])

        self.branch3x1 = Conv(in_chan, 448, kernel_size=1)
        self.branch3x2 = Conv(448, 384, kernel_size=3, stride=1, padding=1)
        self.branch3x3 = Conv(384, 384, kernel_size=[1, 3], padding=[0, 1])
        self.branch3x4 = Conv(384, 384, kernel_size=[3, 1], padding=[1, 0])

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(in_chan, 192, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2x1(x)
        out2 = torch.cat(
            [self.branch2x2(out2), self.branch2x3(out2)], dim=1)
        out3 = self.branch3x1(x)
        out3 = self.branch3x2(out3)
        out3 = torch.cat([self.branch3x3(out3), self.branch3x4(out3)], dim=1)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)


class GoogleNet(nn.Module):
    def __init__(self, nc: int) -> None:
        super().__init__()
        self.nc = nc

        self.conv1 = nn.Sequential(
            Conv(3, 32, kernel_size=3, stride=2),
            Conv(32, 32, kernel_size=3),
            Conv(32, 64, kernel_size=3, padding=1),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Sequential(
            Conv(64, 80, kernel_size=1),
            Conv(80, 192, kernel_size=3, stride=1),
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.mixer1 = nn.Sequential(
            InceptionBlockx1(192, 32),
            InceptionBlockx1(256, 64),
            InceptionBlockx1(288, 64),
        )

        self.mixer2 = nn.Sequential(
            InceptionBlockx2(288),
        )
        self.mixer3 = nn.Sequential(
            InceptionBlockx3(768, 128),
            InceptionBlockx3(768, 160),
            InceptionBlockx3(768, 160),
            InceptionBlockx3(768, 192),
        )

        self.mixer4 = nn.Sequential(
            InceptionBlockx4(768),
            InceptionBlockx5(1280),
            InceptionBlockx5(2048),
        )

        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(2048, self.nc)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            stddev = float(m.stddev) if hasattr(
                m, "stddev") else 0.1  # type: ignore
            torch.nn.init.trunc_normal_(
                m.weight, mean=0.0, std=stddev, a=-2, b=2)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.maxpool2(out)
        out = self.mixer1(out)
        # print(f'out1: {out.shape}')
        out = self.mixer2(out)
        out = self.mixer3(out)
        out = self.mixer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    net = GoogleNet(nc=1000)
    print(net(x).shape)

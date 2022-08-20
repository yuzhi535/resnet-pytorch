import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_


# 基本块
class Conv(nn.Module):
    def __init__(self, in_chan, out_chan, stride) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class BasicLayerX2(nn.Module):
    def __init__(self, in_chan, stride, down_sample: bool) -> None:
        super().__init__()
        out_chan = in_chan
        in_chan = in_chan // 2 if down_sample else in_chan

        print(f'in_chan={in_chan}, out_chan={out_chan}')

        self.conv = nn.Sequential(
            Conv(in_chan, out_chan, stride),
            Conv(out_chan, out_chan, 1),
        )

        self.shortcut = None
        self.act = nn.ReLU()

        if down_sample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, 1, stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        conv = self.conv(x)
        shortcut = self.shortcut(x)
        return self.act(conv+shortcut)


class BasicLayerX3(nn.Module):
    def __init__(self, in_chan, stride, down_sample: bool) -> None:
        super().__init__()
        out_chan = in_chan
        in_chan = in_chan // 2 if down_sample else in_chan
        self.conv = nn.Sequential(
            Conv(in_chan, out_chan, stride),
            Conv(out_chan, out_chan, stride),
        )

        self.shortcut = None
        self.act = nn.ReLU()

        if down_sample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, 1, stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x)+self.shortcut(x))


class Resnet(nn.Module):
    def __init__(self, num_classes, num_layers: list, chan: list) -> None:
        super().__init__()

        # 原论文是先来个7X7卷积，然后下采样，但是考虑数据集为CIFA-10，就稍微改了一下，希望影响不会太大
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, chan[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(chan[0]),
        )

        self.net = nn.ModuleList()

        for i in range(len(num_layers)):
            # 除了第一层，其他层都是先下采样的block，接着普通block
            self.net += self._make_block(num_layers[i],
                                         chan[i], 2 if i != 0 else 1)

        self.fc = nn.Linear(chan[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _make_block(self, num_layer, chan, stride):
        layer = []
        is_down = False

        strides = [stride] + [1] * (num_layer - 1)

        for stride in strides:
            if stride == 2:
                is_down = True
            layer.append(BasicLayerX2(chan, stride, is_down))
            is_down = False

        return layer

    def forward(self, x):
        out = self.first_conv(x)
        for net in self.net:
            out = net(out)

        out = F.avg_pool2d(out, out.shape[2])
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return out


class Restnet34(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.net = Resnet(num_classes, [3, 4, 6, 3], [16, 32, 64, 128])

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)

    net = Restnet34(10)
    out = net(x)
    print(out.shape)

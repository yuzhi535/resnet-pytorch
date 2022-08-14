import torch
import torch.nn as nn


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
    def __init__(self, conv_index, in_chan, out_chan, stride, down_sample: bool) -> None:
        super().__init__()
        self.conv = nn.ModuleList()

        for i in range(conv_index):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(in_chan, out_chan, 3, stride, 1),
                    nn.BatchNorm2d(out_chan),
                    nn.ReLU(),
                )
            )

        self.shortcut = nn.Sequential()

    def forward(self, x):
        return x


class BasicLayerX3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        return x

class Resnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

    def _make_layer(self, ):

        pass

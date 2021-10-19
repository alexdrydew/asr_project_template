from torch import nn
from torch.nn import Sequential, Conv1d, BatchNorm1d, ReLU

from hw_asr.base import BaseModel


def _get_separable_conv_bn(in_channels, out_channels, kernel_size, stride=1, dilation=1):
    return Sequential(
        Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            padding=kernel_size // 2,
            stride=stride,
            dilation=dilation
        ),
        Conv1d(in_channels, out_channels, kernel_size=1),
        BatchNorm1d(out_channels)
    )


def _get_conv_bn(in_channels, out_channels, kernel_size, stride=1, dilation=1):
    return Sequential(
        Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=kernel_size // 2),
        BatchNorm1d(out_channels)
    )


class QuartzNet(BaseModel):

    class Block(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, R):
            super().__init__()

            self.activation = nn.ReLU()

            self.net = [_get_separable_conv_bn(in_channels, out_channels, kernel_size)]
            for i in range(1, R):
                self.net.append(self.activation)
                self.net.append(_get_separable_conv_bn(out_channels, out_channels, kernel_size))

            self.net = Sequential(*self.net)

            self.residual_net = Sequential(
                Conv1d(in_channels, out_channels, kernel_size=1),
                BatchNorm1d(out_channels)
            )
            self.activation = nn.ReLU()

        def forward(self, x):
            return self.activation(self.net(x) + self.residual_net(x))

    def __init__(self, n_feats, n_class, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)

        self.c_1 = _get_separable_conv_bn(n_feats, 256, 33, stride=2)
        self.b_1 = QuartzNet.Block(256, 256, 33, R=5)
        self.b_2 = QuartzNet.Block(256, 256, 39, R=5)
        self.b_3 = QuartzNet.Block(256, 512, 51, R=5)
        self.b_4 = QuartzNet.Block(512, 512, 63, R=5)
        self.b_5 = QuartzNet.Block(512, 512, 75, R=5)
        self.c_2 = _get_separable_conv_bn(512, 512, 87)
        self.c_3 = _get_separable_conv_bn(512, 1024, 1)
        self.c_4 = Conv1d(1024, n_class, 1, dilation=2)

        self.activation = ReLU()

    def forward(self, spectrogram, *args, **kwargs):
        spectrogram = spectrogram.transpose(1, 2)
        x = self.activation(self.c_1(spectrogram))
        x = self.b_1(x)
        x = self.b_2(x)
        x = self.b_3(x)
        x = self.b_4(x)
        x = self.b_5(x)
        x = self.activation(self.c_2(x))
        x = self.activation(self.c_3(x))
        logits = self.c_4(x)
        return {"logits": logits.transpose(1, 2)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2  # we don't reduce time dimension here

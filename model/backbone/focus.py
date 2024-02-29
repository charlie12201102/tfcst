import torch
from torch import nn


class Focus(nn.Module):

    def __init__(self, inplanes, planes, k=1, s=1):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(in_channels=inplanes * 4, out_channels=planes, kernel_size=k, stride=1)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


if __name__ == '__main__':
    x = torch.rand((1, 3, 640, 640))
    f = Focus(3, 32)
    print(f(x).shape)

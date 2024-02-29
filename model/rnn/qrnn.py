import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F

class QRNN3DLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, act='relu'):
        super(QRNN3DLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # quasi_conv_layer

        self.conv = nn.Conv3d(in_channels, 2 * hidden_channels, (5, 3, 3), padding=(2, 1, 1))
        self.conv_h0 = nn.Conv2d(in_channels, 2 * hidden_channels, (3, 3), padding=(1, 1))
        # self.conv_h0 = BottleneckCSP(in_channels, 2*hidden_channels, 4, e=1)
        # self.rep_pad = nn.ReplicationPad3d((1, 1, 1, 1, 1, 0))
        self.act = act

    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        gates_h0 = self.conv_h0(inputs[:, :, 0, :, :]).unsqueeze(2)
        Z, F = gates.split(split_size=self.hidden_channels, dim=1)
        Zh0, Fh0 = gates_h0.split(split_size=self.hidden_channels, dim=1)

        if self.act == 'tanh':
            return Z.tanh(), F.sigmoid(), (1 - Fh0.sigmoid()) * Zh0.tanh()
        elif self.act == 'relu':
            return Z.relu(), F.sigmoid(), (1 - Fh0.sigmoid()) * Zh0.relu()
        elif self.act == 'none':
            return Z, F.sigmoid, (1 - Fh0.sigmoid()) * Zh0
        else:
            raise NotImplementedError

    def _rnn_step(self, z, f, h):
        # uses 'f pooling' at each time step
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, reverse=False):

        Z, F, h = self._conv_step(inputs)

        return torch.cat([h, F[:, :, 0:1, :, :] * h + (1 - F[:, :, 0:1, :, :]) * Z[:, :, 0:1, :, :]], dim=2)

    def extra_repr(self):
        return 'act={}'.format(self.act)


if __name__ == '__main__':
    x = torch.rand((8, 64, 5, 320, 320))
    qrnn = QRNN3DLayer(64, 64)
    res = qrnn(x)
    print(res)

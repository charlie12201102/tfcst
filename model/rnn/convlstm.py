import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

#torch.cuda.set_device(1)


class BottleneckLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, phase='train'):
        super(BottleneckLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Conv2d(in_channels=self.input_size, out_channels=self.input_size, kernel_size=3,groups=self.input_size, stride=1, padding=1)
        self.Wy = nn.Conv2d(int(self.input_size + self.hidden_size), self.hidden_size, kernel_size=1)
        self.Wi = nn.Conv2d(self.hidden_size, self.hidden_size, 3, 1, 1, groups=self.hidden_size, bias=False)
        self.Gates = nn.Conv2d(hidden_size, 4 * hidden_size, 3, padding=1)

        self.relu = nn.ReLU6()
        self.phase = phase

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size), volatile=(False, True)[self.phase=='test']),
                Variable(torch.zeros(state_size), volatile=(False, True)[self.phase=='test'])
            )
        prev_cell, prev_hidden = prev_state
        # prev_hidden_drop = F.dropout(prev_hidden, training=(False, True)[self.phase=='train'])
        # data size is [batch, channel, height, width]

        input_ = self.W(input_)
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        stacked_inputs = self.Wi(self.Wy(stacked_inputs))  # reduce to hidden layer size # depth wise 3*3

        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = self.relu(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * self.relu(cell)

        return cell, hidden

    def init_state(self, input_):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        state = (
            Variable(torch.zeros(state_size), volatile=(False, True)[self.phase == 'test']),
            Variable(torch.zeros(state_size), volatile=(False, True)[self.phase == 'test'])
        )
        return state

class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, phase='train'):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=1)
        self.phase = phase

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size), volatile=(False, True)[self.phase=='test']),
                Variable(torch.zeros(state_size), volatile=(False, True)[self.phase=='test'])
            )
        prev_cell, prev_hidden = prev_state
        # prev_hidden_drop = F.dropout(prev_hidden, training=(False, True)[self.phase=='train'])
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((F.dropout(input_, p=0.2, training=(False,True)[self.phase=='train']), prev_hidden), 1)
        # stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return cell, hidden

    def init_state(self, input_):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        state = (
            Variable(torch.zeros(state_size), volatile=(False, True)[self.phase == 'test']),
            Variable(torch.zeros(state_size), volatile=(False, True)[self.phase == 'test'])
        )
        return state



class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, cuda_flag=False, phase='train'):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, 3,
                                   padding=self.kernel_size // 2)
        self.Conv_ct = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, 3, padding=self.kernel_size // 2)
        dtype = torch.FloatTensor
        self.phase = phase

    def forward(self, input, hidden):
        if hidden is None:
            size_h = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
            if self.cuda_flag == True:
                hidden = (Variable(torch.zeros(size_h), volatile=(False, True)[self.phase=='test']).cuda(), )
            else:
                hidden = (Variable(torch.zeros(size_h), volatile=(False, True)[self.phase=='test']), )
        hidden = hidden[-1]
        c1 = self.ConvGates(torch.cat((F.dropout(input,p=0.2,training=(False,True)[self.phase=='train']), hidden), 1))
        (rt, ut) = c1.chunk(2, 1)
        reset_gate = torch.sigmoid(rt)
        update_gate = torch.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = torch.tanh(p1)
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return (next_h, )

    def init_state(self, input):
        size_h = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
        if self.cuda_flag == True:
            hidden = (Variable(torch.zeros(size_h), volatile=(False, True)[self.phase == 'test']).cuda(),)
        else:
            hidden = (Variable(torch.zeros(size_h), volatile=(False, True)[self.phase == 'test']),)
        return hidden



def _main():
    """
    Run some basic tests on the API
    """

    # define batch_size, channels, height, width
    b, c, h, w = 4, 3, 320, 320
    d = 5           # hidden state size
    lr = 1e-1       # learning rate
    T = 8           # sequence length
    max_epoch = 20  # number of epochs

    # set manual seed
    torch.manual_seed(0)

    print('Instantiate model')
    device = torch.device('cpu')
    model = ConvGRUCell(c, d)
    model.to(device)
    # model = ConvGRUCell(c, d)
    print(repr(model))

    print('Create input and target Variables')
    x = Variable(torch.rand(T, b, c, h, w))
    y = Variable(torch.randn(T, b, d, h, w))

    print('Create a MSE criterion')
    loss_fn = nn.MSELoss()

    print('Run for', max_epoch, 'iterations')
    for epoch in range(0, max_epoch):
        state = None
        loss = 0
        for t in range(0, T):
            state = model(x[t], state)
            loss += loss_fn(state[0], y[t])

        # print(' > Epoch {:2d} loss: {:.3f}'.format((epoch+1), loss.data[0]))
        print(' > Epoch {:2d} loss: {:.3f}'.format((epoch+1), loss))

        # zero grad parameters
        model.zero_grad()

        # compute new grad parameters through time!
        loss.backward()

        # learning_rate step against the gradient
        for p in model.parameters():
            p.data.sub_(p.grad.data * lr)

    print('Input size:', list(x.data.size()))
    print('Target size:', list(y.data.size()))
    print('Last hidden state size:', list(state[0].size()))


if __name__ == '__main__':
    _main()
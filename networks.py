import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, channel_compression=20):
        super(Net, self).__init__()
        self.chan_compress = channel_compression
        self.conv1 = nn.Conv2d(1, 20, 3, 2)
        self.conv2 = nn.Conv2d(20, 50, 3, 2)
        self.conv3 = nn.Conv2d(50, self.chan_compress, 1, 1)
        view = 6*6*self.chan_compress
        self.fc1 = nn.Linear(view, 10)
        self.ordered_layers = [self.conv1,
                               self.conv2,
                               self.conv3,
                               self.fc1]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 6 * 6 * self.chan_compress)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class NetPytorchExample(nn.Module):
    def __init__(self):
        super(NetPytorchExample, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
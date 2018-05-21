import torch.nn as nn
from utilities.utils import *

class CNN(nn.Module):
    def __init__(self, parameter_dict=cnn_params):
        self.channels = parameter_dict['channel_count']
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 833, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
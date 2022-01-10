import torch.nn.functional as F
from torch import nn

# FNN version
# class MyAwesomeModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.fc1 = nn.Linear(784, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, 10)

#         self.dropout = nn.Dropout(p=0.3)

#     def forward(self, x):

#         x = x.view(x.shape[0], -1)

#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.dropout(F.relu(self.fc2(x)))
#         x = self.dropout(F.relu(self.fc3(x)))

#         x = F.log_softmax(self.fc4(x), dim=1)

#         return x


# CNN version
class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.pool_1 = nn.MaxPool2d(2, 2, padding=0)
        self.dropout_1 = nn.Dropout(0.5)

        self.conv_2 = nn.Conv2d(16, 64, 3, stride=1, padding=1)
        self.pool_2 = nn.MaxPool2d(2, 2, padding=0)
        self.dropout_2 = nn.Dropout(0.4)

        self.fc_1 = nn.Linear(3136, 64)
        self.fc_2 = nn.Linear(64, 10)

    def forward(self, x):
        if (x.ndim != 3):
            raise ValueError('Expected shape to be [x, 28, 28]')
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        x = self.pool_1(F.relu(self.conv_1(x)))
        x = self.dropout_1(x)
        x = self.pool_2(F.relu(self.conv_2(x)))
        x = self.dropout_2(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc_1(x))
        x = F.log_softmax(self.fc_2(x), dim=1)

        return x

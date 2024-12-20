import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class FlappyBirdNN(nn.Module):
    def __init__(self, stack_frame_len=4, num_actions=2):
        super(FlappyBirdNN, self).__init__()

        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(in_channels=stack_frame_len, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm for first conv layer
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)  # BatchNorm for second conv layer
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)  # BatchNorm for third conv layer

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

        # Initialize weights
        # self._initialize_weights()

    def forward(self, x):
        # Apply convolutional layers with ReLU and Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def _initialize_weights(self):
        # Initialize weights for convolutional layers
        init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')

        # Initialize weights for fully connected layers
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

        # Initialize biases to zero
        init.zeros_(self.conv1.bias)
        init.zeros_(self.conv2.bias)
        init.zeros_(self.conv3.bias)
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)

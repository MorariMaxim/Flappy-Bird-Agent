import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class FlappyBirdNN(nn.Module):
    def __init__(self, stack_frame_len=4, num_actions=2):
        super(FlappyBirdNN, self).__init__()
 
        self.conv1 = nn.Conv2d(in_channels=stack_frame_len, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)  
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)  
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)  

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions) 

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x 

import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define a lightweight architecture for fast training
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)  # Reduce channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)  # Downsample
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)  # Further downsample
        self.fc1 = nn.Linear(128 * 56 * 56, 256)  # Adjust input size based on downsampling
        self.fc2 = nn.Linear(256, 200)  # Output layer for 200 classes

    def forward(self, x):
        # Define forward pass
        # B x 3 x 224 x 224
        x = self.conv1(x).relu()  # B x 32 x 224 x 224
        x = self.conv2(x).relu()  # B x 64 x 112 x 112
        x = self.conv3(x).relu()  # B x 128 x 56 x 56
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.fc1(x).relu()  # B x 256
        x = self.fc2(x)  # B x 200
        return x

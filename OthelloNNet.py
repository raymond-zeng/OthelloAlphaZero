import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class OthelloResNet(nn.Module):
    def __init__(self, num_res_blocks=10, num_hidden_channels=128):
        super(OthelloResNet, self).__init__()
        
        self.initial_conv = nn.Conv2d(3, num_hidden_channels, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(num_hidden_channels)
        
        # Residual tower
        self.res_tower = nn.Sequential(
            *[ResidualBlock(num_hidden_channels) for _ in range(num_res_blocks)]
        )
        
        # --- The Heads ---
        # Value Head
        self.value_conv = nn.Conv2d(num_hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8*8, 64) # 8*8 is the board size
        self.value_fc2 = nn.Linear(64, 1)
        
        # Policy Head
        self.policy_conv = nn.Conv2d(num_hidden_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2*8*8, 65) # 65 possible moves (64 squares + pass)

    def forward(self, x):
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        out = self.res_tower(out)
        
        # --- Value Head ---
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1) # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)) # Output scaled to [-1, 1]
        
        # --- Policy Head ---
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1) # Flatten
        policy = self.policy_fc(policy)
        
        return policy, value
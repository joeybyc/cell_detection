import torch
import torch.nn as nn


# Neural network model
class Net(nn.Module):
    def __init__(self, size_img):
        super(Net, self).__init__()
        self.size_img = size_img
        self.fc1 = nn.Linear(self.size_img * self.size_img, 512)  # Assuming images are resized to 20x20 pixels
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)  # Binary classification

    def forward(self, x):
        x = x.view(-1, self.size_img * self.size_img)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
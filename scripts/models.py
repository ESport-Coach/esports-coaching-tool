import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import glob

class LabeledDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []

        for label_dir in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_dir)
            if os.path.isdir(label_path):
                for img_path in glob.glob(f"{label_path}/*.jpg"):
                    self.data.append((img_path, label_dir))

        self.label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(data_dir)))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        label_idx = self.label_map[label]
        if self.transform:
            image = self.transform(image)
        return image, label_idx

# Define the model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)

        # Placeholder for fc1, actual size set in forward method
        self.fc1 = None  
        self.num_classes = num_classes

    def _initialize_fc1(self, x):
        """
        Dynamically initialize fc1 with the correct input size after seeing the input.
        """
        flattened_size = x.size(1) * x.size(2) * x.size(3)  # C * H * W
        self.fc1 = nn.Linear(flattened_size, 128)  # Initialize fc1
        self.fc2 = nn.Linear(128, self.num_classes)  # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        if self.fc1 is None:  # Initialize fc1 dynamically the first time
            self._initialize_fc1(x)

        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
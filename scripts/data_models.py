import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import glob
from typing import Tuple, Dict, List

class LabeledDataset(Dataset):
    def __init__(self, data_dir: str, transform: transforms.Compose = None) -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.data: List[Tuple[str, str]] = []
        
        # Collect all valid image paths and labels
        for label_dir in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_dir)
            if os.path.isdir(label_path):
                image_paths = glob.glob(os.path.join(label_path, "*.jpg"))
                self.data.extend([(img_path, label_dir) for img_path in image_paths])

        # Create label mapping
        unique_labels = sorted(set(label for _, label in self.data))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            label_idx = self.label_map[label]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label_idx
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a default item or raise the exception based on your needs
            raise

class GameStateCNN(nn.Module):
    """
    CNN for classifying game states (looting, map, interface, gameplay)
    """
    def __init__(self, num_classes: int, input_size: Tuple[int, int] = (224, 224)) -> None:
        super(GameStateCNN, self).__init__()
        
        # Calculate sizes based on input
        h, w = input_size
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate size after convolutions
        feature_size = (h // 8) * (w // 8) * 128
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)
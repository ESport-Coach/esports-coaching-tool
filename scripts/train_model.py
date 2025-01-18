import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import logging
from pathlib import Path
from typing import Tuple, Dict
import json
from datetime import datetime

from scripts.data_models import LabeledDataset, GameStateCNN

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        input_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        learning_rate: float = 0.001,
        val_split: float = 0.2
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_split = val_split
        
    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        dataset = LabeledDataset(str(self.data_dir), transform=self.transform)
        
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader
        
    def train(self, epochs: int = 15) -> Dict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        train_loader, val_loader = self._prepare_data()
        
        model = GameStateCNN(
            num_classes=len(train_loader.dataset.dataset.label_map)
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=2
        )
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = self.model_dir / f"game_state_cnn_{timestamp}.pth"
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_acc = 100. * correct / total
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Acc: {val_acc:.2f}% | "
                f"{time.time()}"
            )
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'label_map': train_loader.dataset.dataset.label_map
                }, model_save_path)
        
        # Save training history
        history_path = self.model_dir / f"training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(history, f)
            
        return history

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    trainer = ModelTrainer(
        data_dir="data/train_frames",
        model_dir="models",
        batch_size=32,
        learning_rate=0.001
    )
    
    try:
        history = trainer.train(epochs=15)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
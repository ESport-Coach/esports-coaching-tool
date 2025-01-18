import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    label: str
    confidence: float
    frame_path: str
    frame_number: Optional[int] = None

class GameStatePredictor:
    def __init__(
        self, 
        model_path: str,
        input_size: Tuple[int, int] = (224, 224),
        confidence_threshold: float = 0.7,
        batch_size: int = 32
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.input_size = input_size
        
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> None:
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model_state = checkpoint['model_state_dict']
                self.label_map = checkpoint['label_map']
            else:
                self.model_state = checkpoint
                logger.warning("Loading legacy model format - label map not included")
                
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}
            logger.info(f"Model loaded successfully with {len(self.label_map)} classes")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
            
    def _preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        try:
            image = Image.open(image_path).convert("RGB")
            return self.transform(image)
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {str(e)}")
            return None
            
    def _process_batch(self, batch_tensors: List[torch.Tensor]) -> torch.Tensor:
        batch = torch.stack(batch_tensors).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = F.softmax(outputs, dim=1)
        return probabilities
        
    def predict_frame(self, frame_path: str) -> Optional[PredictionResult]:
        """Predict the game state for a single frame."""
        tensor = self._preprocess_image(frame_path)
        if tensor is None:
            return None
            
        batch_result = self._process_batch([tensor])
        probs = batch_result[0]
        confidence, pred_idx = torch.max(probs, 0)
        confidence = confidence.item()
        
        if confidence < self.confidence_threshold:
            logger.warning(
                f"Low confidence prediction ({confidence:.2f}) for {frame_path}"
            )
            
        return PredictionResult(
            label=self.reverse_label_map[pred_idx.item()],
            confidence=confidence,
            frame_path=frame_path,
            frame_number=self._extract_frame_number(frame_path)
        )
        
    def predict_directory(
        self, 
        test_dir: str, 
        output_file: Optional[str] = None
    ) -> List[PredictionResult]:
        """Predict game states for all frames in a directory."""
        test_dir = Path(test_dir)
        image_paths = sorted(
            p for p in test_dir.glob("*.jpg")
        )
        
        if not image_paths:
            logger.warning(f"No images found in {test_dir}")
            return []
            
        results = []
        batch_tensors = []
        batch_paths = []
        
        for image_path in tqdm(image_paths, desc="Processing frames"):
            tensor = self._preprocess_image(str(image_path))
            if tensor is not None:
                batch_tensors.append(tensor)
                batch_paths.append(str(image_path))
                
                if len(batch_tensors) == self.batch_size:
                    batch_results = self._process_batch(batch_tensors)
                    results.extend(self._process_predictions(
                        batch_results, batch_paths
                    ))
                    batch_tensors = []
                    batch_paths = []
                    
        # Process remaining images
        if batch_tensors:
            batch_results = self._process_batch(batch_tensors)
            results.extend(self._process_predictions(
                batch_results, batch_paths
            ))
            
        if output_file:
            self._save_results(results, output_file)
            
        return results
        
    def _process_predictions(
        self, 
        batch_results: torch.Tensor, 
        image_paths: List[str]
    ) -> List[PredictionResult]:
        results = []
        confidences, predictions = torch.max(batch_results, 1)
        
        for conf, pred_idx, path in zip(
            confidences, predictions, image_paths
        ):
            conf_value = conf.item()
            if conf_value < self.confidence_threshold:
                logger.warning(
                    f"Low confidence prediction ({conf_value:.2f}) for {path}"
                )
                
            results.append(PredictionResult(
                label=self.reverse_label_map[pred_idx.item()],
                confidence=conf_value,
                frame_path=path,
                frame_number=self._extract_frame_number(path)
            ))
            
        return results
        
    def _extract_frame_number(self, frame_path: str) -> Optional[int]:
        """Extract frame number from filename."""
        try:
            filename = Path(frame_path).stem
            if 'frame_' in filename:
                return int(filename.split('frame_')[1])
            return None
        except:
            return None
            
    def _save_results(
        self, 
        results: List[PredictionResult], 
        output_file: str
    ) -> None:
        """Save predictions to JSON file."""
        output_data = [{
            'frame_path': r.frame_path,
            'frame_number': r.frame_number,
            'label': r.label,
            'confidence': r.confidence
        } for r in results]
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
    def analyze_predictions(
        self, 
        predictions: List[PredictionResult]
    ) -> Dict:
        """Calculate basic metrics from predictions."""
        total_frames = len(predictions)
        state_counts = {}
        state_durations = {}  # Duration in frames
        state_confidences = {}
        current_state = None
        current_state_start = 0
        
        for i, pred in enumerate(predictions):
            # Update state counts
            state_counts[pred.label] = state_counts.get(pred.label, 0) + 1
            
            # Track state confidences
            if pred.label not in state_confidences:
                state_confidences[pred.label] = []
            state_confidences[pred.label].append(pred.confidence)
            
            # Track state durations
            if pred.label != current_state:
                if current_state is not None:
                    duration = i - current_state_start
                    if current_state not in state_durations:
                        state_durations[current_state] = []
                    state_durations[current_state].append(duration)
                current_state = pred.label
                current_state_start = i
        
        # Add final state duration
        if current_state is not None:
            duration = total_frames - current_state_start
            if current_state not in state_durations:
                state_durations[current_state] = []
            state_durations[current_state].append(duration)
        
        return {
            'total_frames': total_frames,
            'state_distribution': {
                state: count / total_frames 
                for state, count in state_counts.items()
            },
            'average_confidences': {
                state: np.mean(confs) 
                for state, confs in state_confidences.items()
            },
            'average_duration': {
                state: np.mean(durations) 
                for state, durations in state_durations.items()
            },
            'max_duration': {
                state: max(durations) 
                for state, durations in state_durations.items()
            }
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    predictor = GameStatePredictor(
        model_path="models/game_state_cnn.pth",
        confidence_threshold=0.7
    )
    
    try:
        predictions = predictor.predict_directory(
            "data/test_frames",
            output_file="predictions.json"
        )
        
        metrics = predictor.analyze_predictions(predictions)
        logger.info("Prediction Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value}")
            
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
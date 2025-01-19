import cv2
import torch
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Dict, Optional
import json
import time
from PIL import Image
import numpy as np

from scripts.inference import GameStatePredictor

logger = logging.getLogger(__name__)

@dataclass
class VideoAnalysisResult:
    """Contains aggregated analysis results for a video."""
    total_frames: int
    fps: float
    duration_seconds: float
    state_frames: Dict[str, int]
    state_duration: Dict[str, float]
    state_percentages: Dict[str, float]
    processing_time: float

class VideoStreamAnalyzer:
    """Analyzes video frames in real-time without saving them to disk."""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.7,
        skip_frames: int = 0  # Skip N frames between predictions to improve performance
    ):
        self.predictor = GameStatePredictor(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        self.skip_frames = skip_frames

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert OpenCV frame to tensor for prediction."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        # Use predictor's transform
        return self.predictor.transform(pil_image).unsqueeze(0)
        
    def analyze_video(self, video_path: str, output_file: Optional[str] = None) -> VideoAnalysisResult:
        """
        Analyze video frame by frame without saving frames to disk.
        
        Args:
            video_path: Path to the video file
            output_file: Optional path to save results as JSON
            
        Returns:
            VideoAnalysisResult with analysis metrics
            
        Raises:
            ValueError: If video cannot be opened or processed
        """
        start_time = time.time()
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
            
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            # Initialize counters
            processed_frames = 0
            state_counts: Dict[str, int] = {}
            frame_count = 0
            
            logger.info(f"Starting analysis of {video_path.name}")
            logger.info(f"Video properties: {total_frames} frames, {fps} FPS, {duration:.1f} seconds")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Skip frames if specified
                if self.skip_frames > 0 and frame_count % (self.skip_frames + 1) != 0:
                    continue
                    
                # Preprocess and get prediction
                tensor = self._preprocess_frame(frame)
                with torch.no_grad():
                    probabilities = self.predictor.model(tensor.to(self.predictor.device))
                    confidence, pred_idx = torch.max(probabilities, 1)
                    confidence = confidence.item()
                    
                    if confidence >= self.predictor.confidence_threshold:
                        label = self.predictor.reverse_label_map[pred_idx.item()]
                        state_counts[label] = state_counts.get(label, 0) + 1
                        processed_frames += 1
                
                # Log progress periodically
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            # Calculate metrics
            processing_time = time.time() - start_time
            
            # Account for skipped frames in the counts
            if self.skip_frames > 0:
                for state in state_counts:
                    state_counts[state] *= (self.skip_frames + 1)
            
            # Create result object
            result = VideoAnalysisResult(
                total_frames=total_frames,
                fps=fps,
                duration_seconds=duration,
                state_frames=state_counts,
                state_duration={
                    state: count / fps 
                    for state, count in state_counts.items()
                },
                state_percentages={
                    state: (count / total_frames) * 100 
                    for state, count in state_counts.items()
                },
                processing_time=processing_time
            )
            
            if output_file:
                self._save_results(result, output_file)
            
            logger.info(f"Analysis completed in {processing_time:.1f} seconds")
            logger.info("Time spent in each state:")
            for state, duration in result.state_duration.items():
                percentage = result.state_percentages[state]
                logger.info(f"  {state}: {duration:.1f} seconds ({percentage:.1f}%)")
            
            return result
            
        finally:
            cap.release()
            
    def _save_results(self, result: VideoAnalysisResult, output_file: str) -> None:
        """Save analysis results to JSON file."""
        output_data = {
            'video_stats': {
                'total_frames': result.total_frames,
                'fps': result.fps,
                'duration_seconds': result.duration_seconds,
                'processing_time_seconds': result.processing_time
            },
            'state_analysis': {
                'frame_counts': result.state_frames,
                'durations': result.state_duration,
                'percentages': result.state_percentages
            }
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            logger.info(f"Results saved to {output_path}")

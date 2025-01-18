import cv2
import os
import json
from typing import List, Dict, Any
from pathlib import Path
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VideoAnnotation:
    video_name: str
    frame: int
    label: str

class AnnotationParser:
    def __init__(self, project_id: str = "cm5fa8szl02xx07y0gu1cedbc"):
        self.project_id = project_id

    def parse_ndjson(self, file_path: str) -> List[VideoAnnotation]:
        """
        Parse labelbox ndjson annotations file.
        
        Args:
            file_path: Path to the ndjson file
            
        Returns:
            List of VideoAnnotation objects
        
        Raises:
            FileNotFoundError: If the annotation file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        annotations = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {file_path}")
            
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        video_name = data['data_row']['external_id']
                        frame_data = data['projects'][self.project_id]['labels'][0]['annotations']['frames']

                        for frame_number, frame_info in frame_data.items():
                            for obj in frame_info['objects'].values():
                                annotations.append(VideoAnnotation(
                                    video_name=video_name,
                                    frame=int(frame_number),
                                    label=obj['value']
                                ))
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Skipping malformed annotation at line {line_num}: {str(e)}")
                        continue
                        
            logger.info(f"Successfully parsed {len(annotations)} annotations from {file_path}")
            return annotations
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON at line {e.lineno}: {str(e)}")
            raise

class FrameExtractor:
    def __init__(self, output_dir: str, fps: int = 5):
        self.output_dir = Path(output_dir)
        self.fps = fps
        
    def extract_frames(self, video_path: str) -> Path:
        """
        Extract frames from a video at specified FPS.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the output directory
            
        Raises:
            ValueError: If video file cannot be opened
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_subdir = self.output_dir / video_path.stem
        output_subdir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS) / self.fps)
            processed_frames = 0
            saved_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if processed_frames % frame_rate == 0:
                    frame_path = output_subdir / f"frame_{saved_frames:05d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    saved_frames += 1
                    
                processed_frames += 1
                
                if processed_frames % 100 == 0:
                    logger.info(f"Processed {processed_frames}/{total_frames} frames")

            logger.info(f"Extracted {saved_frames} frames to {output_subdir}")
            return output_subdir
            
        finally:
            cap.release()

if __name__ == "__main__":
    parser = AnnotationParser()
    try:
        annotations = parser.parse_ndjson("data/batch1.ndjson")
        logger.info(f"Parsed {len(annotations)} annotations.")
    except Exception as e:
        logger.error(f"Failed to parse annotations: {str(e)}")
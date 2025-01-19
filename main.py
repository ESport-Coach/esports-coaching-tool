import argparse
from pathlib import Path
import logging
import sys
import yaml
import json

from scripts.utils import FrameExtractor
from scripts.inference import GameStatePredictor
from scripts.train_model import ModelTrainer
from scripts.video_analyzer import VideoStreamAnalyzer

logger = logging.getLogger(__name__)

def setup_logging(output_dir: str) -> None:
    """Configure logging to file and stdout."""
    log_path = Path(output_dir) / 'app.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

def process_video_command(args):
    """Process video into frames."""
    extractor = FrameExtractor(args.output, fps=args.fps)
    output_dir = extractor.extract_frames(args.video)
    logger.info(f"Extracted frames to {output_dir}")
    return output_dir

def analyze_video_command(args):
    """Analyze video without saving frames to disk."""
    analyzer = VideoStreamAnalyzer(
        model_path=args.model,
        confidence_threshold=args.confidence_threshold,
        skip_frames=args.skip_frames
    )
    
    try:
        result = analyzer.analyze_video(args.video, args.output)
        logger.info("Analysis Summary:")
        logger.info(f"Video duration: {result.duration_seconds:.1f} seconds")
        logger.info(f"Processing time: {result.processing_time:.1f} seconds")
        logger.info("Time spent in each state:")
        for state, duration in result.state_duration.items():
            percentage = result.state_percentages[state]
            logger.info(f"  {state}: {duration:.1f} seconds ({percentage:.1f}%)")
    except Exception as e:
        logger.error(f"Video analysis failed: {str(e)}")
        raise

def train_model_command(args):
    """Train a new model."""
    trainer = ModelTrainer(
        data_dir=args.data,
        model_dir=Path(args.model_output).parent,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.validation_split
    )
    
    history = trainer.train(epochs=args.epochs)
    
    # Save training history
    history_path = Path(args.model_output).with_suffix('.history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
        
    logger.info(f"Model and training history saved to {args.model_output}")
    return history

def analyze_command(args):
    """Analyze gameplay using trained model."""
    frames_dir = args.frames
    
    # If video provided, process it first
    if args.video:
        extractor = FrameExtractor(args.output, fps=args.fps)
        frames_dir = extractor.extract_frames(args.video)
        
    predictor = GameStatePredictor(
        model_path=args.model,
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size
    )
    
    predictions = predictor.predict_directory(
        frames_dir,
        output_file=args.output
    )
    
    metrics = predictor.analyze_predictions(predictions)
    
    # Save metrics
    metrics_path = Path(args.output).with_suffix('.metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    logger.info("Analysis completed. Results:")
    for state, percentage in metrics['state_distribution'].items():
        logger.info(f"{state}: {percentage*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Esports Coaching Tool')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Process video command
    process_parser = subparsers.add_parser('process-video', help='Process video into frames')
    process_parser.add_argument('video', type=str, help='Path to video file')
    process_parser.add_argument('--output', type=str, default='data/test_frames',
                               help='Output directory for frames')
    process_parser.add_argument('--fps', type=int, default=5,
                               help='Frames per second to extract')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data', type=str, required=True,
                             help='Directory with training data')
    train_parser.add_argument('--model-output', type=str, required=True,
                             help='Path to save trained model')
    train_parser.add_argument('--epochs', type=int, default=15,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.001,
                             help='Learning rate')
    train_parser.add_argument('--validation-split', type=float, default=0.2,
                             help='Validation data fraction')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze gameplay')
    analyze_parser.add_argument('--video', type=str, help='Path to video file')
    analyze_parser.add_argument('--frames', type=str, help='Path to existing frames')
    analyze_parser.add_argument('--model', type=str, required=True,
                               help='Path to trained model')
    analyze_parser.add_argument('--output', type=str, required=True,
                               help='Path for analysis output')
    analyze_parser.add_argument('--confidence-threshold', type=float, default=0.7,
                               help='Minimum confidence for predictions')
    analyze_parser.add_argument('--batch-size', type=int, default=32,
                               help='Batch size for inference')
    analyze_parser.add_argument('--fps', type=int, default=5,
                               help='Frames per second (if processing video)')
    
    # Analyze video (stream) command
    analyze_video_parser = subparsers.add_parser('analyze-video', 
                                                 help='Analyze video without saving frames')
    analyze_video_parser.add_argument('video', type=str, 
                                      help='Path to video file')
    analyze_video_parser.add_argument('--model', type=str, required=True, 
                                      help='Path to trained model')
    analyze_video_parser.add_argument('--output', type=str, 
                                      help='Path for analysis output JSON')
    analyze_video_parser.add_argument('--confidence-threshold', type=float, default=0.7, 
                                      help='Minimum confidence for predictions')
    analyze_video_parser.add_argument('--skip-frames', type=int, default=0, 
                                      help='Number of frames to skip between predictions')

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    try:
        # Set up logging
        output_dir = getattr(args, 'output', 'output')
        Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
        setup_logging(Path(output_dir).parent)
        
        # Execute command
        if args.command == 'process-video':
            process_video_command(args)
        elif args.command == 'train':
            train_model_command(args)
        elif args.command == 'analyze':
            analyze_command(args)
        elif args.command == 'analyze-video':
            analyze_video_command(args)
            
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
# Esports Coaching Tool

An AI-powered tool for analyzing gameplay videos and providing actionable insights. Currently supports Apex Legends gameplay analysis with metrics for time spent looting, checking inventory, and map usage.

## Project Structure
```
esports-coaching-tool/
├── data/
│   ├── test_frames/      # Frames for testing models
│   ├── train_frames/     # Training frames with annotations
│   │   └── annotations/  # Frame annotations (looting, map, interface, gameplay)
│   └── videos/           # Source gameplay videos
├── models/               # Trained model checkpoints
├── scripts/
│   ├── inference.py      # Prediction logic
│   ├── models.py         # Neural network architecture
│   ├── process_video.py  # Video processing utilities
│   ├── train_model.py    # Model training script
│   └── utils.py         # Common utilities
├── venv/                # Python virtual environment
├── main.py             # Main application entry point
└── requirements.txt    # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/esports-coaching-tool.git
cd esports-coaching-tool
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The tool provides several commands for different operations:

### Process a Video into Frames

Extract frames from a gameplay video for analysis:
```bash
python main.py process-video path/to/video.mp4 --output data/frames --fps 5
```

Options:
- `--fps`: Frames per second to extract (default: 5)
- `--output`: Output directory for frames (default: data/test_frames)

### Train a New Model

Train the model on annotated frames:
```bash
python main.py train \
    --data data/train_frames \
    --model-output models/game_state_cnn.pth \
    --epochs 15 \
    --batch-size 32 \
    --learning-rate 0.001
```

Options:
- `--epochs`: Number of training epochs (default: 15)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--validation-split`: Validation data fraction (default: 0.2)

### Analyze Gameplay

Analyze extracted frames using a trained model:
```bash
python main.py analyze \
    --frames data/test_frames \
    --model models/game_state_cnn.pth \
    --output output/analysis.json
```

Or analyze a video directly:
```bash
python main.py analyze \
    --video path/to/video.mp4 \
    --model models/game_state_cnn.pth \
    --output output/analysis.json
```

Options:
- `--confidence-threshold`: Minimum confidence for predictions (default: 0.7)
- `--batch-size`: Batch size for inference (default: 32)

## Output Metrics

The analysis provides several metrics:
- Time distribution across different game states
- Average duration of each state
- State transition patterns
- Confidence scores for predictions

Example output:
```json
{
    "state_distribution": {
        "Gameplay": 0.65,
        "Looting": 0.20,
        "Map": 0.10,
        "Interface": 0.05
    },
    "average_duration": {
        "Gameplay": 45.2,
        "Looting": 12.3,
        "Map": 3.1,
        "Interface": 2.8
    }
}
```

## Development

### Adding New Game States

1. Create a new annotation folder in `data/train_frames/annotations/`
2. Add annotated frames
3. Update the label map in `config.yaml`
4. Retrain the model

### Model Architecture

The tool uses a CNN architecture optimized for game state classification:
- Input size: 224x224 RGB images
- Multiple convolutional layers with max pooling
- Fully connected layers with dropout
- Output: Probability distribution over game states

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

@ George Kanailov


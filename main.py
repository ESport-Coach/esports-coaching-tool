import os

import torch
from scripts.inference import predict_multiple_frames
from scripts.models import SimpleCNN
from scripts.utils import parse_ndjson, extract_frames

if __name__ == "__main__":
    # Paths and Configurations
    output_frames_dir = "data/test_frames"
    model_path = "models/simple_cnn.pth"
    label_map = {0: "Map", 1: "Looting", 2: "Interface", 3: "Gameplay"}

    #video_path = "data/batch1_000.mp4"
    #extract_frames(video_path, output_frames_dir, fps=5)
    #print(f"Extracted frames to {output_frames_dir}.")

    model = SimpleCNN(num_classes=len(label_map))
    state_dict = torch.load(model_path)
        # Match model architecture
    model_state_dict = model.state_dict()
    for key in state_dict.keys():
        if key in model_state_dict and state_dict[key].size() != model_state_dict[key].size():
            print(f"Skipping parameter {key} due to size mismatch: {state_dict[key].size()} vs {model_state_dict[key].size()}")
            del state_dict[key]
    
    # Load matching parameters
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded model from {model_path}.")

    # 3. Run Predictions
    results = predict_multiple_frames(model, output_frames_dir, label_map)
    print("Predictions:")
    for res in results:
        print(res)

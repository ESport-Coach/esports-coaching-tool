import torch
from torchvision import transforms
from PIL import Image
import os
import json

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

def predict_single(model, image_path, label_map):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
    return label_map[predicted_idx.item()]

def predict_multiple_frames(model, test_dir, label_map, output_file="results.json"):
    results = []
    for frame in sorted(os.listdir(test_dir)):
        frame_path = os.path.join(test_dir, frame)
        if frame.endswith(".jpg"):
            label = predict_single(model, frame_path, label_map)
            results.append({"frame": frame, "label": label})

    with open(output_file, "w") as f:
        json.dump(results, f)
    return results

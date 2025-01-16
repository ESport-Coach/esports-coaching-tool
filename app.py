from flask import Flask, request, jsonify
from process_video import extract_frames, load_model, predict_frames

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_video():
    video_path = request.json['video_path']
    model_path = "models/trained_model.pth"

    # Process video
    frame_dir = extract_frames(video_path, "data/frames")
    model = load_model(model_path)
    label_map = {0: "Gameplay", 1: "Looting", 2: "Interface"}
    results = predict_frames(model, frame_dir, label_map)

    return jsonify(results)

if __name__ == "__main__":
    app.run(port=5000)
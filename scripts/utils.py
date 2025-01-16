import cv2
import os
import json

def parse_ndjson(file_path):
    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            video_name = data['data_row']['external_id']
            frame_data = data['projects']['cm5fa8szl02xx07y0gu1cedbc']['labels'][0]['annotations']['frames']

            for frame_number, frame_info in frame_data.items():
                for obj in frame_info['objects'].values():
                    annotations.append({
                        "video_name": video_name,
                        "frame": int(frame_number),
                        "label": obj['value']
                    })
    return annotations

def extract_frames(video_path, output_dir, fps=5):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS) / fps)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_rate == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_id:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_id += 1
    cap.release()
    return output_dir

# Example usage
if __name__ == "__main__":
    annotations = parse_ndjson("data/batch1.ndjson")
    print(f"Parsed {len(annotations)} annotations.")
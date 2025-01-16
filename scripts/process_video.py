import cv2
import os

from utils import parse_ndjson

def extract_frames(video_dir, annotations, output_dir, fps=5):
    os.makedirs(output_dir, exist_ok=True)
    video_cache = {}

    for ann in annotations:
        video_name = ann['video_name']
        if video_name not in video_cache:
            video_cache[video_name] = cv2.VideoCapture(os.path.join(video_dir, video_name))

        cap = video_cache[video_name]
        frame_num = ann["frame"]
        label = ann["label"]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        frame_path = os.path.join(label_dir, f"{video_name}_frame_{frame_num:05d}.jpg")
        cv2.imwrite(frame_path, frame)

    for cap in video_cache.values():
        cap.release()

if __name__ == "__main__":
    annotations = parse_ndjson("data/batch2.ndjson")
    extract_frames("data", annotations, "data/train_frames")
    print("Frames extracted and saved.")

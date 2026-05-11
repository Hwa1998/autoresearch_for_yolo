from ultralytics import YOLO
import os

MODEL_PATH = r"runs/detect/autoresearch_dingweixiao_100ep-final/weights/best.pt"
SOURCE = r"D:\DingReihwa\6_AI\datasets\yolo_datasets\detect\dingweixiao\images"
OUT_DIR = r"runs/detect/autoresearch_dingweixiao_100ep-final/predict"

os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
print('Running prediction with', MODEL_PATH)
model.predict(source=SOURCE, save=True, imgsz=640, conf=0.25, save_dir=OUT_DIR)
print('Predictions saved to', OUT_DIR)

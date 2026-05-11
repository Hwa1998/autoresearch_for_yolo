from ultralytics import YOLO
import os

MODEL_PATH = r"best.pt"
SOURCE = r"images"
OUT_DIR = r"predict"

os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
print('Running prediction with', MODEL_PATH)
model.predict(source=SOURCE, save=True, imgsz=640, conf=0.25, save_dir=OUT_DIR)
print('Predictions saved to', OUT_DIR)

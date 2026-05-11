import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os


if __name__ == '__main__':
    model = YOLO(r'best.pt')  # select your model.pt path
    results = model.predict(source=r'test_images',
                  imgsz=[640, 640],
                  project=r'segment',
                  name='results',
                  save=True,
                  save_txt=True,
                  stream=True,  # 如果会有内存泄漏的风险，就按这个走
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs


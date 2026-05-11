from ultralytics import YOLO
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["WANDB_MODE"] = "offline"
import warnings

# 忽略所有警告
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    model = YOLO('cfg/models/v8/yolov8n.yaml').load(r'best.pt')
    model.train(data='cfg/datasets/xxx.yaml',
                epochs=3000, imgsz=[640, 640], batch=16, workers=12, lr0=0.003,
                name='results')

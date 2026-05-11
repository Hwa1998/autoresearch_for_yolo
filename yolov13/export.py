import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu

if __name__ == '__main__':
    model = YOLO(r'best.pt')
    model.export(format='onnx', simplify=True, opset=12, imgsz=[640, 640], dynamic=False) # [h, w]
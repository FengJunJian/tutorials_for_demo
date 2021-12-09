import torch
import cv2
from PIL import Image
from openvino.inference_engine import IECore
import onnxruntime
# Model
# modeln = torch.hub.load('ultralytics/yolov5', 'yolov5n')
# models = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom
modelm = torch.hub.load('ultralytics/yolov5', 'yolov5m')
# modell = torch.hub.load('ultralytics/yolov5', 'yolov5l')
# modelx = torch.hub.load('ultralytics/yolov5', 'yolov5x')
# Images
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
imgpath='../MVI_1587_VIS_00423.jpg'

img=Image.open(imgpath)
# im=cv2.resize(img,(640,640))
# im=torch.Tensor(img)
# im=torch.unsqueeze(im.permute(2,0,1),0)
# Inference
results = modelm(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.show()

import torch
import cv2
from PIL import Image
from openvino.inference_engine import IECore
import onnxruntime
#https://pytorch.org/hub/hustvl_yolop/
# Model
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
# Images
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
imgpath='../MVI_1587_VIS_00423.jpg'
img=cv2.imread(imgpath)
img1=Image.open(imgpath)
im=cv2.resize(img,(640,640))
im=torch.Tensor(img)
im=torch.unsqueeze(im.permute(2,0,1),0)
# Inference
# img = torch.randn(1,3,640,640)
det_out, da_seg_out,ll_seg_out = model(im)
results = model(im)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.show()

#modelm.model.model.model(im.cuda())
#model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
# onnx_model_name='yolov5m.onnx'
# input_names = ["input"]
# output_names = ["output"]
#torch.onnx.export(modelm, (img), onnx_model_name, export_params=True,verbose=True, input_names=input_names, output_names=output_names)

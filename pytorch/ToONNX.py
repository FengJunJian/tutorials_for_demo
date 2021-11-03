import torch
import torchvision
from torchvision import transforms
import onnx
import onnxruntime
from PIL import Image
from ImageNetClass import class_name

normalize = transforms.Normalize(mean=(0.51264606, 0.55715489, 0.6386575), std=(0.15772002, 0.14560729, 0.13691749))
transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224, 224)),
                                transforms.RandomHorizontalFlip(),normalize])

img=Image.open('MVI_1469_VIS_00003_1.jpg')
img=transform(img)
img=torch.unsqueeze(img,0)
dummy_input = torch.randn(1, 3, 224, 224, )#device="cuda"

def build_model2onnx(input):

    model = torchvision.models.alexnet(pretrained=True)#.cuda() resnet18
    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]
    torch.onnx.export(model, input, "alexnet.onnx", export_params=True,verbose=True, input_names=input_names, output_names=output_names)
    return model

model=build_model2onnx(img)
o1=model(dummy_input)
o2=model(img)

def inference(filename,input_dict):
    # model_onnx = onnx.load("alexnet.onnx")
    # onnx.checker.check_model(model_onnx)
    # # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model_onnx.graph))
    session=onnxruntime.InferenceSession(filename)
    output=session.run(None,input_dict)
    return output
input1={"actual_input_1":img.numpy()}
filename='alexnet.onnx'
output=inference(filename,input1)
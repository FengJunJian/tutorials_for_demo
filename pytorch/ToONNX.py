import torch
import torchvision
from torchvision import transforms
import onnx
import onnxruntime
from PIL import Image
from DatasetClass import ImageNet_className,COCO_className


def build_model2onnx(input):

    model = torchvision.models.alexnet(pretrained=True)#.cuda() resnet18
    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]
    torch.onnx.export(model, input, "alexnet.onnx", export_params=True,verbose=True, input_names=input_names, output_names=output_names)
    return model



def inference(onnxfilename,input_dict,output_name=None):
    # model_onnx = onnx.load("alexnet.onnx")
    # onnx.checker.check_model(model_onnx)
    # # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model_onnx.graph))
    if output_name:
        assert isinstance(output_name,list)
    session=onnxruntime.InferenceSession(onnxfilename)
    output=session.run(output_name,input_dict)
    return output

if __name__ == "__main__":
    normalize = transforms.Normalize(mean=(0.51264606, 0.55715489, 0.6386575), std=(0.15772002, 0.14560729, 0.13691749))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(), normalize])

    img = Image.open('MVI_1469_VIS_00003_1.jpg')
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    dummy_input = torch.randn(1, 3, 224, 224, )  # device="cuda"

    model = build_model2onnx(img)
    o1 = model(dummy_input)
    o2 = model(img)

    input1={"actual_input_1":img.numpy()}
    filename='alexnet.onnx'
    output=inference(filename,input1)
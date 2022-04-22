import numpy as np
import onnxruntime
import time
import cv2

def clip_boxes(boxes, shape):
      """
            Args:
                boxes: (...)x4, float
                shape: h, w
      """
      orig_shape = boxes.shape
      boxes = boxes.reshape([-1, 4])
      h, w = shape
      boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
      boxes[:, 2] = np.minimum(boxes[:, 2], w)
      boxes[:, 3] = np.minimum(boxes[:, 3], h)
      return boxes.reshape(orig_shape)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
def drawBox(im,xyxy,label=None):
    box=xyxy
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    if label:
        th = 3.0#max(self.lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=th / 3, thickness=th-1)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, (125,125,125), -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, th / 3, (250,2,2),
                    thickness=th-1, lineType=cv2.LINE_AA)
    return cv2.rectangle(im, p1, p2, (0, 0, 250), thickness=3, lineType=cv2.LINE_AA)




def mainFasterRCNN(img,onnxSession,flagShow=False):
    #imgPath='E:/fjj/SeaShips_SMD/JPEGImages/MVI_1478_VIS_00239.jpg'#图片
    #onnxfilename = 'detection/fasterRCNN.onnx'  # 模型

    #img=cv2.imread(imgPath)
    resized_img = cv2.resize(img,(1067,600))#(H,W,C)
    resized_img=resized_img.astype(np.float32)
    input_dict={'image:0':resized_img}
    Wr,Hr=resized_img.shape[:2]
    W,H=img.shape[:2]
    scale=min(Wr/W,Hr/H)

    #session=onnxruntime.InferenceSession(onnxfilename)

    a=time.time()
    boxes, probs, labels=onnxSession.run(None,input_dict)#boxes, probs, labels
    b=time.time()
    print('time:',b-a)
    boxes = boxes / scale
    boxes = clip_boxes(boxes, (W,H))
    if flagShow:
        showImg=img.copy()
        for box in boxes:
            showImg=drawBox(showImg,box)
        cv2.imshow('a',showImg)
        cv2.waitKey(1)
    return boxes#(xmin,ymin,xmax,ymax)


if __name__ == '__main__':
    imgPath='E:/fjj/SeaShips_SMD/JPEGImages/MVI_1478_VIS_00239.jpg'#图片
    onnxfilename = 'detection/fasterRCNN.onnx'  # 模型

    img=cv2.imread(imgPath)
    resized_img = cv2.resize(img,(1067,600))#(H,W,C)
    resized_img=resized_img.astype(np.float32)
    input_dict={'image:0':resized_img}
    Wr,Hr=resized_img.shape[:2]
    W,H=img.shape[:2]
    scale=min(Wr/W,Hr/H)

    session=onnxruntime.InferenceSession(onnxfilename)

    a=time.time()
    boxes, probs, labels=session.run(None,input_dict)#boxes, probs, labels
    b=time.time()
    print('time:',b-a)
    boxes = boxes / scale
    boxes = clip_boxes(boxes, (W,H))
    showImg=img.copy()
    for box in boxes:
        showImg=drawBox(showImg,box)
    cv2.imshow('a',showImg)
    cv2.waitKey()


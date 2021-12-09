from PIL import Image
import cv2
import os
import numpy as np
from pytorch.ToONNX import inference
import time
#from DatasetClass import ImageNet_className,COCO_className


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    keep = []
    if not dets.size>0:
        return keep
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]


    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)
def drawBox(im,xyxy):
    box=xyxy
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    return cv2.rectangle(im, p1, p2, (0,0,255), thickness=3, lineType=cv2.LINE_AA)
    # if label:
    #     tf = max(self.lw - 1, 1)  # font thickness
    #     w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
    #     outside = p1[1] - h - 3 >= 0  # label fits outside box
    #     p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    #     cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
    #     cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
    #                 thickness=tf, lineType=cv2.LINE_AA)

def box_iou_py(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    # interT = (torch.min(torch.tensor(box1[:, None, 2:]), torch.tensor(box2[:, 2:])) - torch.max(torch.tensor(box1[:, None, :2]), torch.tensor(box2[:, :2]))).clamp(0).prod(2)
    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])).clip(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def xywh2xyxy_py(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    #y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression_py(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    #output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output = [np.zeros((0, 6), dtype=prediction.dtype)] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5), dtype=x.dtype)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy_py(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype(np.float32)), 1)
        else:  # best class only
            conf = np.max(x[:, 5:], axis=1, keepdims=True)
            j=np.reshape(np.argmax(x[:, 5:],axis=1),conf.shape)
            #conf, j = x[:, 5:].max(1, keepdim=True)
            #x = np.concatenate((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            x = np.concatenate((box, conf, j.astype(np.float32)), 1)[conf.flatten() > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes, dtype=x.dtype)).any(1)]
            # x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            #x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            x=x[np.argsort(-x[:, 4])[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = py_cpu_nms(np.concatenate((boxes, scores[:,None]),axis=-1), iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou_py(boxes[i], boxes) > iou_thres  # iou matrix
            #iou = py_cpu_nms(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            # a=torch.mm(torch.tensor(weights), torch.tensor(x[:, :4])).float() / torch.tensor(weights).sum(1, keepdim=True)  # merged boxes
            # b = np.matmul(weights, x[:, :4]).astype(np.float32) / weights.sum(1, keepdims=True)  # merged boxes
            x[i, :4] = np.matmul(weights, x[:, :4]).astype(np.float32) / weights.sum(1, keepdims=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output
def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


imgpath='../MVI_1587_VIS_00423.jpg'
#imgpath='./ultralytics_yolov5_master\data\images/bus.jpg'
img=cv2.imread(imgpath)
cv2.imshow('src',img)
imv=img.copy()
img=letterbox(img.copy(),(640,640),stride=64,auto=False)[0]
# img=cv2.resize(img,(640,640),interpolation=cv2.INTER_LINEAR)
cv2.imshow('resize',img)
img=np.array(img,dtype=np.float32)
img=np.transpose(img,(2,0,1))[::-1]
img/=255
if len(img.shape) == 3:
    img = img[None]
# img=np.expand_dims(img,0)
w,h=img.shape[2:]
#im=transforms.ToTensor()(img)
# input1={"images":np.array(img).astype(np.float32)}#.numpy()
input1={"images":img}
output_name=['output']
filename='./ultralytics_yolov5_master/yolov5m.onnx'
#session=onnxruntime.InferenceSession(filename)
# print(session.get_inputs()[0].name)
# print(session.get_outputs()[0].name)
#output=session.run(output_name,input_dict)
output=inference(filename,input1,output_name=output_name)[0]
# output[...,0]*=w
# output[...,0]*=h
# output[...,0]*=w
# output[...,0]*=h
classes=None
agnostic_nms=False
max_det=1000
#pred = non_max_suppression(torch.tensor(output), 0.25, 0.45, classes, agnostic_nms, max_det=max_det)
pred1 = non_max_suppression_py(output, 0.25, 0.45, classes, agnostic_nms, max_det=max_det)
imt=imv.copy()
for i, det in enumerate(pred1):
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], imt.shape).round()
    for *xyxy, conf, cls in reversed(det):
        imt=drawBox(imt, xyxy)

cv2.imshow('draw', imt)
cv2.waitKey()
# tmp=output[1][0,:,:,:,0]#-output[1][0,:,:,:,0].min()
# tmp=np.transpose(tmp,(1,2,0))

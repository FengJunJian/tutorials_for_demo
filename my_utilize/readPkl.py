import pickle as pkl
import os

path='E:/fjj/SeaShips_SMD/annotations_cache'
path1='E:/fjj/Semi-Faster-RCNN-revised/data/cache'
file='test1300_annots_vessel.pkl'
fileP=os.path.join(path,file)
with open(fileP,'rb') as f:
    output=pkl.load(f)
file1='test1283_gt_roidb.pkl'
fileP1=os.path.join(path1,file1)
with open(fileP1,'rb') as f:
    output1=pkl.load(f)

det_file='E:\\fjj\\semi_ship_last\\vgg16_semi\\test1283\\label0_unlabel0none_145000top250tt03\\semi_model/detections.pkl'
with open(fileP1,'rb') as f:
    det_output=pkl.load(f)

path='E:/fjj/data_processing_cache/sfrp_detections1300.pkl'
with open(path,'rb') as f:
    o1=pkl.load(f)
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

path = 'E:/fjj/semi_ship_test/vgg16_semi/label0_unlabel2/even/events.out.tfevents.1605695687.DESKTOP-PIBGHC6'
load_num = 20
scalars = []
gt_imgs = [[],[]]
vat_imgs=[[],[]]
vadv=[[],[]]

i=0
for vs in tf.train.summary_iterator(path):
    for v in vs.summary.value:
        #print(v.tag)
        i+=1
        if v.tag=="prediction_1/vat_image/image/0":
            vat_imgs[0].append(tf.image.decode_jpeg(v.image.encoded_image_string))
        elif v.tag=="prediction_1/vat_image/image/1":
            vat_imgs[1].append(tf.image.decode_jpeg(v.image.encoded_image_string))
        elif v.tag=="prediction_1/ori_image/image/0":
            gt_imgs[0].append(tf.image.decode_jpeg(v.image.encoded_image_string))
        elif v.tag == "prediction_1/ori_image/image/1":
            gt_imgs[1].append(tf.image.decode_jpeg(v.image.encoded_image_string))

        # elif v.tag=="prediction_1/r_vadv/image/0":
        #     vadv[0].append(tf.image.decode_jpeg(v.image.encoded_image_string))
        # elif v.tag=="prediction_1/r_vadv/image/1":
        #     vadv[1].append(tf.image.decode_jpeg(v.image.encoded_image_string))
    if len(gt_imgs[0]) > load_num:
        break
print(i)
sess = tf.Session()

oimgs = sess.run(gt_imgs[0][0:load_num])
vimgs = sess.run(vat_imgs[0][0:load_num])
#adv = sess.run(vadv[0][0:load_num])

i=4
# advimg=vimgs[i].astype(np.float32)+adv[i].astype(np.float32)/3
# advimg=cv2.cvtColor(advimg,cv2.COLOR_RGB2BGR)
# cv2.imshow('adv',np.uint8(advimg))
dimg=vimgs[i].astype(np.float32)-oimgs[i].astype(np.float32)
dimg=(dimg-dimg.min())*10
cv2.imshow('dimg',np.uint8(dimg))

imgtmp=oimgs[i]+dimg
src=cv2.cvtColor(oimgs[i],cv2.COLOR_RGB2BGR)
imgtmp=cv2.cvtColor(imgtmp,cv2.COLOR_RGB2BGR)
cv2.imshow('oriimg',src)
cv2.imshow('advimg',np.uint8(imgtmp))

cv2.waitKey()
# cv2.imdecode(v.image.encoded_image_string,cv2.IMREAD_COLOR)
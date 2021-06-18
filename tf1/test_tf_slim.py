# import tf_slim as slim
import tensorflow as tf
# import tensorlayer
path='E:/fjj/Semi-Faster-RCNN-revised/data/imagenet_weights/resnet_v1_50.ckpt'
model=tf.train.load_checkpoint(path)
tf.config.experimental.get_visible_devices()

import tensorflow as tf
# import tensorflow_estimator as tfe
from tensorflow.python import pywrap_tensorflow
import tensorflow_hub.module as iu


import os
old_ckptpath='E:\\fjj\\semi_ship_test\\vgg16_semi\\label0_unlabel0\\vgg16_semi_faster_rcnn_iter_5000.ckpt'#@vgg16_semi_faster_rcnn_iter_5000.ckpt'
old_meta='E:\\fjj\\semi_ship_test\\vgg16_semi\\label0_unlabel0\\vgg16_semi_faster_rcnn_iter_192000.ckpt.meta'#@vgg16_semi_faster_rcnn_iter_5000.ckpt'
save_ckptpath='E:\\fjj\\semi_ship_test\\vgg16_semi\\label0_unlabel0\\vgg16_semi_faster_rcnn_iter_192002.ckpt'#@vgg16_semi_faster_rcnn_iter_5000.ckpt'

new_ckptpath='E:\\fjj\\semi_ship_test\\vgg16_semi\\label0_unlabel0\\vgg16_semi_faster_rcnn_iter_192001.ckpt'
new_meta='E:\\fjj\\semi_ship_test\\vgg16_semi\\label0_unlabel0\\vgg16_semi_faster_rcnn_iter_192001.ckpt.meta'#@vgg16_semi_faster_rcnn_iter_5000.ckpt'
# ckptpath1='E:\\fjj\\Semi-Faster-RCNN-revised\\data\\imagenet_weights\\vgg16.ckpt'
# pbpath='E:\\fjj\\semi_ship_test\\vgg16_semi\\label0_unlabel0\\'


a=0
sess=tf.Session()
old_model=tf.train.load_checkpoint(old_ckptpath)
old_saver=tf.train.import_meta_graph(old_meta)
old_saver.restore(sess,old_ckptpath)
# old_model.get_tensor('Variable')
# old_saver.export_meta_graph('a.meta')
# with tf.variable_scope('',reuse=tf.AUTO_REUSE):
#     b=tf.get_variable('Variable1',shape=1)
#     c = tf.Variable(1,name='Variable')
new_model=tf.train.load_checkpoint(new_ckptpath)
new_saver=tf.train.import_meta_graph(new_meta)
# for v in tf.global_variables():
#     if v.name=='Variable:0':
#        a=0

new_model=tf.train.load_checkpoint(new_ckptpath)
# tf.train.import_meta_graph()

# aa=tf.train.list_variables(new_ckptpath)
print(new_model.get_variable_to_shape_map())
old_model=tf.train.load_checkpoint(old_ckptpath)
print(old_model.get_variable_to_shape_map())
old_keys=old_model.get_variable_to_shape_map().keys()
new_keys=new_model.get_variable_to_shape_map().keys()
for k in old_keys:
    if k not in new_keys:
        print(k)
for k in new_keys:
    if k not in old_keys:
        print(k)
saver=tf.train.import_meta_graph(new_meta)
tf.train.list_variables()
# saver.restore(sess,original_ckptpath)

tf.train.list_variables(old_ckptpath)
v=tf.train.load_variable(old_ckptpath,'Variable')
# new_var_name = var_name.replace('InceptionV3', 'IV')
# lr = tf.Variable(v, name='lr',trainable=False)
# saver = tf.train.Saver()
# sess.run(tf.global_variables_initializer())
# saver.save(sess, 'E:\\fjj\\semi_ship_test\\vgg16_semi\\label0_unlabel0\\vgg16_semi_faster_rcnn_iter_192001.ckpt')
# # model1=tf.train.load_checkpoint(ckptpath1)
# print(model.get_variable_to_shape_map())
# print(model1.get_variable_to_shape_map())


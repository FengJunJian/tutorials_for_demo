import tensorflow as tf
import os
ckptpath='E:\\fjj\\semi_ship_test\\vgg16_semi\\label0_unlabel0\\vgg16_semi_faster_rcnn_iter_192000.ckpt'#@vgg16_semi_faster_rcnn_iter_5000.ckpt'
ckptpath1='E:\\fjj\\semi_ship_test\\vgg16_semi\\label0_unlabel0\\vgg16_semi_faster_rcnn_iter_15000.ckpt'
# ckptpath1='E:\\fjj\\Semi-Faster-RCNN-revised\\data\\imagenet_weights\\vgg16.ckpt'
pbpath='E:\\fjj\\semi_ship_test\\vgg16_semi\\label0_unlabel0\\'
meta_path='E:\\fjj\\semi_ship_test\\vgg16_semi\\label0_unlabel0\\vgg16_semi_faster_rcnn_iter_192000.ckpt.meta'#@vgg16_semi_faster_rcnn_iter_5000.ckpt'

sess=tf.Session()
model=tf.train.load_checkpoint(ckptpath)
# saver=tf.train.import_meta_graph(meta_path,clear_devices=True)
# saver.restore(sess,ckptpath)
tf.train.list_variables(ckptpath)
v=tf.train.load_variable(ckptpath,'Variable')
# new_var_name = var_name.replace('InceptionV3', 'IV')
lr = tf.Variable(v, name='lr',trainable=False)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'E:\\fjj\\semi_ship_test\\vgg16_semi\\label0_unlabel0\\vgg16_semi_faster_rcnn_iter_192001.ckpt')
model1=tf.train.load_checkpoint(ckptpath1)
print(model.get_variable_to_shape_map())
print(model1.get_variable_to_shape_map())


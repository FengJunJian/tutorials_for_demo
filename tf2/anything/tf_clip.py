import tensorflow as tf
a=tf.constant([-1,-0.2,0,1.2,255,256,257],dtype=tf.float32)
b=tf.clip_by_value(a,0,10)
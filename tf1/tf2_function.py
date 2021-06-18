import tensorflow as tf
tensor_list = []

@tf.function #加上这一行切换成Autograph结果将不符合预期！！！
def append_tensor(x):
    tensor_list.append(x)
    return tensor_list
x=tf.Variable(0.0,dtype=tf.float32)


append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)
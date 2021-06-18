import tensorflow as tf
tf.compat.v1.enable_eager_execution()
#sess=tf.InteractiveSession()

ds = tf.data.Dataset.from_tensor_slices([[1,2,3],[4,5,6],[7,8,9]])

ds_flatmap = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x + 1))
iter=ds_flatmap.make_initializable_iterator()
print(tf.keras.__version__)
# tf.print
# sess.run(iter.initializer)
# data=sess.run(iter.get_next())
#
# for x in data:
#     print(x)
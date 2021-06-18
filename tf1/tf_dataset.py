import tensorflow as tf

ds=tf.data.Dataset.range(1,4)
ds=ds.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(5),
            cycle_length=2, block_length=2)
#ds=ds.take(2)
for d in ds:
    tf.print(d)



#sess=tf.Session()
#d=d.interleave(lambda x:tf.data.Dataset.from_tensors(x).map(lambda x:x+1),1,1)
# sess.run(d)

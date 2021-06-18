import tensorflow as tf
import tensorflow_datasets as tfds
tf.enable_eager_execution()
print(tf.__version__)
print(tfds.list_builders())
#dataset = tfds.load("mnist")
dataset=tfds.load('cifar10',split=tfds.Split.TRAIN,as_supervised=True)

dataset=dataset.map(lambda img, label: (tf.image.resize(img, [224, 224]) / 255.0, label)).shuffle(1024).batch(32)

# dataset=dataset.cache().repeat().batch(2).prefetch(100)

i=0
for img,label in dataset:
    print(img,label)
    i=i+1
    if i>10:
        break
# dataset = tfds.load("cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)
# dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
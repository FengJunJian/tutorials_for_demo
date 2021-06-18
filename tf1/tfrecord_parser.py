import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import os
import glob

path='tfrecords/seasmd_label0_*.tfrecord'
files=glob.glob(path)
# files=['tfrecords/seasmdbb_label0_0000.tfrecord']#glob.glob(path)
#filename_queue=tf.train.string_input_producer(files)
# reader=tf.TFRecordReader()
# serialized_example=reader.read(filename_queue)


keys_to_features = {
        'image/filename':tf.FixedLenFeature((),tf.string,default_value=''),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }
items_to_handlers = {
        'filename':slim.tfexample_decoder.Tensor('image/filename'),
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }


def _parse_record(example):
        features = tf.parse_single_example(
           example,
           features=keys_to_features)
        filename=features['image/filename']
        # image=tf.image.decode_jpeg(features['image/encoded'])
        image=tf.image.decode_image(features['image/encoded'])
        shape=features['image/shape']
        #image=tf.reshape(image,features['image/shape'])
        xmin = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'])
        ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'])
        xmax = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'])
        ymax = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'])
        label = tf.sparse_tensor_to_dense(features['image/object/bbox/label'])
        boundingbox=tf.stack([ymin,xmin,ymax,xmax],-1)

        return features,filename,boundingbox,label,image,shape

ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}

'''decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
dataset=slim.dataset.Dataset(data_sources=files,reader=reader,decoder=decoder,num_samples=15,items_to_descriptions=ITEMS_TO_DESCRIPTIONS)
provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=1,
                    common_queue_capacity=20 * 1,
                    common_queue_min=10 * 1,
                    shuffle=True)
[filename,image, shape, glabels, gbboxes] = provider.get(['filename','image', 'shape',
                                                             'object/label','object/bbox'])'''


coord = tf.train.Coordinator()
sess=tf.Session()
threads=tf.train.start_queue_runners(sess=sess,coord=coord)


dataset=tf.data.TFRecordDataset(files)
dataset=dataset.map(_parse_record).prefetch(100)
iterator = dataset.make_initializable_iterator()
# itera=dataset.make_one_shot_iterator()
sess.run(iterator.initializer)
#sess.run(iterator.initializer())
a=sess.run(iterator.get_next())




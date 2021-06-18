import tensorflow as tf
import glob
import matplotlib.pyplot as plt
# message Example
# {
#     Features features = 1;
# };
#
# message Features
# {
#     map< string, Feature > featrue = 1;
# };
#
# message Feature
#
#     oneof kind{
#     BytesList bytes_list = 1;
#     FloatList float_list = 2;
#     Int64List int64_list = 3;
# }
# };


path='cifar/cifar10*.tfrecord*'
files=glob.glob(path)
# files=['tfrecords/seasmdbb_label0_0000.tfrecord']#glob.glob(path)
#filename_queue=tf.train.string_input_producer(files)
# reader=tf.TFRecordReader()
# serialized_example=reader.read(filename_queue)



keys_to_features = {
        "id": tf.FixedLenFeature([], tf.string, default_value=''),
        "image":tf.FixedLenFeature([],tf.string,default_value=''),
        "label":tf.FixedLenFeature([1],tf.int64,default_value=0),
        # 'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        # 'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        # 'image/height': tf.FixedLenFeature([1], tf.int64),
        # 'image/width': tf.FixedLenFeature([1], tf.int64),
        # 'image/channels': tf.FixedLenFeature([1], tf.int64),
        # 'image/shape': tf.FixedLenFeature([3], tf.int64),
        # 'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        # 'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        # 'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        # 'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        # 'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        # 'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        # 'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }


def _parse_record(example):
        features = tf.parse_single_example(
           example,
           features=keys_to_features)
        id=features['id']
        image=features['image']
        label=features['label']
        image=tf.image.decode_image(image)

        #tf.image.resize(image)
        return id,image,label
        # filename=features['image/filename']
        # # image=tf.image.decode_jpeg(features['image/encoded'])
        # image=tf.image.decode_image(features['image/encoded'])
        # shape=features['image/shape']
        # #image=tf.reshape(image,features['image/shape'])
        # xmin = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'])#可变维度
        # ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'])
        # xmax = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'])
        # ymax = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'])
        # label = tf.sparse_tensor_to_dense(features['image/object/bbox/label'])
        # boundingbox=tf.stack([ymin,xmin,ymax,xmax],-1)
        #
        # return features,filename,boundingbox,label,image,shape

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



##################################################
#未知tf键值
if False:
    with tf.Session() as sess:
        Keys2Features={}

        reader = tf.TFRecordReader()
        filename_queue = tf.train.string_input_producer(files)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        file_sampleId, example = reader.read(filename_queue)
        sample_name,serialized_example=sess.run([file_sampleId,example])
        print(sample_name)
        example_proto = tf.train.Example.FromString(serialized_example)
        features = example_proto.features
        print("Features:",features)
        for key in features.feature:
            feature = features.feature[key]
            if len(feature.bytes_list.value) > 0:
                  ftype = 'bytes_list'
                  Keys2Features[key]="tf.FixedLenFeature([], tf.string, default_value=''"
                  #fvalue = feature.bytes_list.value
            if len(feature.float_list.value) > 0:
                  ftype = 'float_list'
                  Keys2Features[key] = "tf.FixedLenFeature([%d], tf.float32, default_value=''"%(len(feature.float_list.value))
                  # fvalue = feature.float_list.value
            if len(feature.int64_list.value) > 0:
              ftype = 'int64_list'
              Keys2Features[key] = "tf.FixedLenFeature([%d], tf.int64, default_value=0" % (len(feature.int64_list.value))
              # fvalue = feature.int64_list.value
        print(Keys2Features)

    reader=tf.TFRecordReader()
    filename_queue=tf.train.string_input_producer([files[0]])
    _,example=reader.read(filename_queue)
    example_proto =tf.train.Example.FromString(example)
    features_message = example_proto.features
    features = tf.parse_single_example(example,keys_to_features)
#


#已知tf键值，使用： tf.data.TFRecordDataset
batch=100
dataset=tf.data.TFRecordDataset(files)
dataset=dataset.map(_parse_record).batch(batch).prefetch(200)
iterator = dataset.make_initializable_iterator()
sess=tf.Session()
sess.run(iterator.initializer)
i=0
while True:
    try:
        id,image,label=sess.run(iterator.get_next())
        plt.imshow(image[0])
        plt.draw()
        plt.pause(0.5)#停留0.5秒
        i=i+1
        print('batch:',i*batch)
    except tf.errors.OutOfRangeError:
        print('OutOfRangeError')
        break
    except Exception as e:
        print(str(Exception))
        print(str(e))
        break





import tensorflow as tf
import glob
import matplotlib.pyplot as plt
tf.enable_eager_execution()
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


path='tf1/cifar/cifar10*.tfrecord*'

files=glob.glob(path)
# files=['tfrecords/seasmdbb_label0_0000.tfrecord']#glob.glob(path)
#filename_queue=tf.train.string_input_producer(files)
# reader=tf.TFRecordReader()
# serialized_example=reader.read(filename_queue)
label = tf.feature_column.numeric_column("label", shape=(), dtype=tf.dtypes.int64)   #创建poi特征
id = tf.feature_column.numeric_column("id",  dtype=tf.dtypes.uint8)
feature_columns = [label]

features = tf.feature_column.make_parse_example_spec(feature_columns)  #生成featuredict

data = tf.data.TFRecordDataset(files)  #读取tfrecord
#分别用tf.io.parse_example 和 tf.io.parse_single_example 解析数据

data1 = data.map(lambda x : tf.io.parse_single_example(x, features = features))

datat=data.batch(2)
data2 = datat.map(lambda x : tf.io.parse_example(x, features = features))



keys_to_features = {
        "id": tf.FixedLenFeature([], tf.string, default_value=''),
        "image":tf.FixedLenFeature([],tf.string,default_value=''),
        "label":tf.FixedLenFeature([1],tf.int64,default_value=0),
    }


def _parse_record(example):
    features = tf.io.parse_single_example(
       example,
       features=keys_to_features)
    id=features['id']
    image=features['image']
    label=features['label']
    image=tf.image.decode_image(image)
    #tf.image.resize(image)
    return id,image,label


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

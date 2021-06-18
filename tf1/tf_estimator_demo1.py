from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves.urllib.request import urlopen
import os
import numpy as np
import tensorflow as tf

# tf.enable_eager_execution()
# Data sets
# sess=tf.Session()
# coord=tf.train.Coordinator()
# threads=tf.train.start_queue_runners(sess=sess,coord=coord)
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


def main():
    # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(IRIS_TRAINING):
        raw = urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, "wb") as f:
          f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, "wb") as f:
          f.write(raw)

    # Specify that all features have real-value data
    def input_fn(files,batch,training):
        fun = lambda x1: tf.equal(tf.strings.regex_full_match(x1, '.*[a-z|A-Z].*'),False)  # 判断是否存在字母
        def funStringSplit(x):
            split_strings = tf.strings.to_number(tf.strings.split(tf.reshape(x,[-1]), ',').values)  # 分割字符串
            features, target = tf.split(split_strings, [4, 1], axis=0)
            target=tf.cast(target,tf.int32)
            #features=tf.reshape(features,[4])
            return ({'x':features}, target)

        dataSet = tf.data.TextLineDataset(files)

        dataSet = dataSet.filter(fun)

        dataSet = dataSet.map(funStringSplit,num_parallel_calls=4)

        if training:
            dataSet=dataSet.shuffle(1000).repeat()
        return dataSet.batch(batch).prefetch(10)

    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
    #input_tensor=tf.feature_column.input_layer({'x':np.array([[1,2,3,4]])},feature_columns=feature_columns)
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="./iris_model1")

    # Listeners = []
    Hooks=[#tf.train.CheckpointSaverHook(checkpoint_dir='./iris_model1/cp',save_steps=2000),
           ]
    # IRIS_TRAINING
    # IRIS_TEST

    classifier.train(input_fn=lambda: input_fn([IRIS_TRAINING], 128,True), max_steps=10000,hooks=Hooks)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=lambda:input_fn([IRIS_TEST], 16,False))["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
    accuracy_score = classifier.evaluate(input_fn=lambda:input_fn([IRIS_TRAINING], 16,False))["accuracy"]
    print("\nTraining Accuracy: {0:f}\n".format(accuracy_score))

    # Classify two new flower samples.
    new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5],#1
       [5.8, 3.1, 5.0, 1.7],
       [6.3,2.9,5.6,1.8]#2
       ],#
        dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]
    print("New Samples, Class Predictions:{}\n".format(predicted_classes))

if __name__ == "__main__":
    main()
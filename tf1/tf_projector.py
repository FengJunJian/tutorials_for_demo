import tensorflow as tf
import mnist_inference
import os

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
learning_rate_base = 0.8
learning_rate_decay = 0.99
training_steps = 10000
moving_average_decay = 0.99

log_dir = 'log'
sprite_file = 'mnist_sprite.jpg'
meta_file = 'mnist_meta.tsv'
tensor_name = 'final_logits'

#获取瓶颈层数据，即最后一层全连接层的输出
def train(mnist):
    with tf.variable_scope('input'):
        x = tf.placeholder(tf.float32,[None,784],name='x-input')
        y_ = tf.placeholder(tf.float32,[None,10],name='y-input')

    y = mnist_inference.build_net(x)
    global_step = tf.Variable(0,trainable=False)

    with tf.variable_scope('moving_average'):
        ema = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
        ema_op = ema.apply(tf.trainable_variables())

    with tf.variable_scope('loss_function'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1)))

    with tf.variable_scope('train_step'):
        learning_rate = tf.train.exponential_decay(
            learning_rate_base,
            global_step,
            mnist.train.num_examples/batch_size,
            learning_rate_decay,
            staircase=True
        )

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

        train_op = tf.group(train_step,ema_op)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(training_steps):
            xs,ys = mnist.train.next_batch(batch_size)
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})

            if step % 100 == 0 :
                print('step:{},loss:{}'.format(step,loss_value))

        final_result = sess.run(y,feed_dict={x:mnist.test.images})

    return final_result

def visualisation(final_result):
    #定义一个新向量保存输出层向量的取值
    y = tf.Variable(final_result,name=tensor_name)
    #定义日志文件writer
    summary_writer = tf.summary.FileWriter(log_dir)

    #ProjectorConfig帮助生成日志文件
    config = projector.ProjectorConfig()
    #添加需要可视化的embedding
    embedding = config.embeddings.add()
    #将需要可视化的变量与embedding绑定
    embedding.tensor_name = y.name

    #指定embedding每个点对应的标签信息，
    #这个是可选的，没有指定就没有标签信息
    embedding.metadata_path = meta_file
    #指定embedding每个点对应的图像，
    #这个文件也是可选的，没有指定就显示一个圆点
    embedding.sprite.image_path = sprite_file
    #指定sprite图中单张图片的大小
    embedding.sprite.single_image_dim.extend([28,28])

    #将projector的内容写入日志文件
    projector.visualize_embeddings(summary_writer,config)

    #初始化向量y，并将其保存到checkpoints文件中，以便于TensorBoard读取
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess,os.path.join(log_dir,'model'),training_steps)
    summary_writer.close()

def main(_):
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

    final_result = train(mnist)
    visualisation(final_result)

if __name__ == '__main__':
    tf.app.run()
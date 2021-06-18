import tensorflow as tf
#from tensorflow.python.framework import graph_util
import os


def ckpt2pb(ckpt_filename,save_dir,output_node_names):
    with tf.Graph().as_default() as graph_old:
        sess = tf.InteractiveSession()
        #ckpt_filename = './model.ckpt'

        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(ckpt_filename+'.meta',clear_devices=True)

        saver.restore(sess, ckpt_filename)

        constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names)
        constant_graph = tf.graph_util.remove_training_nodes(constant_graph)
        basename=os.path.basename(ckpt_filename)
        with tf.io.gfile.GFile(os.path.join(save_dir,'%s.pb'%basename), mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        sess.close()

# with tf.Session() as sess:
with tf.Graph().as_default() as g:
    a=tf.constant(1)
    b=tf.constant(2)
    c=a+b

    sess=tf.Session()
    ckpt_filename='iris_model1/model.ckpt-10000'
    saver = tf.train.import_meta_graph(ckpt_filename + '.meta')
    saver.restore(sess,ckpt_filename)

    save_dir='iris_model1'
    #meta=tf.train.import_meta_graph(ckpt_filename+'.meta')
    #vs=tf.global_variables()
    # graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
    # node_list = [n.name for n in graph_def.node]
    # for node in node_list:
    #     print("node_name", node)

    # sess.close()
    model=tf.train.load_checkpoint(ckpt_filename)
    #['dnn/logits/bias/part_0']
    output_node=['dnn/logits/bias/part_0']
    # [v.name.split(':')[0] for v in vs]
    ckpt2pb(ckpt_filename,save_dir,output_node)

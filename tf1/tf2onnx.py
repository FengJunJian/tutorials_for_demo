import os
import shutil
import tensorflow as tf
from assets.tensorflow_to_onnx_example import create_and_train_mnist
import torch.optim as optim



def save_model_to_saved_model(sess, input_tensor, output_tensor):
    from tensorflow.saved_model import simple_save
    save_path = r"./output/saved_model"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    simple_save(sess, save_path, {input_tensor.name: input_tensor}, {output_tensor.name: output_tensor})

print("please wait for a while, because the script will train MNIST from scratch")
tf.reset_default_graph()
sess_tf, saver, input_tensor, output_tensor = create_and_train_mnist()
print("save tensorflow in format \"saved_model\"")
save_model_to_saved_model(sess_tf, input_tensor, output_tensor)
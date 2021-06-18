import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Flatten,Dropout,MaxPool2D,AveragePooling2D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import cv2
import os
import numpy as np

from sklearn.utils import class_weight

check_path='logs'
training_flag=True
if not os.path.exists(check_path):
    os.mkdir(check_path)
data_path='E:/fjj/dataset_processing/Classification'
target_size=(224,224)#(H,W) ,(60,228)
batch_size=32
train_generator=image.ImageDataGenerator(rescale=1./255,).flow_from_directory(directory=data_path,
                                                                              target_size=target_size,
                                                                              batch_size=batch_size)
ind_classname=list(train_generator.class_indices.keys())
weight_dict={}
# for c in np.unique(train_generator.classes):
#     count_dict[c]=np.sum(train_generator.classes==c)
class_weight = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_generator.classes),
                                                 train_generator.classes)
for i in range(len(np.unique(train_generator.classes))):
    weight_dict[i]=class_weight[i]

valid_generator=image.ImageDataGenerator(rescale=1./255).flow_from_directory(directory=data_path,
                                                                             target_size=target_size,
                                                                             batch_size=batch_size,)
total=train_generator.n

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5',save_weights_only=True),
    tf.keras.callbacks.TensorBoard(log_dir='logs',update_freq=batch_size*10),
]
#checkpoint=ModelCheckpoint(check_path,verbose=1,save_freq=1)
# with tf.device('cpu'):
    # base_model = ResNet50(input_shape=(224,224,3),include_top=False,weights='resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
base_model = VGG16(input_shape=target_size+(3,), include_top=False,
                      weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

x = base_model.output

x = GlobalAveragePooling2D()(x)
# x=MaxPool2D()(x)
# a3=AveragePooling2D()(x)
x = Flatten(name='flatten')(x)
x = Dense(2048, activation='relu', name='fc1')(x)
# if training_flag:
#     x=Dropout(0.5)(x)
x = Dense(2048, activation='relu', name='fc2')(x)
# if training_flag:
#     x=Dropout(0.5)(x)
predictions = Dense(15, activation='softmax', name='predictions')(x)
# x=Flatten()(x)
# predictions = Dense(15, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions,trainable=True)

if training_flag:
    initial_epoch=5
    for layer in base_model.layers:
        layer.trainable = False
    model.summary()
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['acc'])
    model.fit_generator(generator=train_generator, epochs=initial_epoch, validation_data=valid_generator,
                        validation_freq=1,
                        callbacks=my_callbacks, class_weight=weight_dict,workers=2,initial_epoch=0)
    model.save('logs/resnet50_model1.h5')

    base_model.trainable=True
    frozen_layers = ['block1', 'block2']
    for layer in base_model.layers:
        for frozen_layer in frozen_layers:
            if frozen_layer in layer.name:
                layer.trainable = False
    model.summary()
    model.fit_generator(generator=train_generator, epochs=50, validation_data=valid_generator,
                        validation_freq=1,
                        callbacks=my_callbacks, class_weight=weight_dict, workers=2, initial_epoch=initial_epoch)
else:
    # model.load_weights('model.h5')
    for i in range(3000):
        imgs,labels=train_generator.next()
        results=model.predict(imgs/1.0/255)
        p_inds=np.argmax(results,1)
        g_inds=np.argmax(labels,1)
        for i in range(len(labels)):
            print('#######################################')
            print('original:',ind_classname[g_inds[i]])
            print('prediction:',ind_classname[p_inds[i]])

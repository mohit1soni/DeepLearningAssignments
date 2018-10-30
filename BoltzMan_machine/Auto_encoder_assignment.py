import numpy as np
import tensorflow as tf
import sys
import os
import keras

# Code Starts Here
from keras.datasets import cifar10
(X_train, Y_train), (X_test, Y_test)=cifar10.load_data()

#Defining Parameters
width = 32
height = 32
batch_size = 100
nb_epochs = 1000
code_length = 128


#Generating the tensor flow graph for the model 
graph = tf.Graph()

with graph.as_default():
#    Definign Global Step 
    global_step=tf.Variable(0,trainable=False)
#    Input Batch change
    input_images=tf.placeholder(tf.float32,shape=(batch_size,height,width,3))

#   Making of the convolution layer extracting features 
    conv_1=tf.layers.conv2d(input_images,filters=32,kernel_size=(3,3),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            activation=tf.nn.tanh,name="conv1")
#    Convolution output flattened 
    conv_out=tf.layers.flatten(conv_1,name="conv1_out")
#    Encoding Layer changing activation function to leaky relu
    encoder=tf.layers.dense(conv_out,units=code_length,activation=tf.nn.leaky_relu,name="encoder")
#    Encoder_output layer changing activation function to leaky relu
    encoder_out=tf.layers.dense(encoder,units=(height-2)*(width-2)*3,
                                activation=tf.nn.leaky_relu,name="encoder_out")
#    Deconvolution Layer
    deconv_input=tf.reshape(encoder_out,shape=(batch_size,height-2,width-2,3),
                                name="deconv_inp")
    deconv_1=tf.layers.conv2d_transpose(deconv_input,filters=3,
                                        kernel_size=(3,3),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        activation=tf.sigmoid,name="deconv1")
#    Making Output Images
    output_images=tf.cast(tf.reshape(deconv_1,(batch_size,height,width,3))*255.0,tf.uint8)
    
#    Loss consruction
    loss=tf.nn.l2_loss(input_images-deconv_1)
    decay_step=int(X_train.shape[0]/(2*batch_size))
#    Traning The Model
    learning_rate=tf.train.exponential_decay(learning_rate=0.005,
                                             global_step=global_step,
                                             decay_steps=decay_step,
                                             decay_rate=0.95,
                                             staircase=True)
    trainer=tf.train.RMSPropOptimizer(learning_rate)
    training_step=trainer.minimize(loss,name="Trainer")

from numba import jit
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
import multiprocessing
import matplotlib.pyplot as plt

use_gpu = True
config = tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(), 
                        inter_op_parallelism_threads=multiprocessing.cpu_count(), 
                        allow_soft_placement=True, 
                        device_count = {'CPU' : 2, 
                                        'GPU' : 2 if use_gpu else 0})

session = tf.InteractiveSession(graph=graph, config=config)

tf.global_variables_initializer().run()

@jit
def create_batch(t, gray=False):
    X = np.zeros((batch_size, height, width, 3 if not gray else 1), dtype=np.float32)
        
    for k, image in enumerate(X_train[t:t+batch_size]):
        if gray:
            X[k, :, :, :] = rgb2gray(image)
        else:
            X[k, :, :, :] = image / 255.0
        
    return X



for e in range(nb_epochs):
    total_loss = 0.0
    if e==0:
        print('Epoch {} - Total loss: {}'.format(e+1, total_loss))
    for t in range(0, X_train.shape[0], batch_size):
        feed_dict = {
            input_images: create_batch(t)
        }

        _, v_loss = session.run([training_step, loss], feed_dict=feed_dict)
        total_loss += v_loss
    print('Epoch {} - Total loss: {}'.format(e+1, total_loss)) 

feed_dict = {input_images: create_batch(0)}

oimages = session.run([output_images], feed_dict=feed_dict)

fig, ax = plt.subplots(2, 2, figsize=(3, 3))

for y in range(3):
    ax[0, y].get_xaxis().set_visible(False)
    ax[0, y].get_yaxis().set_visible(False)
    ax[1, y].get_xaxis().set_visible(False)
    ax[1, y].get_yaxis().set_visible(False)

    ax[0, y].imshow(X_train[y])
    ax[1, y].imshow(oimages[0][y])


session.close()





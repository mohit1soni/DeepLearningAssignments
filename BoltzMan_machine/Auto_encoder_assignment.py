import numpy as np
import tensorflow as tf
import sys
import os
import keras
# Code Starts Here
from keras.datasets import cifar10
X_train,Y_train=cifar10.load_data()

#Defining Parameters
width = 32
height = 32
batch_size = 10
nb_epochs = 15
code_length = 128


#Generating the tensor flow graph for the model 
graph = tf.Graph()

with graph.as_default():
#    Definign Global Step 
    global_step=tf.Variable(0,trainable=False)
#    Input Batch change
    input_images=tf.placeholder(tf.float32,shape=(batch_size,height,width,3))

#   Making of the convolution layer extracting features 
    conv_1=tf.layers.conv2d(input=input_images,filter=32,kernel_size=(3,3),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            activation=tf.nn.tanh,name="conv1")
#    Convolution output flattened
    conv_out=tf.layers.flatten(conv_1,name="cpmv1_out")
#    Encoding Layer
    
    
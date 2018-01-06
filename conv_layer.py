import tensorflow as tf
import numpy as np

#strides=(1,1,1,(width and height both equal one)1)
def conv_layer(input_tensor,name,kernel_size,n_output_channels,padding_mode='SAME',strides=(1,1,1,1)):
    with tf.variable_scope(name):
        ##  input tensor shape:
        ##  [batch x width x height x channels_in]
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]
        weights_shape = list(kernel_size) + [n_input_channels, n_output_channels]
        weights = tf.get_variable(name='_weights', shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[n_output_channels]))
        print(biases)
        conv = tf.nn.conv2d(input=input_tensor,filter=weights, strides=strides, padding=padding_mode)
        print(conv)
        conv = tf.nn.bias_add(conv, biases, name='net_pre-activation')
        print(conv)
        conv = tf.nn.relu(conv, name='activation')
        print(conv)
        return conv

    pass

'''
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    conv_layer(x, name='convtest',kernel_size=(3, 3),n_output_channels=32)
del g, x
'''

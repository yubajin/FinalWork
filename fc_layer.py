import tensorflow as tf
import numpy as np
def fc_layer(input_tensor,name,n_output_units,activation_fn=None):
    with tf.variable_scope(name):
        # input tensor shape:  [batch x width x height x channels_in]
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, shape=(-1, n_input_units))  # -1: batch size
            # total : [batch x width x height x channels_in]
            # n_input_units:[ width x height x channels_in]
            weights_shape = [n_input_units, n_output_units]
            weights = tf.get_variable(name='_weights', shape=weights_shape)
            print(weights)
            biases = tf.get_variable(name='_biases',initializer=tf.zeros(shape=[n_output_units]))
            print(biases)
            layer = tf.matmul(input_tensor, weights)
            print(layer)
            layer = tf.nn.bias_add(layer, biases, name='net_pre-activaiton')
            print(layer)
            if activation_fn is None: return layer
            layer = activation_fn(layer, name='activation')
            print(layer)
            return layer

'''
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32,shape=[None, 28, 28, 1])
    fc_layer(x, name='fctest', n_output_units=32,activation_fn=tf.nn.relu)
    del g, x
'''
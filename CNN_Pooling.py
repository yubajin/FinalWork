'''
Code outline:
input X data and y data
process x and y data:
X: reshape
y : One-hot encoding transform

➢ 1st layer: Conv_1
➢ MaxPooling
➢ 2n layer: Conv_2
➢ MaxPooling
➢ 3rd layer: Fully Connected
➢ Dropout
➢ 4th layer: Fully Connected (linear activation) ➢ Prediction
➢ Loss function
➢ Optimization
➢ Computing the prediction accuracy
'''

import tensorflow as tf
import numpy as np
import os
from . import conv_layer
from . import fc_layer

def buid_cnn():
    ##Placeholders for X and y:
    tf_x = tf.placeholder(tf.float32,shape=[None,784],name='tf_x')
    tf_y = tf.placeholder(tf.int32,shape=[None],name='tf_y')

    #reshape x to a 4D tensor:
    #[batchsize,width,height,1]
    tf_x_image = tf.reshape(tf_x, shape=[-1,28,28,1],name='tf_x_reshape')

    #one-hot encoding
    tf_y_onehot = tf.one_hot(indices=tf_y,depth=10,dtype=tf.float32,name='tf_y_onehot')

    #1st layer:Conv_1
    print('\nBuilding 1stlayer:')
    h1 = conv_layer(tf_x_image, name='conv_1', kernel_size=(5, 5), padding_mode='VALID', n_output_channels=32)

    ## MaxPooling
    h1_pool = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    ## 2n layer: Conv_2
    print('\nBuilding 2nd layer:')
    h2 = conv_layer(h1_pool, name='conv_2',
                    kernel_size=(5, 5), padding_mode='VALID', n_output_channels=64)

    ## MaxPooling
    h2_pool = tf.nn.max_pool(h2,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    ## 3rd layer: Fully Connected print('\nBuilding 3rd layer:')
    h3 = fc_layer(h2_pool, name='fc_3',n_output_units=1024, activation_fn=tf.nn.relu)

    ## Dropout
    keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
    h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob, name='dropout_layer')

    ## 4th layer: Fully Connected (linear activation) print('\nBuilding 4th layer:')
    h4 = fc_layer(h3_drop, name='fc_4', n_output_units=10, activation_fn=None)

    ## Prediction
    predictions = {
        'probabilities':
            tf.nn.softmax(h4, name='probabilities'),
        'labels':
            tf.cast(tf.argmax(h4, axis=1), tf.int32, name='labels')
    }

    ## Loss Function and Optimization
    cross_entropy_loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits=h4, labels=tf_y_onehot),
    name = 'cross_entropy_loss')

    ## Optimizer:
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(learning_rate=0.1,global_step=global_step,
                                               decay_steps=100, decay_rate=0.96,staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')

    ## Computing the prediction accuracy
    correct_predictions = tf.equal(predictions['labels'], tf_y, name='correct_preds')
    accuracy = tf.reduce_mean( tf.cast(correct_predictions, tf.float32), name='accuracy')
    pass

def save(saver, sess, epoch, path='./model/'):
  if not os.path.isdir(path):
      os.makedirs(path)
      print('Saving model in %s' % path)
  saver.save(sess, os.path.join(path, 'cnn-model.ckpt'), global_step=epoch)

def load(saver, sess, path, epoch):
    print('Loading model from %s' % path)
    saver.restore(sess, os.path.join(
    path, 'cnn-model.ckpt-%d' % epoch))

def train(sess, training_set, validation_set=None, initialize=True, epochs=20, shuffle=True,
    dropout=0.5, random_seed=None):
    X_data = np.array(training_set[0])
    y_data = np.array(training_set[1])
    training_loss = []

    ## initialize variables if initialize:
    sess.run(tf.global_variables_initializer())

    np.random.seed(random_seed)
    # for shuflling in batch_generator
    for epoch in range(1, epochs + 1):
        batch_gen = batch_generator(X_data, y_data, shuffle=shuffle)
        avg_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(batch_gen):
            feed = {'tf_x:0': batch_x,
                    'tf_y:0': batch_y, 'fc_keep_prob:0':
                        dropout}
            loss, _ = sess.run(['cross_entropy_loss:0', 'train_op'],
                               feed_dict=feed)
            avg_loss += loss
        training_loss.append(avg_loss / (i + 1))

        print('Epoch %02d Training Avg. Loss: %7.3f' % (epoch, avg_loss), end=' ')

        if validation_set is not None:
            feed = {'tf_x:0': validation_set[0], 'tf_y:0':
                validation_set[1], 'fc_keep_prob:0': 1.0}

            valid_acc = sess.run('accuracy:0', feed_dict=feed)
            print(' Validation Acc: %7.3f' % valid_acc)
        else:
            print()

def predict(sess, X_test, return_proba=False):
    feed = {'tf_x:0': X_test,'fc_keep_prob:0': 1.0}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    else:
        return sess.run('labels:0', feed_dict=feed)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


# initialize weights
def init_weights(shape):
    # Outputs random values from a truncated normal distribution
    init_random_dist = tf.truncated_normal(shape,stddev=0.1)
    # A variable maintains state in the graph across calls to run()
    return tf.Variable(init_random_dist)

    
# initialize bias
def init_bias(shape):
    # creates a constant tensor for bias
    init_bias_vals = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_vals)


# conv 2d
def conv2d(x,w):
    # x = input tensor [batch #,height,width,channels]
    # w = kernal [filter height, filter width, channels in, channels out
    # Computes a 2-D convolution given 4-D input and filter tensors
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME') # same = 0 for padding


# pooling layer
def max_pool_2by2(x):
    # x = [batch,h,w,c]
    # ksize = size of window for each dimension of input tensor: [1 batch,2 height,2 wide,1 channel]
    # strides = slide of window for each dimension of input tensor: [1 batch,2 height,2 wide,1 channel]
    # strides = movement of window
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# convolutional layer
def convolutional_layer(input_x,shape):
    w = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,w)+b)


# normal_layer
def normal_full_layer(input_layer,size):
    input_size = int(input_layer.get_shape()[1])
    w = init_weights([input_size,size])
    b = init_bias([size])
    return tf.matmul(input_layer,w)+b


# create placeholders
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])


# create layers
# reshape flattened array into image again
x_image = tf.reshape(x,[-1,28,28,1])


# lst convolutional layer
# patch size = 5,5  channel = 1  output channels = 32
convo_1 = convolutional_layer(x_image,shape=[5,5,1,32])


# 1st pooling layer
conv_1_pooling = max_pool_2by2(convo_1)


# 2nd convolutional layer
convo_2 = convolutional_layer(conv_1_pooling,shape=[5,5,32,64])


# 2nd pooling layer
conv_2_pooling = max_pool_2by2(convo_2)


# flatten out result layer to connect it to fully connected layer
conv_2_flat = tf.reshape(conv_2_pooling,[-1,7*7*64])
full_layer_one = tf.nn.relu(normal_full_layer(conv_2_flat,1024))


# droppout
hold_prob = tf.placeholder(tf.float32)
# hold prob = probability a neuron is held during droppout
full_one_droppout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
y_pred = normal_full_layer(full_one_droppout,10)


# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))


# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()

steps = 800

with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        batch_x,batch_y = mnist.train.next_batch(50)
        sess.run(train,feed_dict={x:batch_x, y_true:batch_y, hold_prob:0.5})

        if i%100 == 0:
##            print('On step: {}'.format(i))
##            print('Accuracy:')
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
##            print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels,hold_prob:1.0}))
##            for n in mnist.test.labels[10:20]:
##                print(np.where(n==1))
####                print (matches)
##            print('\n')


for i in range(10,20):
    single_image = mnist.test.images[i].reshape(28,28)

    plt.imshow(single_image,cmap='gist_gray')
    plt.show()

    







                     

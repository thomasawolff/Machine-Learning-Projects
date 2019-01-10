##import numpy as np
##import matplotlib.pyplot as plt
##import tensorflow as tf
##from tensorflow.examples.tutorials.mnist import input_data
##mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

from nueralCNN import *

for i in range(10,20):
    single_image = mnist.test.images[i].reshape(28,28)

    plt.imshow(single_image,cmap='gist_gray')
    plt.show()



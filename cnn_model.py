import tensorflow as tf
import net

class HGRNetwork(net.Net):
    def __init__(self, batch_size=4):
        net.Net.__init__(self)
        self.batch_size = batch_size

    def predictSignal(self, x):
        """
        x: Input Tensor
        """

        with tf.variable_scope('generator'):
            self.conv1 = self.conv_layer(x, 16, stride=[1, 4, 4, 1], name='conv1')
            self.conv2 = self.conv_layer(self.conv1, 8, stride=[1, 4, 4, 1], name='conv2')
            self.conv3 = self.conv_layer(self.conv2, 4, stride=[1, 4, 4, 1], name='conv3')
            self.conv3 = tf.reshape(self.conv3, [self.batch_size, -1])
            self.flat1 = self.two_d_layer(self.conv3, 64, 4, name='flat1')
            self.flat2 = self.two_d_layer(self.flat1, 4, 3, act=None, name='flat2')

        return self.flat2

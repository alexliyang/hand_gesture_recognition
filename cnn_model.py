import tensorflow as tf
import net

class HGRNetwork(net.Net):
    def __init__(self, batch_size=4):
        net.Net.__init__(self)
        self.batch_size = batch_size
        self.is_training = True
        self.epsilon = 1e-12

    def predictSignal(self, x):
        """
        x: Input Tensor
        """

        with tf.variable_scope('CNNModel'):
            self.conv1 = self.conv_layer(x, 16, stride=[1, 2, 2, 1], norm=False, name='conv1')
            self.conv2 = self.conv_layer(self.conv1, 8, stride=[1, 4, 4, 1], name='conv2')
            self.conv3 = self.conv_layer(self.conv2, 4, stride=[1, 4, 4, 1], name='conv3')
            self.conv3 = tf.reshape(self.conv3, [self.batch_size, -1])
            self.flat1 = self.two_d_layer(self.conv3, 64, 4, name='flat1')
            self.flat2 = self.two_d_layer(self.flat1, 4, 2, act=None, name='flat2')
            self.flat2 = tf.nn.sigmoid(self.flat2)

        return self.flat2

    def batch_normalize(self, inputs, num_maps, decay=.9):
        with tf.variable_scope("batch_normalization"):
            # Trainable variables for scaling and offsetting our inputs
            scale = tf.get_variable('scale', initializer=tf.ones([num_maps], dtype=tf.float32), trainable=True)
            tf.summary.histogram('scale', scale)
            offset = tf.get_variable('offset', initializer=tf.constant(.1, shape=[num_maps]), trainable=True)
            tf.summary.histogram('offset', offset)

            # Mean and variances related to our current batch
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])

            # Create an optimizer to maintain a 'moving average'
            ema = tf.train.ExponentialMovingAverage(decay=decay)

            def ema_retrieve():
                return ema.average(batch_mean), ema.average(batch_var)

            # If the net is being trained, update the average every training step
            def ema_update():
                ema_apply = ema.apply([batch_mean, batch_var])

                # Make sure to compute the new means and variances prior to returning their values
                with tf.control_dependencies([ema_apply]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            # Retrieve the means and variances and apply the BN transformation
            mean, var = tf.cond(tf.equal(self.is_training, True), ema_update, ema_retrieve)
            bn_inputs = tf.nn.batch_normalization(inputs, mean, var, offset, scale, self.epsilon)

        return bn_inputs
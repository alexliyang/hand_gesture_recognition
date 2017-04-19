import tensorflow as tf

class Net(object):
    
    def conv_layer(self, t, nf, stride, name, kernel_shape=[3, 3], act=tf.nn.relu, norm=True, pad='SAME'):

        """
        Performs Convolution Operations
        t: input tensor to perform convolution on
        nf: number of filter maps
        name: operation name
        act: nonlinear function
        norm: whether or not to normalize input distribution
        pad: padding type
        """

        with tf.variable_scope(name): 
            shape = kernel_shape + [t.get_shape().as_list()[3]] + [nf]

            # Create filters and perform convolution
            filters = tf.get_variable('weights', initializer=tf.truncated_normal(shape, stddev=.02))
            
            # Not performing logs as of now. Actually, I can't :P
            # tf.summary.histogram('weights', filters)
            
            maps_ = tf.nn.conv2d(t, filters, padding=pad, strides=stride)

            # Add bias
            bias = tf.get_variable('bias', initializer=tf.constant(.0, shape=[nf]))
            # tf.summary.histogram('bias', bias)
            maps = tf.nn.bias_add(maps_, bias)

            maps = self.batch_normalize(maps, nf) if norm else maps
            maps = act(maps) if act is not None else maps

            return maps

    def two_d_layer(self, t, nw, nn, name, act=tf.nn.relu, norm=True, pad='SAME'):

        """
        Performs matrix multiplications for hidden layers operations
        t: input tensor to perform convolution on
        nn: number of neurons
        name: operation name
        act: nonlinear function
        norm: whether or not to normalize input distribution
        pad: padding type
        """

        with tf.variable_scope(name): 
            shape = [nw] + [nn]

            # Create Hypothesis
            hyp = tf.get_variable('weights', initializer=tf.truncated_normal(shape, stddev=.02))
            
            bias = tf.get_variable('bias', initializer=tf.constant(.0, shape=[nn]))

            res = tf.add(tf.matmul(t, hyp), bias)

            res = act(res) if act is not None else res

            return res

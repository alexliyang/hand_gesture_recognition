import logging
import os
import tensorflow as tf
import time
import glob

class Trainer:
    def __init__(self, session, hgr_network, opts):
        self.hgr_network = hgr_network
        self.session = session

        # Assign each option as self.option_title = value
        for key, value in opts.items():
            eval_string = "self.%s = %s" % (key, value)
            exec(eval_string)

    def train(self):
        # Set initial training shapes and placeholders
        x_shape = [None, self.training_height, self.training_width, 1]
        y_shape = [None, 3] 

        x_ph = tf.placeholder(dtype=tf.float32, shape=x_shape, name='grayscale_placeholder')
        y_ph = tf.placeholder(dtype=tf.float32, shape=y_shape, name='label_placeholder')


        sample = self.hgr_network.predictSignal(x_ph)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=sample, labels=y_ph, name='loss')

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss, global_step=global_step)

        # Creating batches
        example, label = self.next_example(height=self.training_height, width=self.training_width)

        capacity = self.batch_size * 3
        input_grayscale, labels = tf.train.batch([example, label], 
                                    self.batch_size,
                                    num_threads=4,
                                    capacity=capacity)

        # Start session and begin threading
        self.session.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()

        for i in range(self.iterations):
            try:
                # Training
                feed_dict = {x_ph: input_grayscale.eval(), y_ph: labels.eval()}
                print("Hello Batching")
                _, batch_loss = self.session.run([train_op, loss], feed_dict=feed_dict)
                print("Hello Training")

                # Print current epoch number and errors if warranted
                if i % self.print_every == 0:
                    log1 = "Epoch %06d || Total Loss %.010f || " % (i, total_loss)
                    print(log1)

                # Save a checkpoint of the model
                if i % self.save_every == 0:
                    model_path = './HGRModel_%s' % time.time()
                    self.__save_model(saver, model_path)

            except tf.errors.OutOfRangeError:
                next
            finally:
                coord.request_stop()
        coord.join(threads)

        # Save the trained model and close the tensorflow session
        model_path = '/HGRModel_%s' % time.time()
        self.__save_model(saver, model_path)
        self.session.close()

    @staticmethod
    # Returns an image in its grayscale 
    def next_example(height, width):
        # Ops for getting training images, from retrieving the filenames to reading the data
        filenames = tf.train.match_filenames_once('./sampleImages/*.jpg')
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.WholeFileReader()
        filename, file = reader.read(filename_queue)
        print filename.eval()   

        img = tf.image.decode_jpeg(file, channels=3)
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.resize_images(img, [height, width])

        filename_split = tf.string_split([filename], delimiter='/')
        label_id = tf.string_to_number(tf.substr(filename_split.values[1], 
                                            0, 1), out_type=tf.int32)
        label = tf.one_hot(
                    label_id, 
                    3, 
                    on_value=1.0, 
                    off_value=0.0, 
                    dtype=tf.float32)
        print("Hello from batches")
        
        return img, label

    def __save_model(self, saver, path):
        print("Proceeding to save weights at '%s'" % path)
        saver.save(self.session, path)
        print("Weights have been saved.")

import logging
import os
import tensorflow as tf
import time
import glob
import numpy as np

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
        y_shape = [None, 2] 

        # Creating batches
        example, label = self.next_example(height=self.training_height, width=self.training_width)

        input_grayscale, labels = tf.train.shuffle_batch([example, label], 
                                    self.batch_size,
                                    num_threads=4,
                                    capacity=20000,
                                    min_after_dequeue=10000)

        sample = self.hgr_network.predictSignal(input_grayscale)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=sample, labels=labels, name='loss')
        model_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CNNModel')

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss, global_step=global_step)

        # Start session and begin threading
        init_op = tf.global_variables_initializer(),tf.local_variables_initializer()
        self.session.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        saver = tf.train.Saver(model_weights)

        for i in range(self.iterations):
            try:
                _, batch_loss = self.session.run([train_op, loss])

                # Print current epoch number and errors if warranted
                if i % self.print_every == 0:
                    total_loss = np.sum(batch_loss)
                    log1 = "Epoch %06d || Total Loss %.010f || " % (i, total_loss)
                    logging.info(log1)
                    print log1

                # Save a checkpoint of the model
                if i % self.save_every == 0:
                    model_path = './HGRModel/hgr_model'
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
        all_files = [tf.train.match_filenames_once(
                                'training_data/1/TrainningSample%d/*.jpg' % x) for x in range(1,5)]
        

        filenames = tf.concat(all_files, 0)
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.WholeFileReader()
        filename, file = reader.read(filename_queue)

        img = tf.image.decode_jpeg(file, channels=3)
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.resize_images(img, [height, width])

        filename_split = tf.string_split([filename], delimiter='/')
        
        label_id = tf.string_to_number(tf.substr(filename_split.values[1], 
                                            0, 1), out_type=tf.int32)
        label = tf.one_hot(
                    label_id-1, 
                    2, 
                    on_value=1.0, 
                    off_value=0.0, 
                    dtype=tf.float32)
        
        return img, label

    def __save_model(self, saver, path):
        print("Proceeding to save weights at '%s'" % path)
        saver.save(self.session, path)
        print("Weights have been saved.")

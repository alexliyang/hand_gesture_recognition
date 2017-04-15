from helpers import Helpers
import logging
import os
import tensorflow as tf
import time
import glob
import generator


def next_example():
    filenames = tf.train.match_filenames_once('./sampleImages/*.jpg')
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()
    filename, file = reader.read(filename_queue)

    img = tf.image.decode_jpeg(file, channels=3)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize_images(img, [256, 256])

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

    
x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 1], name='conditional_placeholder')

# Training data retriever ops
example = next_example()
capacity = 15
batch_condition_gen, batch_condition_disc, batch_label = tf.train.batch([example_condition, example_grayscale, example_label], 
                                              5,
                                              num_threads=4,
                                              capacity=capacity)

print(batch_condition_gen)
gen = generator.Generator()
output = gen.generate(x_ph)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer(),tf.local_variables_initializer()
    # Initialize the variables (like the epoch counter).
    sess.run(init_op)
    Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    

    try:
        while not coord.should_stop():
            # Run training steps or whatever
            intermediate_image = sess.run(output, feed_dict={x_ph: batch_condition_gen.eval()})

            print intermediate_image.shape
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

# coord.join(threads)                
sess.close()
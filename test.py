import tensorflow as tf
import cnn_model

def next_example():
    # Ops for getting training images, from retrieving the filenames to reading the data
    all_files = [tf.matching_files(
                    'training_data/1/TrainningSample%d/*.jpg' % x) for x in range(1,5)]
    #all_files = [tf.train.match_filenames_once('./sampleImages/*.jpg')]

    filenames = tf.concat(all_files, 0)
    #print filenames.get_shape()
    # filenames = tf.train.match_filenames_once("training_data/1/TrainningSample1/*.jpg")
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()
    filename, file = reader.read(filename_queue)

    img = tf.image.decode_jpeg(file, channels=3)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize_images(img, [128, 128])

    filename_split = tf.string_split([filename], delimiter='/')
    
    label_id = tf.string_to_number(tf.substr(filename_split.values[2], 
                                        0, 1), out_type=tf.int32)
    label = tf.one_hot(
                label_id-1, 
                2, 
                on_value=1.0, 
                off_value=0.0, 
                dtype=tf.float32)
    
    return img, label

example, label = next_example()

minimum = 100
capacity = minimum + 2 * 3
input_grayscale, labels = tf.train.shuffle_batch([example, label], 
                            2,
                            num_threads=4,
                            capacity=capacity,
                            min_after_dequeue=minimum)


print input_grayscale.get_shape()
print labels.get_shape()
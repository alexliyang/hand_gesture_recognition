import argparse
import cnn_model
import tensorflow as tf
from trainer import Trainer

# Model hyperparamaters
opts = {
    'batch_size': 4,
    'iterations': 2000000,
    'learning_rate': 5e-5,
    'print_every': 100,
    'save_every': 10000,
    'training_height': 128,
    'training_width': 128,
}


def parse_args():
    # Parse command line arguments to assign to the global opt variable
    parser = argparse.ArgumentParser(description='colorize images using conditional generative adversarial networks')
    for opt_name, value in opts.items():
        parser.add_argument("--%s" % opt_name, default=value)

    # Update global opts variable using flag values
    args = parser.parse_args()
    for opt_name, _ in opts.items():
        opts[opt_name] = getattr(args, opt_name)

parse_args()
with tf.Session() as sess:
    # Initialize networks
    hgr_network = cnn_model.HGRNetwork(opts['batch_size'])

    # Train them
    t = Trainer(sess, hgr_network, opts)
    t.train()

import argparse
import cnn_model
import tensorflow as tf
from trainer import Trainer

# Model hyperparamaters
opts = {
    'batch_size': 2,
    'iterations': 1000000,
    'learning_rate': 5e-2,
    'print_every': 1,
    'save_every': 100,
    'training_height': 256,
    'training_width': 256,
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

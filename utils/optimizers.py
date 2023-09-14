import tensorflow as tf
import tensorflow_addons as tfa


def get_optimizer(CFG):
    opt_name = CFG.optimizer
    lr = CFG.lr
    if opt_name == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
    else:
        raise ValueError("Wrong Optimzer Name")
    return opt

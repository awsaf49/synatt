import tensorflow as tf
import tensorflow_addons as tfa

# LOSS
def get_loss(CFG):
    loss_name = CFG.loss
    if loss_name == 'CCE':
        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=CFG.label_smoothing)
    elif loss_name == 'BCE':
        loss = tf.keras.losses.BinaryCrossentropy(
            label_smoothing=CFG.label_smoothing)
    else:
        raise NotImplemented
    return loss

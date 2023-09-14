import tensorflow as tf
import os
import wandb
from utils.schedulers import get_lr_scheduler
import sys
import numpy as np


def print_msg(message, line_break=True):
    if line_break:
        sys.stdout.write(message + '\n')
    else:
        sys.stdout.write(message)

    sys.stdout.flush()


def get_callbacks(CFG, monitor='val_acc'):
    # model checkpoint
    ckpt_dir = os.path.join(CFG.output_dir, 'ckpt')
    if not CFG.all_data:
        ckpt_path = 'fold-%i.h5' % (CFG.fold)
    else:
        ckpt_path = 'model.h5'
    ckpt_path = os.path.join(ckpt_dir, ckpt_path)
    sv = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor=monitor,
        verbose=CFG.verbose,
        save_best_only=not CFG.all_data,
        save_weights_only=False,
        mode=CFG.monitor_mode,
        save_freq='epoch')
    # learning rate scheduler
    lr_scheduler = get_lr_scheduler(CFG.batch_size * CFG.replicas, CFG=CFG)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        lr_scheduler, verbose=False)
    callbacks = [sv, lr_callback]
    # w&b
    if CFG.wandb:
        WandbCallback = wandb.keras.WandbCallback(save_model=False)
        callbacks.append(WandbCallback)
    return callbacks

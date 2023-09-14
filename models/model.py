# MODELING
from sklearn import metrics
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import efficientnet.tfkeras as efn
from keras_cv_attention_models import (
    nfnets,
    resnet_family,
    resnest,
    efficientnet)
import resnet_rs
import tfimm
from utils.metrics import get_metrics
from utils.losses import get_loss
from utils.optimizers import get_optimizer


def fix_bn(model, disable=True, freeze=True):
    for layer in tqdm(model.layers, desc='fixing_bn '):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            if disable:
                ch = layer.input_shape[-1]
                layer.set_weights([np.array([1] * ch, dtype=np.float32),
                                   np.array([0] * ch, dtype=np.float32),
                                   np.array([0] * ch, dtype=np.float32),
                                   np.array([1] * ch, dtype=np.float32)])
            if freeze:
                layer.trainable = False
        elif isinstance(layer, tf.keras.Sequential):
            fix_bn(layer, disable=disable, freeze=freeze)
    return


# models which do not have gap ex: transformers
NO_GAP_MODELS = ['convnext']


def check_gap(model_name):
    for gap_model in NO_GAP_MODELS:
        if gap_model in model_name:
            return False
    return True


def get_base(CFG):
    model_name = CFG.model_name
    DIM = CFG.img_size
    pretrain = CFG.pretrain
    if 'EfficientNet' in model_name and 'V2' not in model_name:
        base = getattr(efn, model_name)(input_shape=(*DIM, 3),
                                        include_top=False,
                                        weights=pretrain,
                                        )
    # `pretrained` = [None, "imagenet", "imagenet21k", "imagenet21k-ft1k"]
    elif 'EfficientNetV' in model_name:
        base = getattr(efficientnet, model_name)(input_shape=(*DIM, 3),
                                                 pretrained=pretrain,
                                                 # drop_connect_rate=0.2,
                                                 num_classes=0, )
    elif 'NFNet' in model_name:
        base = getattr(nfnets, model_name)(input_shape=(*DIM, 3),
                                           pretrained=pretrain,
                                           num_classes=0)
    elif 'ResNetRS' in model_name:
        base = getattr(resnet_rs, model_name)(input_shape=(*DIM, 3),
                                              weights='imagenet',
                                              include_top=False)
    elif 'ResNet' in model_name or 'ResNeXt' in model_name or 'RegNet' in model_name:
        base = getattr(resnet_family, model_name)(input_shape=(*DIM, 3),
                                                  pretrained=pretrain,
                                                  num_classes=0)
    elif 'ResNest' in model_name:
        base = getattr(resnest, model_name)(input_shape=(*DIM, 3),
                                            pretrained=pretrain,
                                            num_classes=0)
    elif 'convnext' in model_name:
        base = tfimm.create_model(model_name, pretrained=True, nb_classes=0)
    else:
        raise NotImplemented
    return base


def build_model(CFG, compile_model=True, steps_per_execution=None):
    base = get_base(CFG)
    inp = tf.keras.layers.Input(shape=(*CFG.img_size, 3))
    out = base(inp)
    if check_gap(CFG.model_name):
        out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(CFG.num_features, activation='selu')(out)
    out = tf.keras.layers.Dense(
        len(CFG.class_names), activation=CFG.final_act)(out)
    model = tf.keras.Model(inputs=inp, outputs=out)
    if CFG.disable_bn or CFG.freeze_bn:
        model = fix_bn(model, disable=CFG.disable_bn, freeze=CFG.freeze_bn)
    if compile_model:
        # optimizer
        opt = get_optimizer(CFG)
        # loss
        loss = get_loss(CFG)
        # metric
        metrics = get_metrics(CFG)
        # compile
        model.compile(optimizer=opt,
                      loss=loss,
                      metrics=metrics,
                      steps_per_execution=steps_per_execution)
    return model

import sklearn
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

# METRIC


def get_metrics(CFG):
    acc = tf.keras.metrics.CategoricalAccuracy(name='acc')
    f1 = tfa.metrics.F1Score(
        num_classes=len(
            CFG.class_names),
        average='macro',
        threshold=None)
    auc = tf.keras.metrics.AUC(curve='ROC', name='auc')
    return [acc, f1, auc]


class MetricFactory(object):
    """Metrics to calculate classification scores"""

    def __init__(self, metrics={'acc': sklearn.metrics.accuracy_score,
                                'f1_score': sklearn.metrics.f1_score,
                                'auc': sklearn.metrics.roc_auc_score}):
        self.metrics = metrics

    def __call__(self, y_true, y_pred):
        result = {}
        for name, metric in self.metrics.items():
            if name == 'f1_score':
                args = {'average': 'macro'}
                try:
                    result[name] = metric(
                        y_true, y_pred.argmax(
                            axis=-1), **args)
                except BaseException:
                    result[name] = -1
            elif name == 'auc':
                args = {'average': 'macro', 'multi_class': 'ovo'}
                try:
                    result[name] = metric(y_true, y_pred, **args)
                except BaseException:
                    result[name] = -1
            else:
                args = {}
                try:
                    result[name] = metric(
                        y_true, y_pred.argmax(
                            axis=-1), **args)
                except BaseException:
                    result[name] = -1
        return result


def print_dict(d):
    strings = ['{}: {:.4f}'.format(k, v) for k, v in d.items()]
    return ' | '.join(strings)

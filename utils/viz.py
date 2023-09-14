import librosa.display as lid
import sklearn
import itertools
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import warnings
warnings.simplefilter('error', UserWarning)
sns.set(style='dark')

# VISUALIZATION - SAMPLES


def plot_spec_batch(batch, n_row=4, n_col=4,
                    sample_rate=16000, hop_length=250, fmin=20, fmax=8000,
                    output_dir='', save=True):
    if isinstance(batch, tuple):
        imgs, tars = batch
    else:
        imgs = batch
        tars = None

    plt.figure(figsize=(n_col * 7, n_row * 5))
    for img_idx in range(n_row * n_col):
        plt.subplot(n_row, n_col, img_idx + 1)
        if tars is not None:
            title = f'label: {tars[img_idx].numpy().argmax().item()}'
            plt.title(title, fontsize=15)
        mel_spec = imgs[img_idx][..., 0].numpy()
        lid.specshow(mel_spec,
                     sr=sample_rate,
                     hop_length=hop_length,
                     fmin=fmin,
                     fmax=fmax,
                     x_axis='time',
                     y_axis='mel',
                     cmap='viridis'
                     )
        # plt.xticks([])
        # plt.yticks([])
    plt.tight_layout()
    if save:
        plt.savefig(
            f'{output_dir}/image/sample_specs.jpg',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0)
    plt.show()
    return


def plot_wave_batch(batch, n_row=4, n_col=4,
                    sample_rate=16000,
                    output_dir='', save=True):
    if isinstance(batch, tuple):
        imgs, tars = batch
    else:
        imgs = batch
        tars = None

    plt.figure(figsize=(n_col * 7, n_row * 5))
    for img_idx in range(n_row * n_col):
        plt.subplot(n_row, n_col, img_idx + 1)
        if tars is not None:
            title = f'label: {tars[img_idx].numpy().argmax().item()}'
            plt.title(title, fontsize=15)
        wave = imgs[img_idx].numpy().squeeze()
        lid.waveplot(wave,
                     sr=sample_rate,
                     x_axis='time',
                     )
        # plt.xticks([])
        # plt.yticks([])
    plt.tight_layout()
    if save:
        plt.savefig(
            f'{output_dir}/image/sample_waves.jpg',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0)
    plt.show()
    return

# GRAD-CAM


def plot_confusion_matrix(y_true, y_pred,
                          classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save=False,
                          output_dir=''):
    plt.rcParams["font.family"] = 'DejaVu Sans'
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
#     plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(b=False)
    if save:
        plt.savefig(
            f'{output_dir}/image/oof_cm.png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.0)
    plt.show()
    return

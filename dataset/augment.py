import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import tensorflow_io as tfio
import tensorflow.keras.backend as K
import math
import numpy as np
import os

# HELPER


def random_int(shape=[], minval=0, maxval=1):
    return tf.random.uniform(
        shape=shape, minval=minval, maxval=maxval, dtype=tf.int32)


def random_float(shape=[], minval=0.0, maxval=1.0):
    rnd = tf.random.uniform(
        shape=shape, minval=minval, maxval=maxval, dtype=tf.float32)
    return rnd


def debuglogger(filename, txt):
    if os.path.exists(filename):
        f = open(filename, "a")
    else:
        f = open(filename, "w")
    f.write(txt + '\n')
    f.close()


def GaussianNoise(img, std=[0.0025, 0.025], prob=0.5):
    std = random_float([], std[0], std[1])
    if random_float() < prob:
        GN = tf.keras.layers.GaussianNoise(stddev=std)
        img = GN(img, training=True)
    return img


def JpegCompress(img, quality=[85, 95], prob=0.5):
    if random_float() < prob:
        img = tf.image.random_jpeg_quality(img, quality[0], quality[1])
    return img


def RandomFlip(img, prob_hflip=0.5, prob_vflip=0.0):
    if random_float() < prob_hflip:
        img = tf.image.flip_left_right(img)
    if random_float() < prob_vflip:
        img = tf.image.flip_up_down(img)
    return img


def TimeFreqMask(spec, time_mask, freq_mask, prob=0.5):
    if random_float() < prob:
        spec = tfio.audio.freq_mask(spec, param=freq_mask)
        spec = tfio.audio.time_mask(spec, param=time_mask)
    return spec


def TimeShift(audio, prob=0.5):
    if random_float() < prob:
        shift = random_int(shape=[], minval=0, maxval=tf.shape(audio)[0])
        if random_float() < 0.5:
            shift = -shift
        audio = tf.roll(audio, shift, axis=0)
    return audio


def TimeReverse(audio, prob=0.5):
    if random_float() < prob:
        audio = tf.reverse(audio, axis=[0])
    return audio


def FadeInOut(audio, fade_in=0.01, fade_out=0.01, mode='linear', prob=0.5):
    if random_float() < prob:
        audio = tfio.audio.fade(
            audio,
            fade_in=fade_in,
            fade_out=fade_out,
            mode=mode)
    return audio


def spec2img(spec):
    img = tf.tile(spec[..., tf.newaxis], [1, 1, 3])
    return img


def img2spec(img):
    return img[..., 0]


def get_audio_augment(audio, CFG=None):
    if random_float() > CFG.audio_augment_prob:
        return audio
    if CFG.is_train:
        audio = TimeShift(audio, prob=CFG.timeshift_prob)
        audio = TimeReverse(audio, prob=CFG.timereverse_prob)
        audio = GaussianNoise(audio, prob=CFG.gn_prob)
    else: # when tta & inference
        # audio = TimeShift(audio, prob=0.5) # for tta
        pass
    return audio


def get_spec_augment(spec, CFG=None):
    if random_float() > CFG.spec_augment_prob:
        return spec
    spec = tf.transpose(
        img2spec(spec),
        perm=[1, 0]
    )  # [mel, time] -> [time, mel]
    if CFG.is_train:
        spec = TimeFreqMask(
            spec,
            time_mask=CFG.time_mask,
            freq_mask=CFG.freq_mask,
            prob=CFG.mask_prob)
    spec = tf.transpose(spec, perm=[1, 0])  # [time, mel] -> [mel, time]
    spec = spec2img(spec)
    if CFG.is_train:
        spec = RandomFlip(spec, prob_hflip=CFG.hflip, prob_vflip=CFG.vflip)
        # spec = GaussianNoise(spec, std=CFG.gn_std, prob=CFG.gn_prob)
        spec = JpegCompress(spec, quality=CFG.jc_quality, prob=CFG.jc_prob)
    return spec


def get_mixup(alpha=0.2, prob=0.5):
    def mixup(audios, labels, alpha=alpha, prob=prob):
        if random_float() > prob:
            return audios, labels

        audio_shape = tf.shape(audios)
        label_shape = tf.shape(labels)
        audio_ndim = tf.rank(audios)
        label_ndim = tf.rank(labels)
        batch_size = tf.cast(audio_shape[0], tf.int32)

        beta = tfp.distributions.Beta(alpha, alpha)
        lam = beta.sample(batch_size)

        lam_audio_shape = tf.concat(
            [[batch_size], tf.ones(audio_ndim - 1, dtype=tf.int32)], axis=0)
        lam_label_shape = tf.concat(
            [[batch_size], tf.ones(label_ndim - 1, dtype=tf.int32)], axis=0)

        lam_audio = tf.reshape(lam, lam_audio_shape)
        lam_label = tf.reshape(lam, lam_label_shape)

        audios = lam_audio * audios + \
            (1 - lam_audio) * tf.roll(audios, shift=1, axis=0)
        labels = lam_label * labels + \
            (1 - lam_label) * tf.roll(labels, shift=1, axis=0)

        audios = tf.reshape(audios, audio_shape)
        labels = tf.reshape(labels, label_shape)
        return audios, labels
    return mixup


def get_cutmix(alpha, prob=0.5):
    def cutmix(audios, labels, alpha=alpha, prob=prob):
        if random_float() > prob:
            return audios, labels
        audio_shape = tf.shape(audios)
        label_shape = tf.shape(labels)
        W = tf.cast(audio_shape[2], tf.int32)  # [batch, mel, time, channel]

        beta = tfp.distributions.Beta(alpha, alpha)
        lam = beta.sample(1)[0]

        audios_rolled = tf.roll(audios, shift=1, axis=0)
        labels_rolled = tf.roll(labels, shift=1, axis=0)

        r_x = random_int([], minval=0, maxval=W)
        r = 0.5 * tf.math.sqrt(1.0 - lam)
        r_w_half = tf.cast(r * tf.cast(W, tf.float32), tf.int32)

        x1 = tf.cast(tf.clip_by_value(r_x - r_w_half, 0, W), tf.int32)
        x2 = tf.cast(tf.clip_by_value(r_x + r_w_half, 0, W), tf.int32)

        # outer-pad patch -> [0, 0, 1, 1, 0, 0]
        patch1 = audios[:, :, x1:x2, :]  # [batch, mel, time, channel]
        patch1 = tf.pad(
            patch1, [[0, 0], [0, 0], [x1, W - x2], [0, 0]])  # outer-pad

        # inner-pad patch -> [1, 1, 0, 0, 1, 1]
        patch2 = audios_rolled[:, :, x1:x2, :]  # [batch, mel, time, channel]
        patch2 = tf.pad(
            patch2, [[0, 0], [0, 0], [x1, W - x2], [0, 0]])  # outer-pad
        patch2 = audios_rolled - patch2  # inner-pad = img - outer-pad

        audios = patch1 + patch2  # cutmix img

        lam = tf.cast((1.0 - (x2 - x1) / (W)),
                      tf.float32)  # no H as (y1 - y2)/H = 1
        labels = lam * labels + (1.0 - lam) * labels_rolled  # cutmix label

        audios = tf.reshape(audios, audio_shape)
        labels = tf.reshape(labels, label_shape)

        return audios, labels

    return cutmix

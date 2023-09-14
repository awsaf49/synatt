# SEEDING
import numpy as np
import random
import os
import tensorflow as tf
import tensorflow_io as tfio
from dataset.augment import get_mixup, get_cutmix, get_audio_augment, get_spec_augment, spec2img, random_float, random_int
from loguru import logger

# SEEDING


def seeding(CFG):
    SEED = CFG.seed
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    if CFG.device == 'TPU':
        os.environ['TF_CUDNN_DETERMINISTIC'] = str(SEED)
    tf.random.set_seed(SEED)
    logger.info('> SEEDING DONE')


def audio_decoder(with_labels=True, dim=1600 * 5, CFG=None):
    def get_audio(filepath):
        file_bytes = tf.io.read_file(filepath)
        audio, sr = tf.audio.decode_wav(file_bytes)
        audio = tf.cast(audio, tf.float32)
        audio = tf.squeeze(audio, axis=-1)
        if CFG.normalize:
            audio = tf.math.divide_no_nan(
                (audio - tf.math.reduce_mean(audio)),
                tf.math.reduce_std(audio))
        return audio

    def get_target(target):
        # target = tf.reshape(target, [1])
        # target = tf.one_hot(target, len(CFG.class_labels))
        target = tf.cast(target, tf.float32)
        target = tf.reshape(target, [len(CFG.class_labels)])
        return target

    def crop_or_pad(audio, crop_size, pad_mode='constant'):
        if pad_mode == 'random':
            pad_mode = random.choice(['constant', 'reflect'])
        audio_len = tf.shape(audio)[0]
        if audio_len < crop_size:
            diff_len = (crop_size - audio_len)
            pad1 = random_int([], minval=0, maxval=diff_len)
            pad2 = diff_len - pad1
            pad_len = [pad1, pad2]
            if pad_mode != 'reflect':
                audio = tf.pad(audio, paddings=[pad_len], mode=pad_mode)
            else:
                n = tf.cast(tf.math.ceil(crop_size / audio_len), tf.int32)
                audio = tf.tile(audio, [n])
                audio = audio[:crop_size]
        elif audio_len > crop_size:
            diff_len = (audio_len - crop_size)
            idx = tf.random.uniform([], 0, diff_len, dtype=tf.int32)
            audio = audio[idx: (idx + crop_size)]
        audio = tf.reshape(audio, [crop_size])
        return audio

    def decode(path):
        audio = get_audio(path)
        audio = crop_or_pad(audio, crop_size=dim, pad_mode=CFG.pad_mode)
        audio = tf.reshape(audio, [dim])
        return audio

    def decode_with_labels(path, label):
        label = get_target(label)
        return decode(path), label

    return decode_with_labels if with_labels else decode

# DATA PIPELINE


def spec_decoder(with_labels=True, dim=[128, 256], CFG=None):
    def get_spectrogram(
            audio,
            spec_shape=[
                128,
                256],
            sr=16000,
            nfft=2048,
            window=2048,
            fmin=20,
            fmax=8000):
        spec_height = spec_shape[0]
        spec_width = spec_shape[1]
        audio_len = tf.shape(audio)[0]
        hop_length = tf.cast((audio_len // (spec_width - 1)), tf.int32)
        spec = tfio.audio.spectrogram(
            audio, nfft=nfft, window=window, stride=hop_length)
        mel_spec = tfio.audio.melscale(
            spec, rate=sr, mels=spec_height, fmin=fmin, fmax=fmax)
        db_mel_spec = tfio.audio.dbscale(mel_spec, top_db=80)
        db_mel_spec = tf.transpose(
            db_mel_spec, perm=[
                1, 0])  # to keep it (mel, time)
        if tf.shape(db_mel_spec)[
                1] > spec_width:  # check if we have desiered shape
            db_mel_spec = db_mel_spec[:, :spec_width]
        db_mel_spec = tf.reshape(db_mel_spec, spec_shape)
        return db_mel_spec

    def decode(audio):
        spec = get_spectrogram(
            audio,
            spec_shape=dim,
            sr=CFG.sample_rate,
            nfft=CFG.nfft,
            window=CFG.window,
            fmin=CFG.fmin,
            fmax=CFG.fmax)
        spec = spec2img(spec)
        spec = tf.reshape(spec, [*dim, 3])
        return spec

    def decode_with_labels(path, label):
        #         label = get_target(label) # audio decoder does it already
        return decode(path), label

    return decode_with_labels if with_labels else decode


def audio_augmenter(with_labels=True, dim=16000 * 5, CFG=None):
    def augment(audio, dim=dim):
        audio = get_audio_augment(audio, CFG=CFG)
        audio = tf.reshape(audio, [dim])
        return audio

    def augment_with_labels(audio, label):
        return augment(audio), label

    return augment_with_labels if with_labels else augment


def spec_augmenter(with_labels=True, dim=[128, 256], CFG=None):
    def augment(spec, dim=dim):
        spec = get_spec_augment(spec, CFG=CFG)
        spec = tf.reshape(spec, [*dim, 3])
        return spec

    def augment_with_labels(spec, label):
        return augment(spec), label

    return augment_with_labels if with_labels else augment


def build_dataset(paths, labels=None,
                  batch_size=32, cache=True,
                  audio_decode_fn=None, audio_augment_fn=None,
                  spec_decode_fn=None, spec_augment_fn=None,
                  augment=True, repeat=True, shuffle=1024,
                  cache_dir="", drop_remainder=False, CFG=None):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)

    CFG.is_train = labels is not None

    if audio_decode_fn is None:
        audio_decode_fn = audio_decoder(
            labels is not None, dim=CFG.audio_len, CFG=CFG)

    if audio_augment_fn is None:
        audio_augment_fn = audio_augmenter(
            labels is not None, dim=CFG.audio_len, CFG=CFG)

    if spec_decode_fn is None:
        spec_decode_fn = spec_decoder(
            labels is not None, dim=CFG.img_size, CFG=CFG)

    if spec_augment_fn is None:
        spec_augment_fn = spec_augmenter(
            labels is not None, dim=CFG.img_size, CFG=CFG)

    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(audio_decode_fn, num_parallel_calls=AUTO)
    ds = ds.cache(cache_dir) if cache else ds
    ds = ds.repeat() if repeat else ds
    opt = tf.data.Options()
    if shuffle:
        ds = ds.shuffle(shuffle, seed=CFG.seed)
        opt.experimental_deterministic = False
    if CFG.device == 'GPU':
        opt.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    ds = ds.with_options(opt)
    ds = ds.map(audio_augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.map(spec_decode_fn, num_parallel_calls=AUTO)
    ds = ds.map(spec_augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    if CFG.mixup_prob and augment and labels is not None:
        ds = ds.map(
            get_mixup(
                alpha=CFG.mixup_alpha,
                prob=CFG.mixup_prob),
            num_parallel_calls=AUTO)
    if CFG.cutmix_prob and augment and labels is not None:
        ds = ds.map(
            get_cutmix(
                alpha=CFG.cutmix_alpha,
                prob=CFG.cutmix_prob),
            num_parallel_calls=AUTO)
    ds = ds.prefetch(AUTO)
    return ds

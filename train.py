# IMPORT LIBRARIES
import json
import warnings
from loguru import logger
from functools import partial
import time
from utils.submission import path2info
from sklearn.metrics import classification_report
from utils.metrics import MetricFactory, print_dict
from utils.callbacks import get_callbacks
from utils.wandb import wandb_init, wandb_logger, wandb_login
from utils.schedulers import get_lr_scheduler
from utils.viz import plot_spec_batch, plot_confusion_matrix
from models.model import build_model
from dataset.split import create_folds
from dataset.dataset import seeding, build_dataset
from dataset.processing import get_metadata
from utils.device import get_device
from utils.config import dict2cfg, NumpyEncoder
import argparse
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import os
from os.path import join
import pandas as pd
import numpy as np
import wandb
import yaml
from tqdm import tqdm
import seaborn as sns
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

sns.set(style='dark')


logger.info('> IMPORT COMPLETE')


@logger.catch
def train(CFG):
    oof_pred = []
    oof_tar = []
    oof_val = []
    oof_ids = []
    oof_folds = []
    if CFG.wandb:
        # try to login using kaggle-api or browser else anonymous
        CFG.anonymous = wandb_login(CFG.anonymous)
    for fold in range(CFG.folds):
        start = time.time()
        CFG.fold = fold if not CFG.all_data else -1
        if fold not in CFG.selected_folds:
            continue
        if CFG.wandb:
            wandb_init(CFG)

        # TRAIN AND VALID DATAFRAME
        train_df = df.query("fold!=@fold").reset_index(drop=True)
        valid_df = df.query("fold==@fold").reset_index(drop=True)

        # CREATE TRAIN AND VALIDATION SUBSETS
        train_paths = train_df.audio_path.values
        train_labels = train_df[CFG.target_col].values.astype('int32')
        train_labels = to_categorical(
            train_labels, num_classes=CFG.num_classes)

        valid_paths = valid_df.audio_path.values
        valid_labels = valid_df[CFG.target_col].values.astype('int32')
        valid_labels = to_categorical(
            valid_labels, num_classes=CFG.num_classes)

        # ALL DATA
        if CFG.all_data:
            train_paths = np.concatenate([train_paths, valid_paths])
            train_labels = np.concatenate([train_labels, valid_labels])

        # PSEUDO DATA
        if CFG.pseudo:
            pseudo_paths = pseudo_df.audio_path.values
            pseudo_labels = pseudo_df[CFG.class_names].values
            train_paths = np.concatenate([train_paths, pseudo_paths])
            train_labels = np.concatenate([train_labels, pseudo_labels])

        # SHUFFLE IMAGE AND LABELS
        index = np.arange(len(train_paths))
        np.random.shuffle(index)
        train_paths = train_paths[index]
        train_labels = train_labels[index]

        if CFG.debug:
            train_paths = train_paths[:CFG.min_samples]
            train_labels = train_labels[:CFG.min_samples]
            valid_paths = valid_paths[:CFG.min_samples]
            valid_labels = valid_labels[:CFG.min_samples]

        # BATCH SIZE FOR INEFERENCE
        infer_bs = int(CFG.batch_size * CFG.infer_bs_scale)

#         logger.info('#' * 25)
        msg = '\nFOLD: %d | IMAGE_SIZE: (%i, %i) | MODEL_NAME: %s | BATCH_SIZE: %i' % (
            fold, CFG.img_size[0], CFG.img_size[1], CFG.model_name, CFG.batch_size * CFG.replicas)
        msg += ' | OPTIMIZER: %s | SCHEDULER: %s | LOSS: %s | PRETRAIN: %s' % (
            CFG.optimizer, CFG.scheduler, CFG.loss, CFG.pretrain)
        # print('#### UPSASMPLE: %i'%CFG.upsample) if CFG.upsample>1 else None
        num_train = len(train_paths)
        num_valid = len(valid_paths)
        if CFG.wandb:
            wandb.log({'num_train': num_train,
                       'num_valid': num_valid})
        msg += ' | NUM_TRAIN: {:,} | NUM_VALID: {:,}'.format(
            num_train, num_valid)
        msg += ' | ALL_DATA: True' if CFG.all_data else ''
        msg += ' | PSEUDO: True' if CFG.pseudo else ''
        logger.info(msg)

        # BUILD MODEL
        K.clear_session()
        with strategy.scope():
            if CFG.resume is not None:
                try:
                    model = tf.keras.models.load_model(
                        CFG.resume, compile=True)
                    logger.info('[Resume] Loading model from :', CFG.resume)
                except BaseException:
                    model = build_model(
                        CFG,
                        compile_model=True,
                        steps_per_execution=CFG.steps_per_execution)
                    model.load_weights(CFG.resume)
                    logger.info('[Resume] Loading weights from :', CFG.resume)
            else:
                model = build_model(CFG, compile_model=True)

        # DATASET
        drop_remainder = False
        CFG.drop_remainder = drop_remainder
        cache = True if (np.sqrt(np.prod(CFG.img_size)) <=
                         768 and CFG.device == 'TPU') else False
        CFG.cache = cache
        train_ds = build_dataset(
            train_paths,
            train_labels,
            cache=cache,
            batch_size=CFG.batch_size *
            CFG.replicas,
            repeat=True,
            shuffle=True,
            augment=CFG.augment,
            drop_remainder=drop_remainder,
            CFG=CFG)
        val_ds = build_dataset(
            valid_paths,
            valid_labels,
            cache=cache,
            batch_size=CFG.batch_size *
            CFG.replicas,
            repeat=False,
            shuffle=False,
            augment=False,
            drop_remainder=drop_remainder,
            CFG=CFG)

#         print('#' * 25)

        # CALLBACKS
        callbacks = get_callbacks(CFG, monitor=CFG.monitor)

        # TRAIN
        logger.info('> Training:')
        history = model.fit(
            train_ds,
            epochs=CFG.epochs if not CFG.debug else 2,
            callbacks=callbacks,
            initial_epoch=CFG.initial_epoch,
            steps_per_epoch=len(train_paths) / CFG.batch_size / CFG.replicas,
            validation_data=val_ds,
            # validation_steps=len(valid_paths)/CFG.batch_size/CFG.replicas,
            verbose=CFG.verbose
        )
        # Loading best model for inference
        logger.info('Loading best model...')
        if not CFG.all_data:
            model.load_weights(f'{CFG.output_dir}/ckpt/fold-%i.h5' % fold)
        else:
            model.load_weights(f'{CFG.output_dir}/ckpt/model.h5')

        scores = {}
        if not CFG.all_data:
            # PREDICT OOF USING TTA
            logger.info('Predicting OOF with TTA...')
            ds_valid = build_dataset(
                valid_paths,
                labels=None,
                cache=False,
                batch_size=infer_bs *
                CFG.replicas,
                repeat=True,
                shuffle=False,
                augment=CFG.tta > 1,
                drop_remainder=drop_remainder,
                CFG=CFG)
            ct_valid = len(valid_paths)
            STEPS = CFG.tta * ct_valid / infer_bs / CFG.replicas
            pred = model.predict(
                ds_valid,
                steps=STEPS,
                verbose=CFG.verbose)[
                :CFG.tta *
                ct_valid,
            ]

            # GET OOF TARGETS AND idS
            oof_pred.append(
                getattr(
                    np, CFG.agg)(
                    pred.reshape(
                        (ct_valid, -1, CFG.tta), order='F'), axis=-1))
            oof_tar.append(valid_df[CFG.target_col].values[:ct_valid])
            oof_folds.append(np.ones_like(oof_tar[-1], dtype='int8') * fold)
            oof_ids.append(valid_paths)

            # REPORT RESULTS
            y_true = oof_tar[-1].reshape(-1).astype('float32')
            y_pred = oof_pred[-1].argmax(axis=-1)
            metrics = MetricFactory()
            scores = metrics(y_true, oof_pred[-1])
            valid_df.loc[:num_valid - 1, 'pred'] = y_pred
            valid_df.loc[:num_valid - 1, 'miss'] = y_true != y_pred
            valid_df.loc[:num_valid - 1,
                         CFG.class_names] = oof_pred[-1].tolist()
            valid_df.to_csv(
                f'{CFG.output_dir}/csv/oof_{CFG.fold:02d}.csv',
                index=False)

        # one based index for keras
        best_epoch = getattr(np, 'arg' + CFG.monitor_mode)(
            history.history[CFG.monitor], axis=-1) + 1
        best_score = history.history[CFG.monitor][best_epoch - 1]
        scores.update({'score': best_score,
                       'epoch': best_epoch})

        # save scores as JSON file
        with open(f'{CFG.output_dir}/score/score_{CFG.fold:02d}.json', 'w') as f:
            json.dump(scores, f, indent=4, cls=NumpyEncoder)

        if not CFG.all_data:
            oof_val.append(best_score)
            logger.info('\n>>> FOLD %i OOF Score without TTA = %.3f, with TTA = %.3f' % (
                fold, oof_val[-1], scores.get('score')))
            try:
                valid_report = classification_report(
                    y_true, y_pred, target_names=CFG.class_names, output_dict=True)
                valid_report = pd.DataFrame(valid_report).T
                valid_report.to_csv(
                    f'{CFG.output_dir}/csv/report_{CFG.fold:02d}.csv',
                    index=False)
                logger.info(
                    '>>> Class Report:\n' +
                    valid_report.to_markdown(
                        index=True,
                        tablefmt='grid'))
            except BaseException:
                logger.warning("<classification_report> failed")

        if CFG.wandb:
            print('\n\n')
            logger.info('> LOGGING TO W&B:')
            wandb_logger(valid_df,scores,CFG)  # log result to wandb
            wandb.run.finish()  # finish the run
        end = time.time()
        eta = (end - start) / 60
        logger.info(f'>>> TIME: {eta:0.2f} min\n\n')
        if CFG.all_data:
            break

    if not CFG.all_data:
        # COMPUTE OVERALL OOF SCORE
        oof = np.concatenate(oof_pred)
        true = np.concatenate(oof_tar)
        ids = np.concatenate(oof_ids)
        folds = np.concatenate(oof_folds)
        scores = metrics(true.astype('float32').reshape(-1), oof)
        logger.info('> Overall OOF Scores ' + print_dict(scores))

        # SAVE OOF TO DISK
        logger.info('> PROCESSING OOF:')
        columns = ['audio_path', 'fold', 'true', 'pred', *CFG.class_names]
        df_oof = pd.DataFrame(np.concatenate([ids[:, None], folds, true, np.argmax(
            oof, axis=1)[:, None], oof], axis=1), columns=columns)
        df_oof['class_name'] = df_oof.true.map(CFG.label2name)
        df_oof['miss'] = df_oof.true != df_oof.pred
        tqdm.pandas(desc='path2info ')
        df_oof = df_oof.progress_apply(path2info, axis=1)
        df_oof.drop(columns=['audio_path'], inplace=True)
        df_oof.to_csv(f'{CFG.output_dir}/csv/oof.csv', index=False)

        # Missed Cases
        logger.info('> Miss Distribution:')
        logger.info(df_oof.query("miss==True").class_name.value_counts())
        logger.info('> Miss Total: %d' % df_oof.query("miss==True").shape[0])

        # Confusion Matrix
        logger.info('> OOF Confusion Matrix Saved')
        plot_confusion_matrix(
            df_oof.true,
            df_oof.pred,
            classes=CFG.class_names,
            save=True,
            output_dir=CFG.output_dir)

        try:
            # Classification Report
            oof_report = classification_report(
                df_oof.true,
                df_oof.pred,
                target_names=CFG.class_names,
                output_dict=True)
            oof_report = pd.DataFrame(oof_report).T
            oof_report.to_csv(
                f'{CFG.output_dir}/csv/oof_report.csv',
                index=False)
            logger.info(
                '> OOF Class Metric:\n' +
                oof_report.to_markdown(
                    index=True,
                    tablefmt='grid'))
        except BaseException:
            logger.warning("<classification_report> failed")
    return


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg',
        type=str,
        default='./configs/sp2022.yaml',
        help='config file')
    parser.add_argument(
        '--paths-json',
        type=str,
        default="./PATHS.json",
        help='paths json file for data')
    parser.add_argument(
        '--debug',
        type=int,
        default=0,
        help='process only small portion in debug mode')
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='verbosity')
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='name of the model')
    parser.add_argument(
        '--img-size',
        type=int,
        nargs='+',
        default=None,
        help='image size: H x W')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='batch_size for the model')
    parser.add_argument('--loss', type=str, default=None,
                        help='name of the loss function')
    parser.add_argument(
        '--scheduler',
        type=str,
        default=None,
        help='lr scheduler')
    parser.add_argument(
        '--all-data',
        type=int,
        default=None,
        help='use all data for training no-val')
    parser.add_argument(
        '--selected-folds',
        type=int,
        nargs='+',
        default=None,
        help='folds to train')
    parser.add_argument(
        '--wandb',
        type=int,
        default=0,
        help='wandb On or OFF')
    parser.add_argument(
        '--anonymous',
        type=str,
        default='never',
        help='anonymous mode of wandb')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='output path to save the model')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='resume location')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start from which epoch')
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='total epochs for train')
    parser.add_argument(
        '--pseudo', 
        type=int, 
        default=0)
    parser.add_argument(
        '--pseudo_csv',
        type=str,
        default='data/pseudo/pred.csv',
        help='path for pseudo csv')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help='pretrain name')
    opt = parser.parse_args()
    return opt


def update_cfg(CFG, opt):
    #     logger.info('> UPDATING CONFIG')
    # OVERWRITE
    if opt.debug is not None:
        CFG.debug = opt.debug
    logger.info('> DEBUG MODE: %s' % bool(CFG.debug))
    if opt.verbose is not None:
        CFG.verbose = opt.verbose
    if opt.model_name:
        CFG.model_name = opt.model_name
    if opt.pretrain:
        CFG.pretrain = opt.pretrain
    if opt.img_size:
        CFG.img_size = opt.img_size
    if opt.epochs:
        CFG.epochs = opt.epochs
    if opt.batch_size:
        CFG.batch_size = opt.batch_size
    if opt.loss:
        CFG.loss = opt.loss
    if opt.scheduler:
        CFG.scheduler = opt.scheduler
    if opt.wandb is not None:
        CFG.wandb = bool(opt.wandb)
    if opt.anonymous:
        CFG.anonymous = opt.anonymous
    if opt.output_dir:
        output_dir = os.path.join(opt.output_dir,
                                  '{}-{}x{}'.format(CFG.model_name,
                                                    CFG.img_size[0],
                                                    CFG.img_size[1]))
    else:
        output_dir = os.path.join(
            'output', '{}-{}x{}'.format(CFG.model_name, CFG.img_size[0], CFG.img_size[1]))
    # Create directories for saving outputs
    os.makedirs(f'{output_dir}', exist_ok=True)
    os.makedirs(f'{output_dir}/image', exist_ok=True)
    os.makedirs(f'{output_dir}/ckpt', exist_ok=True)
    os.makedirs(f'{output_dir}/csv', exist_ok=True)
    os.makedirs(f'{output_dir}/cfg', exist_ok=True)
    os.makedirs(f'{output_dir}/score', exist_ok=True)
    CFG.output_dir = output_dir
    if opt.selected_folds:
        CFG.selected_folds = opt.selected_folds
    if opt.all_data:
        CFG.all_data = opt.all_data
    if CFG.all_data:
        CFG.selected_folds = [0]
    CFG.num_classes = len(CFG.class_names)
    CFG.resume = opt.resume
    CFG.initial_epoch = opt.start_epoch
    CFG.pseudo = opt.pseudo
    return CFG


if __name__ == '__main__':
    # PARSE OPTIONS
    opt = parse_opt()

    # LOADING CONFIG
    CFG_PATH = opt.cfg
    cfg_dict = yaml.load(open(CFG_PATH, 'r'), Loader=yaml.FullLoader)
    CFG = dict2cfg(cfg_dict)  # dict to class
    # print('config:', cfg)

    # UPDATE CONFIG WITH OPTIONS
    CFG = update_cfg(CFG, opt)

    # CONFIGURE DEVICE
    strategy, device = get_device()
    CFG.device = device
    AUTO = tf.data.experimental.AUTOTUNE
    CFG.replicas = strategy.num_replicas_in_sync
#     logger.info(f'> REPLICAS: {CFG.replicas}')

    # MINIMUM SAMPLES FOR DEBUG
    CFG.min_samples = CFG.batch_size * CFG.replicas * 2

    # SEEDING
    seeding(CFG)

    # METADATA
    df = get_metadata(CFG, opt)

    # PSEUDO DATA
    if CFG.pseudo:
        pseudo_df = pd.read_csv(opt.pseudo_csv)
        print('> PSEUDO SAMPLES: {:,}'.format(len(pseudo_df)))

    # CHECK FILE FROM GCS_PATH
    if not os.path.isfile(df.audio_path.iloc[0]):
        logger.error('audio_path not exist')

    # DATA SPLIT
    df = create_folds(df, CFG=CFG)

    # PLOT SOME DATA
    fold = 0
    fold_df = df.query('fold==@fold')[100:200]
    paths = fold_df.audio_path.tolist()
    labels = fold_df[CFG.target_col].values
    labels = to_categorical(labels, num_classes=CFG.num_classes)
    ds = build_dataset(
        paths,
        labels,
        cache=False,
        batch_size=CFG.batch_size *
        CFG.replicas,
        repeat=True,
        shuffle=True,
        augment=True,
        CFG=CFG)
    ds = ds.unbatch().batch(20)
    batch = next(iter(ds))
    plot_spec_batch(batch, n_row=2, n_col=3,
                    sample_rate=CFG.sample_rate, hop_length=CFG.hop_length,
                    fmin=CFG.fmin, fmax=CFG.fmax, output_dir=CFG.output_dir)

    # PLOT LR SCHEDULE
    get_lr_scheduler(CFG.batch_size * CFG.replicas, CFG=CFG, plot=True)

    # Training
    # logger.info('> TRAINING:')
    train(CFG)

    # REMOVE WANDB FILE
    if CFG.wandb:
        os.system('rm -r wandb')

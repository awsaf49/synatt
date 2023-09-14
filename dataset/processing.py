import pandas as pd
import numpy as np
import json
from os.path import join
import tensorflow as tf

def get_test_data(opt):
    # DATA DIRECTORY
    PATH_JSON = json.load(open(opt.paths_json, 'r'))
    TEST1_DIR = PATH_JSON['TEST1_DATA_DIR']
    TEST2_DIR = PATH_JSON['TEST2_DATA_DIR']
    # Test Data Part1
    test_df1 = pd.read_csv(join(TEST1_DIR, 'labels_eval_part1.csv'))
    test_df1['audio_path'] = test_df1.track.map(lambda x: join(TEST1_DIR, x))
    test_df1['part'] = 1
    print('> TEST1 SAMPLES: {:,}'.format(len(test_df1)))

    # Test Data Part2
    test_df2 = pd.read_csv(join(TEST2_DIR, 'labels_eval_part2.csv'))
    test_df2['audio_path'] = test_df2.track.map(lambda x: join(TEST2_DIR, x))
    test_df2['part'] = 2
    print('> TEST2 SAMPLES: {:,}'.format(len(test_df2)))

    # Test Data Merge
    test_df = pd.concat([test_df1, test_df2], axis=0)
    try:
        # index column exist for test data
        test_df.set_index("Unnamed: 0", inplace=True)
    except BaseException:
        pass
    test_df = test_df.reset_index(drop=True)  # to keep index monotonic
    test_df['filename'] = test_df.track
    test_df['source'] = 'sp'
    print('> TEST SAMPLES: {:,}'.format(len(test_df)))
    return test_df

def get_metadata(CFG, opt):
    # DATA DIRECTORY
    PATH_JSON = json.load(open(opt.paths_json, 'r'))
    TRAIN_DIR = PATH_JSON['TRAIN_DATA_DIR']
    UNSEEN_DIR = PATH_JSON['UNSEEN_DATA_DIR']
    LJ_DIR = PATH_JSON['LJ_DATA_DIR']
    VCTK_DIR = PATH_JSON['VCTK_DATA_DIR']
    LIBRI_DIR = PATH_JSON['LIBRI_DATA_DIR']
    SYNTHETIC_DIR = PATH_JSON['SYNTHETIC_DATA_DIR']

    # META DATA
    # Train Data
    seen_df = pd.read_csv(join(TRAIN_DIR, 'labels.csv'))
    seen_df['audio_path'] = seen_df.track.map(lambda x: join(TRAIN_DIR, x))

    unseen_df = pd.read_csv(join(UNSEEN_DIR, 'labels.csv'))
    unseen_df['audio_path'] = unseen_df.track.map(
        lambda x: join(UNSEEN_DIR, x))

    sp_df = pd.concat([seen_df, unseen_df], axis=0)
    sp_df['filename'] = sp_df.track
    sp_df['label'] = sp_df.algorithm
    sp_df.loc[sp_df.label == 0, 'speaker_id'] = 's0'
    sp_df.loc[(sp_df.label != 0) & (sp_df.label != 4), 'speaker_id'] = 'lj'
    sp_df.loc[sp_df.label == 4, 'speaker_id'] = 'multi'
    sp_df['source'] = 'sp'
    sp_df.drop(columns=['track', 'algorithm'], inplace=True)
    print('> SPCUP SAMPLES: {:,}'.format(len(sp_df)))

    # NATURAL DATA
    nat_dfs = []

    if CFG.lj_data:
        # LJ Speech Data
        lj_df = pd.read_csv(join(LJ_DIR, 'metadata.csv'))
        try:
            # index column exist for ljspeech data
            lj_df.set_index("Unnamed: 0", inplace=True)
        except BaseException:
            pass
        lj_df.drop(columns=['id', 'sentence'], inplace=True)
        lj_df = lj_df.rename({'file_name': 'filename'}, axis=1)
        lj_df['audio_path'] = join(LJ_DIR, 'wavs/') + lj_df.filename
        lj_df['source'] = 'ljsp'
        lj_df['speaker_id'] = 'lj'
        lj_df = lj_df.sample(
            frac=CFG.lj_frac,  # 0.30
            random_state=CFG.seed).reset_index(
            drop=True)
        print('> LJ SAMPLES: {:,}'.format(len(lj_df)))
        nat_dfs.append(lj_df)

    if CFG.vctk_data:
        # VCTK Data
        vc_df = pd.read_csv(join(VCTK_DIR, 'metadata.csv'))
        try:
            # index column exist for vctk data
            vc_df.set_index("Unnamed: 0", inplace=True)
        except BaseException:
            pass
        vc_df.drop(columns=['sentence'], inplace=True)
        vc_df = vc_df.rename({'file_name': 'filename'}, axis=1)
        vc_df = vc_df[vc_df.filename.str.contains('wav')]
        vc_df['audio_path'] = join(VCTK_DIR, 'wavs/') + vc_df.filename
        vc_df['source'] = 'vctk'
        vc_df['speaker_id'] = 'p' + vc_df['speaker_id'].astype(str)
        vc_df = vc_df.groupby('speaker_id').sample(
            frac=CFG.vctk_frac,  # 0.10
            random_state=CFG.seed).reset_index(
            drop=True)
        print('> VCTK SAMPLES: {:,}'.format(len(vc_df)))
        nat_dfs.append(vc_df)

    if CFG.libri_data:
        # LIBRISPEECH Data
        libri_df = pd.read_csv(join(LIBRI_DIR, 'metadata.csv'))
        libri_df.drop(columns=['file_id', 'sex'], inplace=True)
        libri_df['audio_path'] = join(LIBRI_DIR, 'wavs/') + libri_df.filename
        libri_df['source'] = 'libri'
        libri_df['speaker_id'] = 'lb' + libri_df['speaker_id'].astype(str)
        libri_df = libri_df.groupby('speaker_id').sample(
            frac=CFG.libri_frac,  # 0.10
            random_state=CFG.seed).reset_index(
            drop=True)
        print('> LIBRI SAMPLES: {:,}'.format(len(libri_df)))
        nat_dfs.append(libri_df)

    if CFG.natural_speech or CFG.lj_data or CFG.vctk_data or CFG.libri_data:
        # Merge Natural Data
        nat_df = pd.concat(nat_dfs, axis=0)
        print('> NATURAL SAMPLES: {:,}'.format(len(nat_df)))

    if CFG.synthetic_speech:
        # SYNTHESSIC DATA
        syn_df = pd.read_csv(join(SYNTHETIC_DIR, 'metadata.csv'))
        syn_df['audio_path'] = syn_df.audio_path.map(
            lambda x: join(SYNTHETIC_DIR, 'wavs/') + x.split('/', 6)[-1])
        syn_df['filename'] = syn_df.audio_path.map(lambda x: x.split('/')[-1])
        syn_df['source'] = 'syn'
        syn_df['speaker_id'] = syn_df.speaker_id.astype(str)
        tmp0 = syn_df[syn_df.algorithm != 3].groupby(['exp_name', 'algorithm']).sample(
            frac=CFG.syn_frac, random_state=CFG.seed).reset_index(drop=True)
        # class 3 has less sample so avoid sub-sample
        tmp1 = syn_df[syn_df.algorithm == 3].copy()
        syn_df = pd.concat([tmp0, tmp1], axis=0)
        print('> SYNTHETIC SAMPLES: {:,}'.format(len(syn_df)))

    # Merge Data
    dfs = []
    if CFG.num_classes < 6:
        dfs.append(sp_df.query("label!=5"))  # exclude unknown data
    else:
        dfs.append(sp_df)

    if CFG.num_classes == 6:
        if CFG.natural_speech:
            nat_df.loc[:, 'label'] = 5  # set label=5 for natural data
            dfs.append(nat_df)
        if CFG.synthetic_speech:
            syn_df.loc[:, 'label'] = 5
            dfs.append(syn_df)

    df = pd.concat(dfs, axis=0)
    df['label'] = df['label'].astype(int)
    print('> TOTAL SAMPLES: {:,}'.format(len(df)))

    return df


def get_pseudo_data(test_df, opt):
    pseudo_df = pd.read_csv(opt.pseudo_csv)
    pseudo_df = pseudo_df.merge(test_df[['filename', 'audio_path', 'part']], on=[
                                'filename'], how='left')
    print('> PSEUDO SAMPLES: {:,}'.format(len(pseudo_df)))
    return pseudo_df

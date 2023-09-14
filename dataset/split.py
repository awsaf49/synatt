import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold

# DATA SPLIT


def create_folds(df, CFG=None):
    sgkf = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=CFG.seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
    gkf = GroupKFold(n_splits=5)

    ndf = df.query("source=='vctk' | source=='libri'")
    adf = df.query("source!='vctk' & source!='libri' & source!='syn'")
    sdf = df.query("source=='syn'")

    adf = adf.reset_index(drop=True)
    adf["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(adf, adf['label'])):
        adf.loc[val_idx, 'fold'] = fold

    if (CFG.natural_speech or CFG.num_classes == 7) and (
            CFG.vctk_data or CFG.libri_data):
        ndf = ndf.reset_index(drop=True)
        ndf["fold"] = -1
        for fold, (train_idx, val_idx) in enumerate(
                gkf.split(ndf, groups=ndf['speaker_id'])):
            ndf.loc[val_idx, 'fold'] = fold

    if CFG.synthetic_speech:
        sdf = sdf.reset_index(drop=True,)
        sdf["fold"] = -1
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(
                sdf, y=sdf['algorithm'].tolist(), groups=sdf['exp_name'].tolist())):
            sdf.loc[val_idx, 'fold'] = fold

    df = pd.concat([adf, ndf, sdf], axis=0)
    return df

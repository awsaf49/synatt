import wandb
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from dataset.dataset import build_dataset
from utils.config import cfg2dict

# LOGIN TO WANDB


def wandb_login(anonymous='never'):
    if anonymous == 'never':
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            api_key = user_secrets.get_secret("WANDB")
            wandb.login(key=api_key, relogin=True)
            anonymous = None
        except BaseException:
            api_key = None
            wandb.login(key=api_key, relogin=True)
            anonymous = None
    else:
        anonymous = "must"
        wandb.login(anonymous=anonymous, relogin=True)
    return anonymous

# INITIALIZE WANDB


def wandb_init(CFG):
    fold = CFG.fold
    config = cfg2dict(CFG)
    # config.update({"fold":int(fold)}) # int is to convert numpy.int -> int
    yaml.dump(config, open(f'{CFG.output_dir}/cfg/fold-{fold}.yaml', 'w'),)
    config = yaml.load(
        open(
            f'{CFG.output_dir}/cfg/fold-{fold}.yaml',
            'r'),
        Loader=yaml.FullLoader)
    run = wandb.init(
        project="sp2022",
        name=f"fold-{fold}|dim-{CFG.img_size[1]}x{CFG.img_size[0]}|model-{CFG.model_name}",
        config=config,
        anonymous=CFG.anonymous,
        group=CFG.comment,
        save_code=True,
        entity="rtx4090-2-0" if CFG.anonymous != 'must' else None)
    return run

# LOG SCORE TO WANDB


def wandb_logger(
        valid_df,
        scores,
        CFG):
    "log best result and grad-cam for error analysis"
    if CFG.all_data:
        # scores = None
        valid_table = None
    else:
        valid_df = valid_df.copy()
        if CFG.debug:
            valid_df = valid_df.iloc[:CFG.min_samples]
        noimg_cols = [*CFG.tab_cols, 'label', 'pred', 'miss', *CFG.class_names]
        valid_data = valid_df.loc[:, noimg_cols].values.tolist()
        valid_table = wandb.Table(data=valid_data, columns=[*noimg_cols])
    wandb.log({'best': scores,
               'valid_table': valid_table,
               })

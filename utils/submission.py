# GLOBAL VARIABLES
import numpy as np
import pandas as pd
import os

# SUBMISSION HELPER


def path2info(row):
    filename = row['audio_path'].split(os.sep)[-1]
    row['filename'] = filename
    return row


def df_to_output(df, output_path, verbose=1):
    sub_df = df[['filename', 'pred']].copy()
    sub_df = sub_df.rename({'filename': 'track', 'pred': 'algorithm'}, axis=1)

    save_path = os.path.join(output_path, 'answer.txt')
    zip_path = os.path.join(output_path, 'answer.zip')
    # sub_df.to_csv('submission.csv',index=False)
    sub_df.to_csv(save_path, sep=',', index=False, header=False)
    os.system(f'zip {zip_path} {save_path}')

    if verbose:
        print(f'Output zip file created at {zip_path}')

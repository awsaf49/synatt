import argparse
import yaml, json
import shutil, time, os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from glob import glob
from utils.config import dict2cfg
from dataset.dataset import build_dataset,seeding
from utils.device import get_device
#from utils.ensemble_v2 import MeanEnsemble
from models.model import build_model
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from tqdm import tqdm
from utils.submission import path2info, df_to_output
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)


def predict_soln(CFG,ensemble=False):
    if CFG.verbose == 1:
        print('='*35)
        print('### Inference')
        print('='*35)
    # PREDICTION FOR ALL MODELS
    
    if CFG.output_dir.endswith('.csv'):
        final_save_loc = CFG.output_dir
        zip_save_loc = os.path.dirname(CFG.output_dir)
    else:
        final_save_loc = os.path.join(CFG.output_dir,'ensemble_submission.csv')
        zip_save_loc = CFG.output_dir
    
    id_keys={}
    class_names = CFG.class_names
    for model_idx, (model_paths, dim,idx) in enumerate(CFG.ckpt_cfg):
        preds=[]
        start = time.time()
        
        source_path = os.path.dirname(os.path.dirname(model_paths[0]))
        model_name = os.path.basename(source_path)
        pred_save_path = CFG.temp_save_dir
        #pred_save_path = os.path.join(source_path, CFG.temp_save_dir)
        #ensemble_search_dir = os.path.dirname(source_path)
        if CFG.verbose:
            print(f'> MODEL({model_idx+1}/{len(CFG.ckpt_cfg)}): {model_name} | DIM: {dim}')


        # META DATA
        test_paths = tf.io.gfile.glob(os.path.join(CFG.infer_path, '*.wav'))
        test_names = [os.path.basename(path) for path in test_paths]

        # DEBUG MODE
        if CFG.debug:
            test_paths = test_paths[:100]
            
        # CONFIGURE BATCHSIZE
        mx_dim = np.sqrt(np.prod(dim))
        if mx_dim>=768 or any(i in model_name for i in ['convnext','ECA_NFNetL2']):
            CFG.batch_size = CFG.replicas * 16
        elif mx_dim>=640  or any(i in model_name for i in ['EfficientNet','RegNet','ResNetRS50','ResNest50']):
            CFG.batch_size = CFG.replicas * 32
        else:
            CFG.batch_size = CFG.replicas * 64
        
        CFG.img_size = dim
        drop_remainder = 'swin' in model_name
        CFG.drop_remainder = drop_remainder
        # BUILD DATASET
        dtest = build_dataset(
                    test_paths,
                    labels=None,
                    augment=CFG.tta > 1,
                    repeat=True,
                    cache=False,
                    shuffle=False,
                    batch_size=CFG.batch_size,
                    drop_remainder=CFG.drop_remainder,
                    CFG=CFG)


        # PREDICTION FOR ONE MODEL -> N FOLDS
        j = 0
        for model_path in sorted(model_paths):

            with strategy.scope():
                model = tf.keras.models.load_model(model_path, compile=False)
            
            if j == 0:
                pred = model.predict(dtest, steps = max(CFG.tta*len(test_paths)/CFG.batch_size,1), verbose=1)

                pred = pred[:CFG.tta*len(test_paths),:]
            j+=1
            ## MULTI CLASS PREDS
            preds.append(getattr(np, CFG.agg)(pred.reshape((len(test_paths),CFG.tta, -1),order='F'),axis=1)) 
            # Multiclass prediction

        end = time.time()
        eta = (end-start)/60
        #print(f'>>> TIME FOR {model_name}: {eta:0.2f} min')
        if CFG.verbose:
            print('> PROCESSING SUBMISSION')


         # PROCESSS PREDICTION
        preds = getattr(np, CFG.agg)(preds, axis=0)
        test_paths = np.array(test_paths)   
        columns = ['audio_path', 'pred', *CFG.class_names]
        pred_df = pd.DataFrame(np.concatenate([test_paths[:,None],
                                            np.argmax(preds,axis=1)[:,None],
                                            preds], axis=1), columns=columns)

        pred_df['class_name'] = pred_df.pred.astype(int).map(CFG.label2name)
        pred_df['filename'] = pred_df.audio_path.map(lambda x: x.split('/')[-1])
        pred_df[CFG.class_names] = pred_df[CFG.class_names].astype('float32')
        tqdm.pandas(desc='id ')
        pred_df = pred_df.progress_apply(path2info,axis=1)
        #pred_path = os.path.abspath(f'{INF_PATH}/submission_{model_name}_{dim[0]}x{dim[1]}.csv')
        #pred_path = os.path.join(pred_save_path, 'pred_sub.csv')  #--------------------------------------------------
        pred_path = os.path.join(pred_save_path, model_name+'_pred.csv') 
        try:
            pred_df.to_csv(pred_path,index=False)
        except:
            #print('Save Failed at ',pred_path)
            pred_path = os.path.join(zip_save_loc,model_name+'_pred.csv')
            pred_df.to_csv(pred_path,index=False)
        
        if CFG.verbose == 1:
            print(F'\n> SUBMISSION SAVED TO: {pred_path}')
            print(pred_df.head(2))
        id_keys[idx]=pred_path #----------------------------------------------------------------------------
        
        print('\n\n')
    
    #print(id_keys)
    all_sub_paths = [id_keys[x] for x in sorted(id_keys.keys())]
    #print(all_sub_paths)
        
#     print('zip save loc: ',zip_save_loc)
#     print('final save loc:', final_save_loc)
    
    if ensemble:
        
        dfs = pd.concat([pd.read_csv(i) for i in all_sub_paths])
        pred_df = dfs.groupby('filename')[class_names].mean().reset_index()
        pred_df['pred'] = pred_df[class_names].values.argmax(axis=-1)
        if not CFG.pseudo:
            pred_df = pred_df[['filename', 'pred']]

        pred_df.to_csv(final_save_loc, index = False)

        
        if CFG.verbose:
            print('Final Prediction saved at ',final_save_loc)
    
    if not CFG.no_zip:
        df_to_output(pred_df, zip_save_loc, verbose = 1)
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/sp22-part2.yaml', help='config file')
    #arser.add_argument('--ckpt-cfg', type=str, default='configs/checkpoints.json', help='config file for checkpoint')
    parser.add_argument('--model-dir', type=str, default='output', help='where checkpoint weights can be found')
    parser.add_argument('--infer-path',type=str,default=None,help='path to the infer data')
    parser.add_argument('--debug', type=int, default=0, help='process only small portion in debug mode')
    parser.add_argument('--verbose', type=int, default=2, help='print progress details')
    parser.add_argument('--output', type=str, default='submission', help='output path to save the submission')
    parser.add_argument('--temp-save-dir', type=str, default='temp', help='where to save the temporary predictions')
    parser.add_argument('--no-zip', action='store_true', help='if zip file is not to be generated')
    parser.add_argument('--pseudo', action = 'store_true', help='preserve soft labels or not')
    parser.add_argument('--tta', type=int, default=1, help='number of TTA')
    opt = parser.parse_args()
    
    # LOADING CONFIG
    CFG_PATH = opt.cfg
    cfg_dict = yaml.load(open(CFG_PATH, 'r'), Loader=yaml.FullLoader)
    CFG      = dict2cfg(cfg_dict) # dict to class
    CFG.pseudo = opt.pseudo
 
    
    # LOADING CKPT CFG
    #CKPT_CFG_PATH = opt.ckpt_cfg
    CFG.no_zip = opt.no_zip
    CFG.verbose = opt.verbose
    CFG.model_dir = opt.model_dir
    #CFG.submission_type = opt.submission_type
    CFG.temp_save_dir = opt.temp_save_dir
    os.system(f'mkdir -p {CFG.temp_save_dir}')
    
    CKPT_CFG = []
    CKPT_CFG_PATH = []
    
    # SCAN FOR MODEL PATH AND IMG SIZE
    j = 0
    for model_path in os.listdir(CFG.model_dir):
        img_size = [int(i) for i in model_path.split('-')[-1].split('x')]
        #models = glob(os.path.join(MODEL_DIR,model_path,'ckpt','*.h5'))
        CKPT_CFG_PATH.append([model_path,img_size,j])
        j+=1
        
    if CFG.verbose == 1:
        _ = [print(i) for i in CKPT_CFG_PATH]
    
    # SCAN FOR MODELS IN MODEL PATHS
    for base_dir,dim,idx in CKPT_CFG_PATH:
        if '.h5' not in base_dir:
            paths = sorted(glob(os.path.join(opt.model_dir,base_dir,'ckpt','*h5')))
            #print(os.path.join(opt.model_dir,base_dir,'ckpt','*h5'))
        else:
            paths = [os.path.join(opt.model_dir,base_dir)]
        if len(paths)==0:
            logger.warning('No model found in ',base_dir,'. Skipping...')
        else:
            #raise ValueError('no model found for :',base_dir)
            #print(paths)
            CKPT_CFG.append([paths, dim,idx])
        
    #print(CKPT_CFG)
    CFG.ckpt_cfg = CKPT_CFG
    CFG.infer_path = opt.infer_path
    
    # OVERWRITE
    if opt.debug is not None:
        CFG.debug = opt.debug
        
    if CFG.verbose:
        print('> DEBUG MODE:', bool(CFG.debug))

    if opt.tta is not None:
        CFG.tta = opt.tta
        
        
    # CREATE SUBMISSION DIRECTORY
    CFG.output_dir = opt.output
    if CFG.output_dir.endswith('.csv'):
        os.system(f'mkdir -p {os.path.dirname(CFG.output_dir)}')
    else:
        os.system(f'mkdir -p {CFG.output_dir}')
        
        
    # CONFIGURE DEVICE
    strategy, device = get_device()
    CFG.device   = device
    AUTO         = tf.data.experimental.AUTOTUNE
    CFG.replicas = strategy.num_replicas_in_sync
    print(f'> REPLICAS: {CFG.replicas}')   
    
    # SEEDING
    seeding(CFG)
    
    # Prediction
    predict_soln(CFG,ensemble=True)
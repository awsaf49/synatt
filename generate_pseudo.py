import argparse, os
import pandas as pd

def merge_pseudo(part1_df, part2_df, save_location, verbose = 1):
    part1 = pd.read_csv(part1_df)
    part2 = pd.read_csv(part2_df)
    
    part1['part'] = 1
    part2['part'] = 2
    
    combine = pd.concat([part1,part2])
    combine.to_csv(save_location, index = False)
    if verbose:
        print('Pseudo saved to',save_location)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part1-infer-path', type=str, default='./output/supervised/part1', help='where part1 infer files are located')
    parser.add_argument('--part2-infer-path', type=str, default='./output/supervised/part2', help='where part2 infer files are located')
    parser.add_argument('--part1-model-dir', type=str, default='./output/supervised/part1', help='where part1 model ckpts are located')
    parser.add_argument('--part2-model-dir', type=str, default='./output/supervised/part1', help='where part2 model ckpts are located')
    parser.add_argument('--debug', type=int, default=0, help='process only small portion in debug mode')
    parser.add_argument('--verbose', type=int, default=2, help='print progress details')
    parser.add_argument('--output', type=str, default='submission', help='output path to save the pseudo csv')
    parser.add_argument('--temp-save-dir', type=str, default='temp', help='where to save the temporary predictions')
    opt = parser.parse_args()
    
    
    DEBUG = opt.debug
    VERBOSE = opt.verbose
    PART1_INFER_PATH = opt.part1_infer_path
    PART2_INFER_PATH = opt.part2_infer_path
    
    TEMP_SAVE_DIR = opt.temp_save_dir
    os.system(f'mkdir -p {TEMP_SAVE_DIR}')
    PART1_SAVE_PATH = os.path.join(TEMP_SAVE_DIR,'part1_pseudo.csv')
    PART2_SAVE_PATH = os.path.join(TEMP_SAVE_DIR,'part2_pseudo.csv')
    
    PART1_MODEL_DIR = opt.part1_model_dir
    PART2_MODEL_DIR = opt.part2_model_dir
    
    OUTPUT_SAVE_PATH = opt.output
    if OUTPUT_SAVE_PATH.endswith('.csv'):
        os.system(f'mkdir -p {os.path.dirname(OUTPUT_SAVE_PATH)}')
    else:
        os.system(f'mkdir -p {OUTPUT_SAVE_PATH}')
        OUTPUT_SAVE_PATH = os.path.join(OUTPUT_SAVE_PATH,'pseudo.csv')
    
    part1_cmd = f"python3 predict.py --model-dir {PART1_MODEL_DIR} --infer-path {PART1_INFER_PATH} --cfg ./configs/sp22-part1.yaml "
    part1_cmd = part1_cmd + f" --output {PART1_SAVE_PATH} --debug {DEBUG} --verbose {VERBOSE} --no-zip --pseudo " #--temp-save-dir {SUBMISSION_TYPE}"
    
    if VERBOSE == 1:
        print('-'*35)
        print(' Part1 Pseudo')
        print('-'*35)
    os.system(part1_cmd)
    
    part2_cmd = f"python3 predict.py --model-dir {PART2_MODEL_DIR} --infer-path {PART2_INFER_PATH} --cfg ./configs/sp22-part2.yaml "
    part2_cmd = part2_cmd + f" --output {PART2_SAVE_PATH} --debug {DEBUG} --verbose {VERBOSE} --no-zip --pseudo " # --temp-save-dir {SUBMISSION_TYPE}"
    if VERBOSE == 1:
        print('-'*35)
        print(' Part2 Pseudo')
        print('-'*35)
    os.system(part2_cmd)


    part1 = pd.read_csv(PART1_SAVE_PATH)
    part2 = pd.read_csv(PART2_SAVE_PATH)

    part1['audio_path'] = PART1_INFER_PATH + os.sep + part1['filename']
    part2['audio_path'] = PART2_INFER_PATH + os.sep + part2['filename']
    
    part1['part'] = 1
    part2['part'] = 2
    
    combine = pd.concat([part1,part2])
    combine.to_csv(OUTPUT_SAVE_PATH, index = False)
    if VERBOSE:
        print('Pseudo saved to',OUTPUT_SAVE_PATH)

    
    #merge_pseudo(PART1_SAVE_PATH, PART2_SAVE_PATH, OUTPUT_SAVE_PATH, VERBOSE)
    
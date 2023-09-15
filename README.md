

# Syn-Att: Synthetic Speech Attribution via Semi-Supervised Unknown Multi-Class Ensemble of CNNs

<img src="https://user-images.githubusercontent.com/36858976/153433391-0c47d037-33c9-4942-aec7-3532b97378d1.jpg" width=600>

**Paper**:

**Abstract**: With the huge technological advances introduced by deep learning in audio & speech processing, many novel synthetic speech techniques achieved incredible realistic results. As these methods generate realistic fake human voices, they can be used in malicious acts such as people imitation, fake news, spreading, spoofing, media manipulations, etc. Hence, the ability to detect synthetic or natural speech has become an urgent necessity. Moreover, being able to tell which algorithm has been used to generate a synthetic speech track can be of preeminent importance to track down the culprit. In this paper, a novel strategy is proposed to attribute a synthetic speech track to the generator that is used to synthesize it. The proposed detector transforms the audio into log-mel spectrogram, extracts features using CNN, and classifies it between five known and unknown algorithms, utilizing semi-supervision and ensemble to improve its robustness and generalizability significantly.  The proposed detector is validated on two evaluation datasets consisting of a total of 18,000 weakly perturbed (Eval 1) & 10,000 strongly perturbed (Eval 2) synthetic speeches. The proposed method outperforms other top teams in accuracy by 12-13% on Eval 2 and 1-2% on Eval 1, in the IEEE SP Cup challenge at ICASSP 2022.

-------------------------------------------------------------------------------------------

# Result on IEEE Sp Cup 2023
Score of the top 3 teams on the leaderboard of [IEEE Signal Processing CUP 2022](https://signalprocessingsociety.org/community-involvement/ieee-signal-processing-cup-2022).


**Strongly Perturbed**:
| Method / Metric        | Acc       | Prc       | Rec       | F1        |
| :--------------------- | :-------: | :-------: | :-------: | :-------: |
| Std. Proc.             | 0\.48     | 0\.62     | 0\.48     | 0\.48     |
| Team IITH              | 0\.49     | 0\.51     | 0\.49     | 0\.49     |
| **Synthesizer (Ours)** | **0\.61** | **0\.71** | **0\.61** | **0\.63** |


**Weakly Perturbed**:
| Method / Metric        | Acc       | Prc       | Rec       | F1        |
| :--------------------- | :-------: | :-------: | :-------: | :-------: |
| Std. Proc.             | 0\.97     | 0\.97     | 0\.96     | 0\.97     |
| Team IITH              | 0\.96     | 0\.96     | 0\.95     | 0\.96     |
| **Synthesizer (Ours)** | **0\.98** | **0\.99** | **0\.97** | **0\.98** |


# How to Run?

<details>
<summary>Important Notes</summary>
<br>

* All audio files must be in `.wav` format.
* Sample Rate must be `16,000`.
* For training, `batch_size` is tuned for `8 x V100`. If models is trained in other device, `batch_size` needs to be tuned accordingly using `--batch` argument.
* `learning_rate` depends on `batch_size` hence if it `batch_size` is altered then `learning_rate` needs to be tuned accordingly.
* Total `epochs` is determined using **Cross-Validation** for provided training data. If **Training** data is changed then Total `epochs` needs to be tuned using **Cross-Validation**, setting `--all-data=0` in [train.py](train.py).
* While training, **Internet** Connection is required to download **ImageNet** weights for CNN Backbones.
* To reproduce the result, it is recommended to run code in same **Device Configuration**.
* For inference, `batch_size` is tuned for `8 x V100`. For any other device, `batch_size` may need to be modified. To modify `batch_size` change following codes in [predict.py](predict.py),

```py
# CONFIGURE BATCHSIZE
mx_dim = np.sqrt(np.prod(dim))
if mx_dim>=768 or any(i in model_name for i in ['convnext','ECA_NFNetL2']):
    CFG.batch_size = CFG.replicas * 16
elif mx_dim>=640  or any(i in model_name for i in ['EfficientNet','RegNet','ResNetRS50','ResNest50']):
    CFG.batch_size = CFG.replicas * 32
else:
    CFG.batch_size = CFG.replicas * 64
```
* For any queries, please contact `awsaf49@gmail.com`.

</details>

## Notebooks
To demonstrate **Training** and **Inference** 2 notebooks have been provided. It is recommended to go through them after `README.md`.
* **Inference**: To directly generate prediction on **eval** data without any **Training** using **provided** checkpoints, refer to [sp2022-infer-gpu](notebooks/sp2022-infer-gpu.ipynb) notebook at `notebooks/sp2022-infer-gpu.ipynb`
* **Training**: For training and then infering using newly trained weights refer to [sp2022-train-gpu](notebooks/sp2022-train-gpu.ipynb) notebook at `notebooks/sp2022-train-gpu.ipynb`

## 0. Requirements

<details>
<summary>Hardware</summary>
<br>
    
* GPU (model or N/A):   8x NVIDIA Tesla V100
* Memory (GB):   8 x 32GB
* OS: Amazon Linux
* CUDA Version : 11.0
* Driver Version : 450.119.04
* CPU RAM : 128 GiB
* DISK : 2 TB

</details>

### Library
Install necessary dependencies using following command,

```shell
!pip install -r requirements.txt
```

## 1. Data Preparation
* Step 1: Competition data needs to be in the `./data/` folder. It is mandatory to have the data in exact same format like it was provided. SP Cup dataset can be accessed from kaggle using below link,
    * [SP Cup 2022 Dataset](https://www.kaggle.com/datasets/awsaf49/sp-cup-2022-dataset)

* Step 2: External datasets need to be downloaded from following links and need to be in the `./data/` folder,
    1. LJSpeech: [link](https://www.kaggle.com/datasets/awsaf49/ljspeech-sr16k-dataset) (~2GB)
    2. VCTK: [link](https://www.kaggle.com/datasets/awsaf49/vctk-sr16k-dataset) (~3GB)
    3. LibriSpeech: [link](https://www.kaggle.com/datasets/awsaf49/librispeech-small-dataset) (~15GB)
    4. Synthetic: [link](https://www.kaggle.com/datasets/awsaf49/sp22-synthetic-dataset) (~5GB)

> **Note:** All the datasets were pre-processed to have exact same **sample_rate** = `16k` and **file_format** = `.wav`. 


### Data Path Format
Datasets are expected to have following format. To use custom directory, `PATHS.json` needs to modified

<details>
<summary>Path Structure</summary>
<br>
    
```shell
├── data
│   ├── sp22-synthetic-dataset
│   ├── librispeech-small-dataset
│   ├── ljspeech-sr16k-dataset
│   ├── vctk-sr16k-dataset
│   ├── spcup_2022_training_part1
│   │   └── spcup_2022_training_part1
│   ├── spcup_2022_unseen
│   │   └── spcup_2022_unseen
│   ├── spcup_2022_eval_part1
│   │   └── spcup_2022_eval_part1
│   ├── spcup_2022_eval_part2
│   │   └── spcup_2022_eval_part2
```

</details>

## 2. Supervisied Training
Competition & external data and their associated labels will be used for **Supervised Training**. All external data is considered as **Unknown Algorithm**.

### Part-1
For Training models for **eval_part1** data run following commands,

<details>
<summary>Code</summary>
<br>

```shell
!python3 train.py\
--cfg ./configs/sp22-part1.yaml\
--output-dir=output/supervised/part1\
--model=EfficientNetB0\
--batch=64\
--epochs=11
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part1.yaml\
--output-dir=output/supervised/part1\
--model=ResNet50D\
--batch=64\
--epochs=9
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part1.yaml\
--output-dir=output/supervised/part1\
--model=ResNetRS50\
--batch=32\
--epochs=13
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part1.yaml\
--output-dir=output/supervised/part1\
--model=ResNest50\
--batch=32\
--epochs=21
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part1.yaml\
--output-dir=output/supervised/part1\
--model=RegNetZD8\
--batch=64\
--epochs=8
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part1.yaml\
--output-dir=output/supervised/part1\
--model=EfficientNetV2S\
--pretrain=imagenet21k\
--batch=32\
--epochs=25
```

</details>


### Part-2
For Training models for **eval_part2** data run following commands,
<details>
<summary>Code</summary>
<br>

```shell
!python3 train.py\
--cfg ./configs/sp22-part2.yaml\
--output-dir=output/supervised/part2\
--model=ECA_NFNetL2\
--batch=16\
--epochs=12
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part2.yaml\
--output-dir=output/supervised/part2\
--model=convnext_base_in22k\
--batch=32\
--epochs=14
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part2.yaml\
--output-dir=output/supervised/part2\
--model=ResNetRS152\
--batch=32\
--epochs=11
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part2.yaml\
--output-dir=output/supervised/part2\
--model=convnext_large_in22k\
--batch=16\
--epochs=15
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part2.yaml\
--output-dir=output/supervised/part2\
--model=RegNetZD8\
--batch=32\
--epochs=5
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part2.yaml\
--output-dir=output/supervised/part2\
--model=EfficientNetB0\
--batch=64\
--epochs=13
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part2.yaml\
--output-dir=output/supervised/part2\
--model=EfficientNetV2M\
--pretrain=imagenet21k\
--batch=32\
--epochs=15
```
</details>

## 3. Generage Semi-Supervised Labels
Run following command to generate **Semi-Supervisied** lables for both **eval_part1&** & **eval_part2** data. Semi-Supervised labels will be saved at `output/supervised/pseudo/pred.csv`.

<details>
<summary>Code</summary>
<br>

```shell
!python generate_pseudo.py\
--part1-model-dir=output/supervised/part1\
--part1-infer-path=data/spcup_2022_eval_part1/spcup_2022_eval_part1\
--part2-model-dir=output/supervised/part2\
--part2-infer-path=data/spcup_2022_eval_part1/spcup_2022_eval_part2\
--output=output/supervised/pseudo/pred.csv
```
</details>

## 4. Semi-Supervised Training
In this stage Competition & External data will be used along with **eval_part1** & **eval_part2** data. For **eval_data** their **semi-supervised** labels will be used which were generated in previous stage.

### Part-1
For Training models for **eval_part1** data run following commands,
<details>
<summary>Code</summary>
<br>

```shell
!python3 train.py\
--cfg ./configs/sp22-part1.yaml\
--output-dir=output/semi-supervised/part1\
--model=EfficientNetB0\
--batch=64\
--epochs=15\
--pseudo 1\
--pseudo_csv=output/supervised/pseudo/pred.csv
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part1.yaml\
--output-dir=output/semi-supervised/part1\
--model=ResNet50D\
--batch=64\
--epochs=18\
--pseudo 1\
--pseudo_csv=output/supervised/pseudo/pred.csv
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part1.yaml\
--output-dir=output/semi-supervised/part1\
--model=ResNetRS50\
--batch=32\
--epochs=17\
--pseudo 1\
--pseudo_csv=output/supervised/pseudo/pred.csv
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part1.yaml\
--output-dir=output/semi-supervised/part1\
--model=ResNest50\
--batch=32\
--epochs=16\
--pseudo 1\
--pseudo_csv=output/supervised/pseudo/pred.csv
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part1.yaml\
--output-dir=output/semi-supervised/part1\
--model=RegNetZD8\
--batch=64\
--epochs=12\
--pseudo 1\
--pseudo_csv=output/supervised/pseudo/pred.csv
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part1.yaml\
--output-dir=output/semi-supervised/part1\
--model=EfficientNetV2S\
--pretrain=imagenet21k\
--batch=64\
--epochs=7\
--pseudo 1\
--pseudo_csv=output/supervised/pseudo/pred.csv
```
</details>

### Part-2
For Training models for **eval_part2** data run following commands,

<details>
<summary>Code</summary>
<br>

```shell
!python3 train.py\
--cfg ./configs/sp22-part2.yaml\
--output-dir=output/semi-supervised/part2\
--model=ECA_NFNetL2\
--batch=16\
--epochs=11\
--pseudo 1\
--pseudo_csv=output/supervised/pseudo/pred.csv
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part2.yaml\
--output-dir=output/semi-supervised/part2\
--model=convnext_base_in22k\
--batch=16\
--epochs=6\
--pseudo 1\
--pseudo_csv=output/supervised/pseudo/pred.csv
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part2.yaml\
--output-dir=output/semi-supervised/part2\
--model=ResNetRS152\
--batch=32\
--epochs=16\
--pseudo 1\
--pseudo_csv=output/supervised/pseudo/pred.csv
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part2.yaml\
--output-dir=output/semi-supervised/part2\
--model=convnext_large_in22k\
--batch=16\
--epochs=10\
--pseudo 1\
--pseudo_csv=output/supervised/pseudo/pred.csv
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part2.yaml\
--output-dir=output/semi-supervised/part2\
--model=RegNetZD8\
--batch=32\
--epochs=8\
--pseudo 1\
--pseudo_csv=output/supervised/pseudo/pred.csv
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part2.yaml\
--output-dir=output/semi-supervised/part2\
--model=EfficientNetB0\
--batch=32\
--epochs=10\
--pseudo 1\
--pseudo_csv=output/supervised/pseudo/pred.csv
```
```shell
!python3 train.py\
--cfg ./configs/sp22-part2.yaml\
--output-dir=output/semi-supervised/part2\
--model=EfficientNetV2M\
--pretrain=imagenet21k\
--batch=32\
--epochs=25\
--pseudo 1\
--pseudo_csv=output/supervised/pseudo/pred.csv
```
</details>

## 5. Prediction with **Trained** Models
For predicting on **eval_data** using newly trained models use following codes,

### Part-1
To generate prediction for **eval_part1** data using **newly-trained** checkpoints run following commands,
<details>
<summary>Code</summary>
<br>

```shell
!python predict.py\
--cfg ./configs/sp22-part1.yaml\
--model-dir=output/semi-supervised/part1\
--infer-path=data/spcup_2022_eval_part1/spcup_2022_eval_part1\
--output=output/result/pred_part1.csv
```
</details>

### Part-2
To generate prediction for **eval_part2** data using **newly-trained** checkpoints run following commands,
<details>
<summary>Code</summary>
<br>

```shell
!python predict.py\
--cfg ./configs/sp22-part2.yaml\
--model-dir=output/semi-supervised/part2\
--infer-path=data/spcup_2022_eval_part2/spcup_2022_eval_part2\
--output=output/result/pred_part2.csv
```
</details>


## 6. Prediction with **Pre-Trained** Checkpoints
To generage prediction on **eval_data** directly using pre-trained **chekpointss**, first download the **checkpoints** using following links,
* part-1: [link](https://www.kaggle.com/dataset/d79184c8abc6c80751a9b25cf64df93f22038e5c4a23a6644db8671fab15143a) (~1GB)
* part-2: [link](https://www.kaggle.com/dataset/bd78f14359c5c0926aa84da4b1636590f2e70d6978bb7fb6f31ef46607ae7cf2) (~6GB)

Extract the `.zip` files and keep the **part1** files on `./checkpoints/part1` folder and **part2**  files on `./checkpoints/part2` folder. So, final file structure will look like this,

<details>
<summary>Part-1 Structure (6 Models)</summary>
<br>

```
./checkpoints/part1
├── EfficientNetB0-128x384
│   └── ckpt
│       └── model.h5
├── EfficientNetV2S-128x384
│   └── ckpt
│       └── model.h5
├── RegNetZD8-128x384
│   └── ckpt
│       └── model.h5
├── ResNest50-128x384
│   └── ckpt
│       └── model.h5
├── ResNet50D-128x384
│   └── ckpt
│       └── model.h5
└── ResNetRS50-128x384
    └── ckpt
        └── model.h5
```
</details>

<details>
<summary>Part-2 Structure (7 Models))</summary>
<br>

```
./checkpoints/part2
├── ECA_NFNetL2-256x512
│   └── ckpt
│       └── model.h5
├── EfficientNetB0-256x512
│   └── ckpt
│       └── model.h5
├── EfficientNetV2M-256x512
│   └── ckpt
│       └── model.h5
├── RegNetZD8-256x512
│   └── ckpt
│       └── model.h5
├── ResNetRS152-256x512
│   └── ckpt
│       └── model.h5
├── convnext_base_in22k-256x512
│   └── ckpt
│       └── model.h5
└── convnext_large_in22k-256x512
    └── ckpt
        └── model.h5
```

</details>

Then use following commands to generate predictions for **eval_data**,

### Part-1
To generate prediction for **eval_part1** data using **provided** checkpoints run following commands,
<details>
<summary>Code</summary>
<br>

```shell
!python predict.py\
--cfg ./configs/sp22-part1.yaml\
--model-dir=checkpoints/part1\
--infer-path=data/spcup_2022_eval_part1/spcup_2022_eval_part1\
--output=output/result/pred_part1.csv
```
</details>

### Part-2
To generate prediction for **eval_part2** data using **provided** checkpoints run following commands,
<details>
<summary>Code</summary>
<br>

```shell
!python predict.py\
--cfg ./configs/sp22-part2.yaml\
--model-dir=checkpoints/part2\
--infer-path=data/spcup_2022_eval_part2/spcup_2022_eval_part2\
--output=output/result/pred_part2.csv
```

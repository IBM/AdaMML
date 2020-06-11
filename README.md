# AdaMML: Adaptive Multi-Modal Learning for Efficient Video Recognition

This is the source codes for the submission: AdaMML: Adaptive Multi-Modal Learning for Efficient Video Recognition (Paper ID: 1207).

## Requirements

```
pip3 install torch torchvision librosa tqdm Pillow numpy 
```

## Data Preparation
The dataloader (utils/video_dataset.py) can load RGB frames stored in the following format:
```
-- dataset_dir
---- train.txt
---- val.txt
---- test.txt
---- videos
------ video_0_folder
-------- 00001.jpg
-------- 00002.jpg
-------- ...
------ video_1_folder
------ ...
```

Each line in `train.txt` and `val.txt` includes 4 elements and separated by a symbol, e.g. space (` `) or semicolon (`;`). 
Four elements (in order) include (1) relative paths to `video_x_folder` from `dataset_dir`, (2) starting frame number, usually 1, (3) ending frame number, (4) label id (a numeric number).

E.g., a `video_x` has `300` frames and belong to label `1`.
```
path/to/video_x_folder 1 300 1
```
The difference for `test.txt` is that each line will only have 3 elements (no label information).

The same format is used for `optical flow` but each file (`00001.jpg`) need to be `x_00001.jpg` and `y_00001.jpg`.

On the other hand, for audio data, you need to change the first elements to the path of corresponding `wav` files, like

```
path/to/audio_x.wav 1 300 1
```

After that, you need to update the `utils/data_config.py` for the datasets accordingly.

We provided the scripts in the `tools` folder to extract RGB frames and audios from a video. To extract the optical flow, we use the docker image provided by [TSN](https://hub.docker.com/r/bitxiong/tsn/). 

Please see the help in the script.

## Pretrained models

We provided pretrained models on the Kinetics-Sounds dataset, including unimodality models and the AdaMML models.

### Unimodality
 - [RGB](https://drive.google.com/file/d/1PaBa3ZCVvgkbkborB4MhPpp1qh1J2VBg) 
 - [Audio](https://drive.google.com/file/d/1rYyGNj4adkGEWR_4MJaybo9CZxyuyryw) 
 - [Flow](https://drive.google.com/file/d/11B4L0ukjXr2okjQo0ARJG2bqwUZph0bG)

### AdaMML:
 - [RGB + Audio](https://drive.google.com/file/d/1S5A1AoRrZsI_pmb3dJAebqpHJ3AgiPZp) 
 - [RGB + Flow with RGBDiff as Proxy](https://drive.google.com/file/d/1gL2hj1WvCirUNkJdIKHyP79MNbPdpwa1) 
 - [RGB + Audio + Flow with RGBDiff as Proxy](https://drive.google.com/file/d/1I_GzU_ODSUZ_hiqPNZ6bzQclXnaKqvmN)

## Training

After downloding the unimodality pretrained models, here is the command template to train AdaMML:

```shell script
python3 train.py --multiprocessing-distributed --backbone_net adamml -d 50 \
--groups 8 --frames_per_group 4 -b 72 -j 96 --epochs 20 --warmup_epochs 5 --finetune_epochs 10 \
--modality MODALITY1 MODALITY2 --datadir /PATH/TO/MODALITY1 /PATH/TO/MODALITY2 --dataset DATASET --logdir LOGDIR \
--dense_sampling --fusion_point logits --unimodality_pretrained /PATH/TO/MODEL_MODALITY1 /PATH/TO/MODEL_MODALITY2 \
--learnable_lf_weights --num_segments 5 --cost_weights 1.0 0.005 --causality_modeling lstm --gammas 10.0 --sync-bn \
--lr 0.001 --p_lr 0.01 --lr_scheduler multisteps --lr_steps 10 15
```

The length of the following arguments depended on how many modalities you would like to include in AdaMML.
 - `--modality`: the modalities, other augments needs to follow this order
 - `--datadir`: the data dir for each modality
 - `--unimodality_pretrained`: the pretrained unimodality model

Note that, to use `rgbdiff` as a proxy, both `rgbdiff` and `flow` needs to be specified in `--modality` and their corresponding `--datadir`.
However, you only need to provided `flow` pretrained model in the `--unimodality_pretrained`

Here are the examples to train AdaMML with different combinations.

RGB + Audio

```shell script
python3 train.py --multiprocessing-distributed --backbone_net adamml -d 50 \
--groups 8 --frames_per_group 4 -b 72 -j 96 --epochs 20 --warmup_epochs 5 --finetune_epochs 10 \
--modality rgb sound --datadir /PATH/TO/RGB_DATA /PATH/TO/AUDIO_DATA --dataset DATASET --logdir LOGDIR \
--dense_sampling --fusion_point logits --unimodality_pretrained /PATH/TO/RGB_MODEL /PATH/TO/AUDIO_MODEL \
--learnable_lf_weights --num_segments 5 --cost_weights 1.0 0.05 --causality_modeling lstm --gammas 10.0 --sync-bn \
--lr 0.001 --p_lr 0.01 --lr_scheduler multisteps --lr_steps 10 15
```

RGB + Flow (with RGBDiff as Proxy)

```shell script
python3 train.py --multiprocessing-distributed --backbone_net adamml -d 50 \
--groups 8 --frames_per_group 4 -b 72 -j 96 --epochs 20 --warmup_epochs 5 --finetune_epochs 10 \
--modality rgb flow rgbdiff --datadir /PATH/TO/RGB_DATA /PATH/TO/FLOW_DATA /PATH/TO/RGB_DATA --dataset DATASET --logdir LOGDIR \
--dense_sampling --fusion_point logits --unimodality_pretrained /PATH/TO/RGB_MODEL /PATH/TO/FLOW_MODEL \
--learnable_lf_weights --num_segments 5 --cost_weights 1.0 1.0 --causality_modeling lstm --gammas 10.0 --sync-bn \
--lr 0.001 --p_lr 0.01 --lr_scheduler multisteps --lr_steps 10 15
```

RGB + Audio + Flow (with RGBDiff as Proxy)

```shell script
python3 train.py --multiprocessing-distributed --backbone_net adamml -d 50 \
--groups 8 --frames_per_group 4 -b 72 -j 96 --epochs 20 --warmup_epochs 5 --finetune_epochs 10 \
--modality rgb sound flow rgbdiff --datadir /PATH/TO/RGB_DATA /PATH/TO/AUDIO_DATA /PATH/TO/FLOW_DATA /PATH/TO/RGB_DATA --dataset DATASET --logdir LOGDIR \
--dense_sampling --fusion_point logits --unimodality_pretrained /PATH/TO/RGB_MODEL /PATH/TO/SOUND_MODEL /PATH/TO/FLOW_MODEL \
--learnable_lf_weights --num_segments 5 --cost_weights 0.5 0.05 0.8 --causality_modeling lstm --gammas 10.0 --sync-bn \
--lr 0.001 --p_lr 0.01 --lr_scheduler multisteps --lr_steps 10 15
```


## Evaluation

To test AdaMML model is straight-forward, you can simply use the training command with following modifications
 - add `-e` in the command
 - use `--pretrained` for the trained model
 - remove `--multiprocessing-distributed`
 - set `--val_num_clips` to test under different number of video segments (default is 10)

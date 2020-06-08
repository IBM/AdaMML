# AdaMML: Adaptive Multi-Modal Learning forEfficient Video Recognition


## Requirements

```
pip3 install torch torchvision librosa
```

## Data Preparation
The dataloader (utils/video_dataset.py) can load videos (image sequences) stored in the following format:
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

After that, you need to update the `utils/data_config.py` for the datasets accordingly.

We provided scripts in the `tools` folder to extract RGB frames, optical flow and audios from a video.
Please see the help in the script.

## Training

We provided the pretrained unimodality model of Kinetics-Sounds. ([rgb]() [sound]() [flow]().)

After download the models, you can use following command to train AdaMML with different combinations.
- rgb + audio
- rgb + flow (with rgbdiff proxy)
- rgb + audio + flow (with rgbdiff proxy)

Here is the command template:

```shell script
python3 train_adamml_stage.py --multiprocessing-distributed --backbone_net adamml -d 50 \
--groups 8 --frames_per_group 4 -b 72 -j 96 --epochs 20 --warmup_epochs 5 --finetune_epochs 10 \
--modality MODALITY1 MODALITY2 --datadir /PATH/TO/MODALITY1 /PATH/TO/MODALITY2 --dataset DATASET --logdir LOGDIR \
--dense_sampling --fusion_point logits --unimodality_pretrained /PATH/TO/MODEL_MODALITY1 /PATH/TO/MODEL_MODALITY2 \
--learnable_lf_weights --num_segments 5 --cost_weights 1.0 0.005 --causality_modeling lstm --gammas 10.0 --sync-bn \
--lr 0.001 --p_lr 0.01 --lr_scheduler multisteps --lr_steps 10 15
```

The length of the following argments depended on how many modalities you would like to include in AdaMML.
 - `--modality`: the modalities, other augments needs to follow this order
 - `--datadir`: the data dir for each modality
 - `--unimodality_pretrained`: the pretrained unimodality model

Note that, to use `rgbdiff` as a proxy, both `rgbdiff` and `flow` needs to be specified in `--modality` and their corresponding `--datadir`.
However, you only need to provided `flow` pretrained model in the `--unimodality_pretrained`

Here are the examples:

RGB + Audio

```shell script
python3 train_adamml_stage.py --multiprocessing-distributed --backbone_net adamml -d 50 \
--groups 8 --frames_per_group 4 -b 72 -j 96 --epochs 20 --warmup_epochs 5 --finetune_epochs 10 \
--modality rgb sound --datadir /PATH/TO/RGB_DATA /PATH/TO/AUDIO_DATA --dataset DATASET --logdir LOGDIR \
--dense_sampling --fusion_point logits --unimodality_pretrained /PATH/TO/RGB_MODEL /PATH/TO/AUDIO_MODEL \
--learnable_lf_weights --num_segments 5 --cost_weights 1.0 0.005 --causality_modeling lstm --gammas 10.0 --sync-bn \
--lr 0.001 --p_lr 0.01 --lr_scheduler multisteps --lr_steps 10 15
```

RGB + Flow (with rgbdiff as proxy)

```shell script
python3 train_adamml_stage.py --multiprocessing-distributed --backbone_net adamml -d 50 \
--groups 8 --frames_per_group 4 -b 72 -j 96 --epochs 20 --warmup_epochs 5 --finetune_epochs 10 \
--modality rgb flow rgbdiff --datadir /PATH/TO/RGB_DATA /PATH/TO/FLOW_DATA /PATH/TO/RGB_DATA --dataset DATASET --logdir LOGDIR \
--dense_sampling --fusion_point logits --unimodality_pretrained /PATH/TO/RGB_MODEL /PATH/TO/FLOW_MODEL \
--learnable_lf_weights --num_segments 5 --cost_weights 1.0 0.005 --causality_modeling lstm --gammas 10.0 --sync-bn \
--lr 0.001 --p_lr 0.01 --lr_scheduler multisteps --lr_steps 10 15
```

RGB + Audio + Flow (with rgbdiff as proxy)

```shell script
python3 train_adamml_stage.py --multiprocessing-distributed --backbone_net adamml -d 50 \
--groups 8 --frames_per_group 4 -b 72 -j 96 --epochs 20 --warmup_epochs 5 --finetune_epochs 10 \
--modality rgb sound flow rgbdiff --datadir /PATH/TO/RGB_DATA /PATH/TO/AUDIO_DATA /PATH/TO/FLOW_DATA /PATH/TO/RGB_DATA --dataset DATASET --logdir LOGDIR \
--dense_sampling --fusion_point logits --unimodality_pretrained /PATH/TO/RGB_MODEL /PATH/TO/SOUND_MODEL /PATH/TO/FLOW_MODEL \
--learnable_lf_weights --num_segments 5 --cost_weights 1.0 0.005 --causality_modeling lstm --gammas 10.0 --sync-bn \
--lr 0.001 --p_lr 0.01 --lr_scheduler multisteps --lr_steps 10 15
```


## Evaluation

To test adaMML model is straight-forward:
 - add `-e` in the command
 - use `--pretrained` for the trained model
 - remove `--multiprocessing-distributed`

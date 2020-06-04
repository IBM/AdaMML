# AdaMML: Adaptive Multi-Modal Learning forEfficient Video Recognition

## Library Requirements

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

### Training Unimodality from ImageNet-pretrained weights 

```shell script
python3 train.py --multiprocessing-distributed --backbone_net resnet -d 50 \
--groups 8 --frames_per_group 4 -b 72 -j 96 --modality MODALITY \
--datadir /PATH/TO/FOLDER --dataset DATASET --logdir LOGDIR --dense_sampling  --epochs 100 \
--augmentor_ver v2 
```
`MODALITY` could be one of `{rgb, rgbdiff, flow, sound}`.

E.g., training with `audio` modality. 

```shell script
python3 train.py --multiprocessing-distributed --backbone_net sound_mobilenet_v2 \
--groups 8 --frames_per_group 4 -b 72 -j 96 --modality sound \
--datadir /PATH/TO/FOLDER --dataset DATASET --logdir LOGDIR --dense_sampling  --epochs 100
```

### Naive joint training
In joint training, few things are different from the unimodality training:
 - set multiple modalities
 - set multiple data directories corresponding to the modalities
 - choose how to fuse different modalities, e.g. `logits`
 
E.g.
```shell script
python3 train_joint.py --multiprocessing-distributed --backbone_net joint_resnet_mobilenet_V2 -d 50 \
--groups 8 --frames_per_group 4 -b 72 -j 96 --modality MODALITY1 MODALITY2 \
--datadir /PATH/TO/MODALITY1 /PATH/TO/MODALITY2 --dataset DATASET --logdir LOGDIR --dense_sampling --epochs 35 \   
--fusion_point logits --learnable_lf_weights   
```

### AdaMML training
After you trained the unimodality models separately, AdaMML will use those models in the recognition network.
The script here is similar to Naive joint training but with few difference:
 - 
 
```shell script
python3 train_adamml_stage.py --multiprocessing-distributed --backbone_net adamml -d 50 \
--groups 8 --frames_per_group 4 -b 72 -j 96 --epochs 20 --warmup_epochs 5 --finetune_epochs 10 \
--modality MODALITY1 MODALITY2 --datadir /PATH/TO/MODALITY1 /PATH/TO/MODALITY2 --dataset DATASET --logdir LOGDIR \
--dense_sampling --fusion_point logits --unimodality_pretrained /PATH/TO/MODEL_MODALITY1 /PATH/TO/MODEL_MODALITY2 \
--learnable_lf_weights --num_segments 5 --cost_weights 1.0 0.005 --causality_modeling lstm --gammas 10.0
```

## Testing

### multi-segment/clip testing
This is only for `unimodality/joint training`.

As the default evaluation in unimodality/joint training only uses `1` segment/clip, we have other scripts to perform 10 segments/clips testing. 
(The number we reported in the paper.) 

Everything will be similar to training script and here are difference:
 - change to `test.py` or `test_joint.py` accordingly
 - add `-e` to perform evalution
 - add `--num_clips 10` to evaluate 10 clips
 - add the to-be-tested model path to `--pretrained` 

Note, if you encounter out of memory error, you can use smaller batch size. 
 
E.g.
```shell script
python3 test.py --multiprocessing-distributed --backbone_net sound_mobilenet_v2 \
--groups 8 --frames_per_group 4 -b 72 -j 96 --modality sound \
--datadir /PATH/TO/FOLDER --dataset DATASET --logdir LOGDIR --dense_sampling \
--pretrained /PATH/TO/MODEL --num_clips 10
```

### Testing AdaMML model

To test adaMML model is straight-forward, simply put `-e` in the command.




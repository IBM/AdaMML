# Action Recognition Study

## Requirements

```
pip3 install torch
pip3 install torchvision
pip3 install tqdm pyarrow lmdb tensorboard_logger librosa
pip3 install git+https://github.com/chunfuchen/pytorch-summary
```

## training script

### Unimodality training

Training with `audio` modality. 

```shell script
python3 train_dist.py --backbone_net sound_resnet -d 18 --groups 8 --frames_per_group 4 -b 72 \
-j 96  --modality sound --logdir LOGDIR --multiprocessing-distribute --dense_sampling \
--dataset DATASET --datadir /PATH/TO/FOLDER --epochs 100 --wd 0.0001
```

Training with other modalities, e.g. RGB, RGBdiff, flow. Need to setup `augmentor_ver=v2` 
`MODALITY={'rgb', 'rgbdiff', 'flow'}`.

```shell script
python3 train_dist.py --multiprocessing-distributed --backbone_net resnet -d 50 \
--groups 8 -b 72 --epochs 100 -j 96 --datadir /PATH/TO/FOLDER --modality MODALITY \
--dataset DATASET --logdir LOGDIR --dense_sampling --wd 0.0001 \
--augmentor_ver v2 --frames_per_group 4
```

### Joint learning

For joint learning, I used to use two nodes to assure the batch size is 72. 
For now, the network used in two modalities needed to be the same, will extend it later.
```shell script
python3 train_joint.py --multiprocessing-distributed --backbone_net joint_resnet -d 50 --groups 8 \
--epochs 100 --dataset DATASET --datadir /PATH/TO/MODALITY1 /PATH/TO/MODALITY2 --dense_sampling 
--logdir LOGDIR --modality MODALITY1 MODALITY2 --wd 0.0001 --augmentor_ver v2 --fusion_point fc2 \ 
--frames_per_group 4 -j 96 -b 72 --world-size 2 --dist-url tcp://URL:PORT --rank 0 
```


## multi-clip testing
Everything will be similar to training script but you need to pass `num_clips` to define how many clips.
Furthermore, `-e` is needed if you are running validation not test (with unknown label).
Do not forget add `pretrained` to load trained models.
Note: you might want to test both the last `checkpoint` and the `best` model to see which one is better. 

```shell script
python3 test.py --backbone_net sound_resnet -d 18 --groups 8 --frames_per_group 4 -b 72 \
-j 96  --modality sound --logdir LOGDIR --multiprocessing-distribute --dense_sampling \
--dataset DATASET --datadir /PATH/TO/FOLDER --epochs 100 --wd 0.0001
```

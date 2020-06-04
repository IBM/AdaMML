
import numpy as np
import torch
import torch.nn as nn

from models.resnet import ResNet
from models.sound_mobilenet_v2 import MobileNetV2

__all__ = ['joint_resnet_mobilenetv2']

class JointResNetMobileNetV2(nn.Module):

    def __init__(self, depth, num_frames, modality, num_classes=1000, dropout=0.5, zero_init_residual=False,
                 without_t_stride=False, pooling_method='max', input_channels=None,
                 fusion_point='logits', learnable_lf_weights=False):
        super().__init__()

        self.depth = depth
        self.num_frames = num_frames
        self.without_t_stride = without_t_stride
        self.pooling_method = pooling_method
        self.fusion_point = fusion_point
        self.modality = modality
        self.learnable_lf_weights = learnable_lf_weights

        self.nets = nn.ModuleList()
        self.last_channels = []
        for i, m in enumerate(modality):
            if m != 'sound':
                net = ResNet(depth, num_frames, num_classes, dropout, zero_init_residual,
                             without_t_stride, pooling_method, input_channels[i])
                if self.fusion_point != 'logits':
                    del net.avgpool
                    del net.dropout
                    del net.fc

                if depth >= 50:
                    self.last_channels.append(2048)
                else:
                    self.last_channels.append(512)
            else:
                net = MobileNetV2(num_classes, dropout=dropout, input_channels=input_channels[i])
                if self.fusion_point != 'logits':
                    del net.classifier
                self.last_channels.append(net.last_channel)
            self.nets.append(net)

        self.lf_weights = None

        if self.fusion_point != 'logits':
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            in_feature_c = sum(self.last_channels)
            out_feature_c = 2048
            self.joint = nn.Sequential(
                nn.Linear(in_feature_c, out_feature_c), nn.ReLU(True),
                nn.Linear(out_feature_c, out_feature_c), nn.ReLU(True)
            )
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(out_feature_c, num_classes)
        else:
            init_prob = 1.0 / len(self.modality)
            if learnable_lf_weights:
                self.lf_weights = nn.Parameter(torch.tensor([init_prob] * (len(self.modality) - 1)))
                self.register_parameter('lf_weights', self.lf_weights)

    def mean(self, modality='rgb'):
        return [0.485, 0.456, 0.406] if modality == 'rgb' or modality == 'rgbdiff' \
            else [0.5]

    def std(self, modality='rgb'):
        return [0.229, 0.224, 0.225] if modality == 'rgb' or modality == 'rgbdiff' \
            else [np.mean([0.229, 0.224, 0.225])]

    @property
    def network_name(self):
        name = 'joint_resnet-{}_mobilenet_v2-{}'.format(self.depth, self.fusion_point)
        if self.lf_weights is not None:
            name += "-llf" if self.learnable_lf_weights else '-llfc'
        if not self.without_t_stride:
            name += "-ts-{}".format(self.pooling_method)

        return name

    def forward(self, multi_modalities, decisions=None):
        # multi_modalities is a list
        bs, _, _, _ = multi_modalities[0].shape
        out = []
        #TODO: now only support 8 frames case
        for i, x in enumerate(multi_modalities):
            tmp = self.nets[i].features(x) if self.fusion_point != 'logits' else self.nets[i].forward(x)
            tmp = self.avgpool(tmp) if self.fusion_point != 'logits' else tmp

            if decisions is not None:
                if self.fusion_point == 'logits':
                    tmp = tmp * decisions[i].view((tmp.size(0), 1))
                else:
                    raise ValueError("only support logits mode")
            out.append(tmp)

        if self.fusion_point != 'logits':
            out = torch.cat(out, dim=1)
            out = out.view(out.size(0), -1)
            out = self.joint(out)
            out = self.dropout(out)
            out = self.fc(out)

            n_t, c = out.shape
            out = out.view(bs, -1, c)

            # average the prediction from all frames
            out = torch.mean(out, dim=1)
        else:
            out = torch.stack(out, dim=0)
            out.squeeze_(-1)
            out.squeeze_(-1)  # MxNxC
            if self.lf_weights is not None:
                if self.lf_weights.dim() > 1:
                    comple_weights = torch.ones((1, self.lf_weights.size(-1)), dtype=self.lf_weights.dtype,
                                                device=self.lf_weights.device) \
                                     - torch.sum(self.lf_weights, dim=0)
                else:
                    comple_weights = torch.ones(1, dtype=self.lf_weights.dtype, device=self.lf_weights.device) - torch.sum(self.lf_weights, dim=0)
                weights = torch.cat((self.lf_weights, comple_weights), dim=0)
                weights = weights.view(weights.size(0), 1, -1)
                out = out * weights
                out = torch.sum(out, dim=0)
            else:
                out = torch.mean(out, dim=0)
        return out


def joint_resnet_mobilenetv2(depth, num_classes, without_t_stride, groups, dropout, pooling_method,
                             input_channels, fusion_point, modality, unimodality_pretrained,
                             learnable_lf_weights, **kwargs):

    model = JointResNetMobileNetV2(depth, num_frames=groups, num_classes=num_classes,
                                   without_t_stride=without_t_stride, dropout=dropout,
                                   pooling_method=pooling_method, input_channels=input_channels,
                                   fusion_point=fusion_point, modality=modality,
                                   learnable_lf_weights=learnable_lf_weights)

    if len(unimodality_pretrained) > 0:
        if len(unimodality_pretrained) != len(model.nets):
            raise ValueError("the number of pretrained models is incorrect.")
        for i, m in enumerate(modality):
            print("Loading unimodality pretrained model from: {}".format(unimodality_pretrained[i]))
            state_dict = torch.load(unimodality_pretrained[i], map_location='cpu')['state_dict']
            new_state_dict = {key.replace("module.", ""): v for key, v in state_dict.items()}
            if fusion_point != 'logits':
                if m != 'sound':
                    new_state_dict.pop('fc.weight', None)
                    new_state_dict.pop('fc.bias', None)
                else:
                    new_state_dict.pop('classifier.1.weight', None)
                    new_state_dict.pop('classifier.1.bias', None)
            model.nets[i].load_state_dict(new_state_dict, strict=True)

    return model

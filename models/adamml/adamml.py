
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.adamml import p_joint_mobilenet
from models.joint_resnet_mobilenetv2 import joint_resnet_mobilenetv2

__all__ = ['adamml']


class AdaMML(nn.Module):

    def __init__(self, policy_net, main_net, num_frames, num_segments, modality, rng_policy, rng_threshold, num_classes):
        super().__init__()
        self.rng_policy = rng_policy
        self.policy_net = policy_net
        self.main_net = main_net
        self.num_segments = num_segments
        self.num_frames = num_frames * num_segments
        self.num_frames_per_segment = num_frames
        self.modality = modality

        if 'rgbdiff' in modality and 'flow' in modality:
            self.num_modality = len(modality) - 1
        else:
            self.num_modality = len(modality)
        self.p_data_idx = [self.modality.index(x) for x in self.policy_net.modality]
        self.m_data_idx = [self.modality.index(x) for x in self.main_net.modality]

        self.rng_threshold = rng_threshold

        self.decay_ratio = 0.965

        self.update_policy_net = True
        self.update_main_net = True

        if self.rng_policy:
            self.freeze_policy_net()
            del self.policy_net.fcs

    def data_layer(self, x, num_segments, p_rgb_size=(160, 160)):
        p_x, m_x = [], []
        idx = 0
        for x_, m in zip(x, self.modality):
            if m == 'sound':
                # when getting consecutive segments for sound, the signals are stacked at the last dim
                # however, if getting 10 clips, the signals are stacked at the second dim
                if x_.size(-1) != x_.size(-2): # in training
                    tmp = x_.chunk(num_segments, dim=-1)
                    tmp = torch.stack(tmp, dim=0).contiguous()
                else:
                    tmp = x_.view((x_.size(0), num_segments, -1,) + x_.shape[-2:]).transpose(0, 1).contiguous()
                p_x.append(tmp)
                m_x.append(tmp)
            else:  # only subsampling non-sound
                if idx in self.p_data_idx:
                    b, s_f_c, h, w = x_.shape
                    tmp = F.interpolate(x_, size=p_rgb_size, mode='bilinear')
                    tmp = tmp.view((b, num_segments, self.num_frames_per_segment, -1, ) + p_rgb_size)
                    tmp = tmp[:, :, range(0, self.num_frames_per_segment, 2), ...]
                    tmp = tmp.view((b, num_segments, -1,) + p_rgb_size).transpose(0, 1).contiguous()
                    p_x.append(tmp)
                if idx in self.m_data_idx:
                    m_x.append(x_.view((x_.size(0), num_segments, -1,) + x_.shape[-2:]).transpose(0, 1).contiguous())
            idx += 1
        return p_x, m_x, num_segments

    def forward(self, x, num_segments=None):
        # x: [Nx(SFC)xHxW], N is batch size, S segment/clip, F frames per clip, C channels, length of list is M
        # [SxNxFCxHxW], S normal input of networks, conversion for all modalities, length of list is M
        num_segments = num_segments if num_segments else self.num_segments
        p_x, m_x, num_segments = self.data_layer(x, num_segments)
        if not self.rng_policy:
            decisions, decision_logits = self.policy_net(p_x)
        else:
            decisions = (torch.rand((num_segments, self.num_modality, x[0].size(0)),
                                    dtype=x[0].dtype, device=x[0].device) > self.rng_threshold).float()
        # SxMxN tensors, M is number modality and N is batch size, each element is 0 or 1
        
        # in main net, run each segment one by one to save memory and
        # use decision to mask out the output, but still run whole main network
        all_logits = []  # NxSxC_class
        for i in range(num_segments):
            tmp_x = ([m_x[m_i][i, ...] for m_i in range(self.num_modality)])
            all_logits.append(self.main_net(tmp_x, decisions[i]))  # NxC

        final_logits = torch.stack(all_logits, dim=1).mean(dim=1)
        # reshape, let batch as the first index
        decisions = decisions.permute((2, 0, 1))
        return final_logits, decisions

    def mean(self, modality='rgb'):
        return [0.485, 0.456, 0.406] if modality == 'rgb' or modality == 'rgbdiff' \
            else [0.5]

    def std(self, modality='rgb'):
        return [0.229, 0.224, 0.225] if modality == 'rgb' or modality == 'rgbdiff' \
            else [np.mean([0.229, 0.224, 0.225])]

    @property
    def network_name(self):
        name = 'adamml'
        if self.rng_policy:
            name += '-rng-{:.1f}'.format(self.rng_threshold)
        else:
            name += '-{}'.format(self.policy_net.network_name)
        name += '-{}'.format(self.main_net.network_name)
        return name

    def decay_temperature(self, decay_ratio=None):
        self.policy_net.decay_temperature(decay_ratio if decay_ratio else self.decay_ratio)

    def freeze_policy_net(self):
        self.update_policy_net = False
        for param in self.policy_net.parameters():
            param.requires_grad = False

    def unfreeze_policy_net(self):
        self.update_policy_net = True
        for param in self.policy_net.parameters():
            param.requires_grad = True

    def freeze_main_net(self):
        self.update_main_net = False
        for param in self.main_net.parameters():
            param.requires_grad = False

    def unfreeze_main_net(self):
        self.update_main_net = True
        for param in self.main_net.parameters():
            param.requires_grad = True

def adamml(
        # shared parameters
        groups, modality, input_channels, num_segments, rng_policy, rng_threshold, whole_video, logits_modeling,
        adamml_ver,
        # policy net parameters
        p_adamml_ver, causality_modeling, hard_gumbel, single_fc, p_unimodality_pretrained, policy_net_pretrained,
        lstm_num_layers,
        # main net parameters
        num_classes, dropout_path, depth, without_t_stride, dw_conv, temporal_module_name,
        blending_frames, blending_method, dropout, pooling_method, fusion_point,
        unimodality_pretrained, learnable_lf_weights, learnable_lf_weights_per_class,
        main_net_pretrained, all_resnet, **kwargs):

    if 'rgbdiff' in modality and 'flow' in modality:
        p_modality = [x for x in modality if x != 'flow']
        m_modality = [x for x in modality if x != 'rgbdiff']
        p_input_channels = [x for x, m in zip(input_channels, modality) if m != 'flow']
        m_input_channels = [x for x, m in zip(input_channels, modality) if m != 'rgbdiff']
    else:
        p_modality = modality
        m_modality = modality
        p_input_channels = input_channels
        m_input_channels = input_channels

    # policy net
    policy_net = p_joint_mobilenet(num_frames=max(1, groups // 2), modality=p_modality,
                                   input_channels=p_input_channels, causality_modeling=causality_modeling)

    main_net = joint_resnet_mobilenetv2(depth=depth, num_classes=num_classes,
                                        without_t_stride=without_t_stride,
                                        groups=groups, dropout=dropout,
                                        pooling_method=pooling_method,
                                        input_channels=m_input_channels,
                                        fusion_point=fusion_point, modality=m_modality,
                                        unimodality_pretrained=unimodality_pretrained,
                                        learnable_lf_weights=learnable_lf_weights)

    model = AdaMML(policy_net, main_net, num_frames=groups,
                   num_segments=num_segments, modality=modality, rng_policy=rng_policy,
                   rng_threshold=rng_threshold, num_classes=num_classes)

    return model

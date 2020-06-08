import torch
from torch import nn
import math
import torch.distributions
import torch.nn.functional as F
import numpy as np
import torch.utils.model_zoo as model_zoo


from models.common import TemporalPooling


model_urls = {
    'mobilenet_v2': 'https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2_160x160-64dc7fa1.pth'
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, num_frames=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.temporal_pool = TemporalPooling(num_frames, mode='max') if num_frames else None
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.temporal_pool:
            x = self.temporal_pool(x)

        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, num_frames=4, input_channels=3, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.input_channels = input_channels
        self.num_frames = num_frames
        self.orig_num_frames = num_frames
        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(input_channels, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            has_tp = True if c == 64 or c == 160 else False
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                num_frames = self.num_frames if i == 0 and has_tp and self.num_frames != 1 \
                    else None
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t,
                                    num_frames=num_frames))
                input_channel = output_channel
            if has_tp:
                self.num_frames //= 2
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.last_channel = int(1280 * width_mult)
        output_channel = _make_divisible(self.last_channel, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def feature_extraction(self, x):
        bs, c_t, h, w = x.shape
        x = x.view(bs * self.orig_num_frames, c_t // self.orig_num_frames, h, w)
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x, with_feature=False):
        bs, c_t, h, w = x.shape
        fea = self.feature_extraction(x)
        # x = self.dropout(fea)
        x = self.classifier(fea)
        n_t, c = x.shape
        out = x.view(bs, -1, c)

        # average the prediction from all frames
        out = torch.mean(out, dim=1)
        if with_feature:
            return out, fea
        else:
            return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def mean(self, modality='rgb'):
        return [0.485, 0.456, 0.406] if modality == 'rgb' or modality == 'rgbdiff'\
            else [0.5]

    def std(self, modality='rgb'):
        return [0.229, 0.224, 0.225] if modality == 'rgb' or modality == 'rgbdiff'\
            else [np.mean([0.229, 0.224, 0.225])]

    @property
    def network_name(self):
        name = 'mobilenet_v2'
        return name

    def load_imagenet_model(self):
        state_dict = model_zoo.load_url(model_urls['mobilenet_v2'], map_location='cpu')
        if self.input_channels != 3:  # convert the RGB model to others, like flow
            value = state_dict['features.0.0.weight']
            o_c, _, k_h, k_w = value.shape
            new_shape = (o_c, self.input_channels, k_h, k_w)
            state_dict['features.0.0.weight'] = value.mean(dim=1, keepdim=True).expand(
                new_shape).contiguous()
        state_dict.pop('classifier.weight', None)
        state_dict.pop('classifier.bias', None)
        self.load_state_dict(state_dict, strict=False)


class JointMobileNetV2(nn.Module):

    def __init__(self, num_frames, modality, num_classes=1000, dropout=0.5, input_channels=None):
        super().__init__()

        self.num_frames = num_frames
        self.modality = modality

        self.nets = nn.ModuleList()
        self.last_channels = []
        for i, m in enumerate(modality):
            net = MobileNetV2(num_classes, num_frames=1 if m == 'sound' else num_frames,
                              input_channels=input_channels[i])
            del net.classifier
            self.last_channels.append(net.last_channel)
            net.load_imagenet_model()
            self.nets.append(net)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        in_feature_c = sum(self.last_channels)
        out_feature_c = 2048
        self.last_channels = out_feature_c
        self.joint = nn.Sequential(
            nn.Linear(in_feature_c, out_feature_c), nn.ReLU(True),
            nn.Linear(out_feature_c, out_feature_c), nn.ReLU(True)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(out_feature_c, num_classes)

    def features(self, multi_modalities):
        # multi_modalities is a list
        bs, _, _, _ = multi_modalities[0].shape
        out = []
        for i, x in enumerate(multi_modalities):
            tmp = self.nets[i].feature_extraction(x)
            out.append(tmp)

        out = torch.cat(out, dim=1)
        out = out.view(out.size(0), -1)
        out = self.joint(out)

        return out

    def forward(self, multi_modalities):
        bs, _, _, _ = multi_modalities[0].shape
        out = self.features(multi_modalities)
        out = self.dropout(out)
        out = self.fc(out)
        n_t, c = out.shape
        out = out.view(bs, -1, c)
        # average the prediction from all frames
        out = torch.mean(out, dim=1)
        return out


class PolicyNet(nn.Module):

    def __init__(self, joint_net, modality, causality_modeling='lstm'):
        super().__init__()
        self.joint_net = joint_net
        if hasattr(self.joint_net, 'fc'):
            del self.joint_net.fc
        if hasattr(self.joint_net, 'dropout'):
            del self.joint_net.dropout
        self.modality = modality
        self.causality_modeling = causality_modeling
        self.num_modality = len(modality)
        self.temperature = 5.0
        feature_dim = self.joint_net.last_channels

        if causality_modeling is not None:
            embedded_dim = 256
            self.lstm = nn.LSTMCell(feature_dim + 2 * self.num_modality, embedded_dim)
            self.fcs = nn.ModuleList([nn.Linear(embedded_dim, 2) for _ in range(self.num_modality)])
        else:
            self.fcs = nn.ModuleList([nn.Linear(feature_dim, 2) for _ in range(self.num_modality)])

    def wrapper_gumbel_softmax(self, logits):
        """
        :param logits: NxM, N is batch size, M is number of possible choices
        :return: Nx1: the selected index
        """
        distributions = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        decisions = distributions[:, -1]
        return decisions

    def set_temperature(self, temperature):
        self.temperature = temperature

    def decay_temperature(self, decay_ratio=None):
        if decay_ratio:
            self.temperature *= decay_ratio
        print("Current temperature: {}".format(self.temperature), flush=True)

    def convert_index_to_decisions(self, decisions):
        """

        :param decisions: Nx1, the index of selection
        :return: NxM, M is the number of modality, equals to the log2(max(decisions))
        """
        out = torch.zeros((decisions.size(0), self.num_modality), dtype=decisions.dtype, device=decisions.device)
        for m_i in range(self.num_modality):
            out[:, m_i] = decisions % 2
            decisions = torch.floor(decisions / 2)
        return out

    def forward(self, x):
        """

        :param x:
        :return: all_logits shape is different when using single_fc.
                - single_fc: SxNx(2**M)
                - separate fc: MxSxNx2
        """
        #  x: M,SxNx(FC)xHxW
        num_segments = x[0].size(0)
        outs = []
        for i in range(num_segments):
            tmp_x = [x[m_i][i, ...] for m_i in range(self.num_modality)] # M,Nx(FC)xHxW
            out = self.joint_net.features(tmp_x)  # NxCout
            outs.append(out)
        outs = torch.stack(outs, dim=0)  # SxNxCout

        # TODO: check the consistency of shape of all_logits

        # SxNxCout
        if self.causality_modeling is None:
            outs = outs.view((-1, outs.size(-1)))  # (SN)xC
            logits = []
            for m_i in range(self.num_modality):
                logits.append(self.fcs[m_i](outs))  # (SN)x2
            logits = torch.cat(logits, dim=0)  # (MSN)x2
            decisions = self.wrapper_gumbel_softmax(logits)  # (MSN)x1
            # (MSN)x1
            decisions = decisions.view((self.num_modality, num_segments, -1)).transpose(0, 1)
            all_logits = logits.view((self.num_modality, num_segments, -1, 2)).transpose(0, 1)
            # SxMxN
        elif self.causality_modeling == 'lstm':
            all_logits = []
            decisions = []
            h_xs, c_xs = None, None
            for i in range(num_segments):
                if i == 0:
                    lstm_in = torch.cat((outs[i],
                                         torch.zeros((outs[i].shape[0], self.num_modality * 2),
                                                      dtype=outs[i].dtype, device=outs[i].device)
                                         ), dim=-1)
                    h_x, c_x = self.lstm(lstm_in)  # h_x: Nxhidden, c_x: Nxhidden
                else:
                    logits = logits.view((self.num_modality, -1, 2)).permute(1, 0, 2).contiguous().view(-1, 2 * self.num_modality)
                    lstm_in = torch.cat((outs[i], logits), dim=-1)

                    h_x, c_x = self.lstm(lstm_in, (h_x, c_x))  # h_x: Nxhidden, c_x: Nxhidden

                logits = []
                for m_i in range(self.num_modality):
                    tmp = self.fcs[m_i](h_x)  # Nx2
                    logits.append(tmp)
                logits = torch.cat(logits, dim=0)  # MNx2
                all_logits.append(logits.view(self.num_modality, -1, 2))
                selection = self.wrapper_gumbel_softmax(logits)  # MNx1
                decisions.append(selection)
            decisions = torch.stack(decisions, dim=0).view(num_segments, self.num_modality, -1)
            all_logits = torch.stack(all_logits, dim=0)
            # SxMxN
        else:
            raise ValueError("unknown mode")

        # dim of decision: SxMxN
        return decisions, all_logits

    @property
    def network_name(self):
        name = 'j_mobilenet_v2{}'.format('-' + self.causality_modeling
                                            if self.causality_modeling else '')
        return name


def p_joint_mobilenet(num_frames, modality, input_channels, causality_modeling):

    joint_net = JointMobileNetV2(num_frames=num_frames, modality=modality, input_channels=input_channels)
    model = PolicyNet(joint_net, modality, causality_modeling=causality_modeling)

    return model

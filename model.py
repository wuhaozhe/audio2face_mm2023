import torchaudio
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import pickle as pkl
from collections import namedtuple

class Bottleneck_Res(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(Bottleneck_Res, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool1d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv1d(in_channel, depth, 1, stride, bias=False), 
                nn.BatchNorm1d(depth))

        self.res_layer = nn.Sequential(
            nn.Conv1d(in_channel, depth, 3, 1, 1, bias=False),
            nn.BatchNorm1d(depth),
            nn.PReLU(depth),
            nn.Conv1d(depth, depth, 3, stride, 1, bias=False),
            nn.BatchNorm1d(depth),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut
    
class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_conv1d_block(in_channel, depth, num_units, stride = 2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_conv1d_blocks(num_layers):
    if num_layers == 10:
        blocks = [
            get_conv1d_block(in_channel=256, depth=512, num_units = 2),
            get_conv1d_block(in_channel=512, depth=512, num_units = 2)
        ]
    elif num_layers == 18:
        blocks = [
            get_conv1d_block(in_channel=256, depth=256, stride = 1, num_units = 2),
            get_conv1d_block(in_channel=256, depth=256, stride = 1, num_units = 2),
            get_conv1d_block(in_channel=256, depth=256, stride = 1, num_units = 2),
            get_conv1d_block(in_channel=256, depth=256, stride = 1, num_units = 2)
        ]
    return blocks

class CNNBackbone(nn.Module):
    def __init__(self, num_layers):
        super(CNNBackbone, self).__init__()

        blocks = get_conv1d_blocks(num_layers)

        unit_module = Bottleneck_Res

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel,
                        bottleneck.depth,
                        bottleneck.stride,
                    )
                )
        self.body = nn.Sequential(*modules)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.body(x)
        return x

class Audio2FaceModel(nn.Module):
    def __init__(self, audio_bundle, template_size, gumbel_softmax = False):
        super(Audio2FaceModel, self).__init__()
        feat_dim = 256
        self.audio_model = audio_bundle.get_model()
        self.template_layer = nn.Linear(template_size, feat_dim, bias = False)
        self.audio_linear_layer = nn.Sequential(
            nn.Conv1d(1024, feat_dim, 1),
            nn.BatchNorm1d(feat_dim)
        )

        self.exp_layer = CNNBackbone(num_layers = 10)
        self.exp_sigma_mu_layer = nn.Linear(512 * 3, 512)
        self.fusion_layer = CNNBackbone(num_layers = 18)
        self.gumbel_softmax = gumbel_softmax

    def freeze_pretrain(self):
        for param in self.audio_model.parameters():
            param.requires_grad = False

    def unfreeeze_pretain(self):
        for param in self.audio_model.parameters():
            param.requires_grad = True

    def regularizer(self):
        # sparsity regularization
        weight = self.template_layer.weight
        weight_l1_norm = torch.sum(torch.abs(weight))
        return weight_l1_norm

    def forward(self, audio, template, exp, target_len):
        """
        :param audio: B * T
        :param template: B * n_vertices * 3
        :param exp: B * T * n_vertices * 3
        :return: pred_geom: B * T * n_vertices * 3
        """
        n_vertices = template.shape[1]
        audio_feat, _ = self.audio_model.extract_features(audio)    # audio feat has shape of B * T * 1024
        audio_feat = torch.permute(audio_feat[-1], (0, 2, 1))
        audio_feat = self.audio_linear_layer(audio_feat)
        audio_feat = F.interpolate(audio_feat, target_len, mode = 'linear')

        template = template.view(template.shape[0], -1)
        template_feat = self.template_layer(template).unsqueeze(2).repeat(1, 1, target_len)
        
        exp = exp.view(exp.shape[0], exp.shape[1], -1)
        exp = exp - template.unsqueeze(1)
        exp_feat = self.template_layer(exp)
        exp_feat = torch.permute(exp_feat, (0, 2, 1))
        exp_feat = self.exp_layer(exp_feat)
        exp_feat_mean = torch.mean(exp_feat, dim = 2)
        exp_feat_std = torch.std(exp_feat, dim = 2)
        exp_feat_diff_std = torch.std(exp_feat[:, :, 1:] - exp_feat[:, :, :-1], dim = 2)
        exp_feat = torch.cat((exp_feat_mean, exp_feat_std, exp_feat_diff_std), dim = 1)
        exp_sigma_mu = self.exp_sigma_mu_layer(exp_feat)
        exp_sigma, exp_mu = exp_sigma_mu[:, 0: 256].unsqueeze(2), exp_sigma_mu[:, 256:].unsqueeze(2)
        
        sum_feat = audio_feat * exp_sigma + exp_mu + template_feat
        sum_feat = self.fusion_layer(sum_feat)
        sum_feat = torch.permute(sum_feat, (0, 2, 1))
        pred_geom = F.linear(sum_feat, self.template_layer.weight.t())
        pred_geom = pred_geom.view(pred_geom.shape[0], pred_geom.shape[1], n_vertices, 3)
        pred_geom = template.view(-1, n_vertices, 3).unsqueeze(1) + pred_geom * 0.1
        return pred_geom
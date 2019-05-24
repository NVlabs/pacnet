"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import math
from collections import OrderedDict
from typing import Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from pac import PacConvTranspose2d


def convert_to_single_channel(x):
    bs, ch, h, w = x.shape
    if ch != 1:
        x = x.reshape(bs * ch, 1, h, w)
    return x, ch


def recover_from_single_channel(x, ch):
    if ch != 1:
        bs_ch, _ch, h, w = x.shape
        assert _ch == 1
        assert bs_ch % ch == 0
        x = x.reshape(bs_ch // ch, ch, h, w)
    return x


def repeat_for_channel(x, ch):
    if ch != 1:
        bs, _ch, h, w = x.shape
        x = x.repeat(1, ch, 1, 1).reshape(bs * ch, _ch, h, w)
    return x


def th_rmse(pred, gt):
    return (pred - gt).pow(2).mean(dim=3).mean(dim=2).sum(dim=1).sqrt().mean()


def th_epe(pred, gt, small_flow=-1.0, unknown_flow_thresh=1e7):
    pred_u, pred_v = pred[:, 0].contiguous().view(-1), pred[:, 1].contiguous().view(-1)
    gt_u, gt_v = gt[:, 0].contiguous().view(-1), gt[:, 1].contiguous().view(-1)
    if gt_u.abs().max() > unknown_flow_thresh or gt_v.abs().max() > unknown_flow_thresh:
        idx_unknown = ((gt_u.abs() > unknown_flow_thresh) + (gt_v.abs() > unknown_flow_thresh)).nonzero()[:, 0]
        pred_u[idx_unknown] = 0
        pred_v[idx_unknown] = 0
        gt_u[idx_unknown] = 0
        gt_v[idx_unknown] = 0
    epe = ((pred_u - gt_u).pow(2) + (pred_v - gt_v).pow(2)).sqrt()
    if small_flow >= 0.0 and (gt_u.abs().min() <= small_flow or gt_v.abs().min() <= small_flow):
        idx_valid = ((gt_u.abs() > small_flow) + (gt_v.abs() > small_flow)).nonzero()[:, 0]
        epe = epe[idx_valid]
    return epe.mean()


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class JointBilateral(nn.Module):
    def __init__(self, factor, channels, kernel_size, scale_space, scale_color):
        super(JointBilateral, self).__init__()
        self.channels = channels
        self.scale_space = float(scale_space)
        self.scale_color = float(scale_color)
        self.convt = PacConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                        stride=factor, dilation=1, padding=1 + int((kernel_size - factor - 1) // 2),
                                        output_padding=(kernel_size - factor) % 2,
                                        normalize_kernel=True, bias=None)
        self.convt.weight.data.fill_(0.0)
        for c in range(channels):
            self.convt.weight.data[c, c] = 1.0

    def forward(self, target_low, guide):
        target_low, ch0 = convert_to_single_channel(target_low)
        bs, ch, h, w = guide.shape
        hh = th.arange(h, dtype=guide.dtype, device=guide.device)
        ww = th.arange(w, dtype=guide.dtype, device=guide.device)
        guide = th.cat([guide * self.scale_color,
                        hh.view(-1, 1).expand(bs, 1, -1, w) * self.scale_space,
                        ww.expand(bs, 1, h, -1) * self.scale_space], dim=1)
        guide = repeat_for_channel(guide, ch0)

        x = self.convt(target_low, guide)
        x = recover_from_single_channel(x, ch0)
        return x


class Bilinear(nn.Module):
    def __init__(self, factor, channels=None, guide_channels=None):
        super(Bilinear, self).__init__()
        self.factor = factor

    def forward(self, target_low, guide):
        return F.interpolate(target_low, scale_factor=self.factor, mode='bilinear', align_corners=False)


class DJIF(nn.Module):
    def __init__(self, factor, channels=1, guide_channels=3, fs=(9, 1, 5), ns_tg=(96, 48, 1), ns_f=(64, 32)):
        super(DJIF, self).__init__()
        assert fs[0] % 2 == 1 and fs[1] % 2 == 1 and fs[2] % 2 == 1
        paddings = tuple(f // 2 for f in fs)
        paddings_tg = sum(paddings) // 3, sum(paddings) // 3, sum(paddings) - 2 * (sum(paddings) // 3)
        self.factor = factor
        self.channels = channels
        self.guide_channels = guide_channels
        self.branch_t = nn.Sequential(
            nn.Conv2d(channels, ns_tg[0], kernel_size=fs[0], padding=paddings_tg[0]),
            nn.ReLU(),
            nn.Conv2d(ns_tg[0], ns_tg[1], kernel_size=fs[1], padding=paddings_tg[1]),
            nn.ReLU(),
            nn.Conv2d(ns_tg[1], ns_tg[2], kernel_size=fs[2], padding=paddings_tg[2])
        )
        self.branch_g = nn.Sequential(
            nn.Conv2d(guide_channels, ns_tg[0], kernel_size=fs[0], padding=paddings_tg[0]),
            nn.ReLU(),
            nn.Conv2d(ns_tg[0], ns_tg[1], kernel_size=fs[1], padding=paddings_tg[1]),
            nn.ReLU(),
            nn.Conv2d(ns_tg[1], ns_tg[2], kernel_size=fs[2], padding=paddings_tg[2])
        )
        self.branch_joint = nn.Sequential(
            nn.Conv2d(ns_tg[2] * 2, ns_f[0], kernel_size=fs[0], padding=paddings[0]),
            nn.ReLU(),
            nn.Conv2d(ns_f[0], ns_f[1], kernel_size=fs[1], padding=paddings[1]),
            nn.ReLU(),
            nn.Conv2d(ns_f[1], channels, kernel_size=fs[2], padding=paddings[2])
        )

    def forward(self, target_low, guide):
        target_low, ch0 = convert_to_single_channel(target_low)
        if target_low.shape[-1] < guide.shape[-1]:
            target_low = F.interpolate(target_low, scale_factor=self.factor, mode='bilinear', align_corners=False)
        target_low = self.branch_t(target_low)
        guide = self.branch_g(guide)
        guide = repeat_for_channel(guide, ch0)
        x = self.branch_joint(th.cat((target_low, guide), dim=1))
        x = recover_from_single_channel(x, ch0)
        return x


class DJIFWide(DJIF):
    def __init__(self, factor, channels=1, guide_channels=3):
        super(DJIFWide, self).__init__(factor, channels, guide_channels, ns_tg=(256, 128, 1), ns_f=(256, 128))


class PacJointUpsample(nn.Module):
    def __init__(self, factor, channels=1, guide_channels=3,
                 n_t_layers=3, n_g_layers=3, n_f_layers=2,
                 n_t_filters:Union[int,tuple]=32, n_g_filters:Union[int,tuple]=32, n_f_filters:Union[int,tuple]=32,
                 k_ch=16, f_sz_1=5, f_sz_2=5, t_bn=False, g_bn=False, u_bn=False, f_bn=False):
        super(PacJointUpsample, self).__init__()
        self.channels = channels
        self.guide_channels = guide_channels
        self.factor = factor
        self.branch_t = None
        self.branch_g = None
        self.branch_f = None
        self.k_ch = k_ch

        assert n_g_layers >= 1, 'Guidance branch should have at least one layer'
        assert n_f_layers >= 1, 'Final prediction branch should have at least one layer'
        assert math.log2(factor) % 1 == 0, 'factor needs to be a power of 2'
        assert f_sz_1 % 2 == 1, 'filter size needs to be an odd number'
        num_ups = int(math.log2(factor))  # number of 2x up-sampling operations
        pad = int(f_sz_1 // 2)

        if type(n_t_filters) == int:
            n_t_filters = (n_t_filters,) * n_t_layers
        else:
            assert len(n_t_filters) == n_t_layers

        if type(n_g_filters) == int:
            n_g_filters = (n_g_filters,) * (n_g_layers - 1)
        else:
            assert len(n_g_filters) == n_g_layers - 1

        if type(n_f_filters) == int:
            n_f_filters = (n_f_filters,) * (n_f_layers + num_ups - 1)
        else:
            assert len(n_f_filters) == n_f_layers + num_ups - 1

        # target branch
        t_layers = []
        n_t_channels = (channels,) + n_t_filters
        for l in range(n_t_layers):
            t_layers.append(('conv{}'.format(l + 1), nn.Conv2d(n_t_channels[l], n_t_channels[l + 1],
                                                               kernel_size=f_sz_1, padding=pad)))
            if t_bn:
                t_layers.append(('bn{}'.format(l + 1), nn.BatchNorm2d(n_t_channels[l + 1])))
            if l < n_t_layers - 1:
                t_layers.append(('relu{}'.format(l + 1), nn.ReLU()))
        self.branch_t = nn.Sequential(OrderedDict(t_layers))

        # guidance branch
        g_layers = []
        n_g_channels = (guide_channels,) + n_g_filters + (k_ch * num_ups,)
        for l in range(n_g_layers):
            g_layers.append(('conv{}'.format(l + 1), nn.Conv2d(n_g_channels[l], n_g_channels[l + 1],
                                                               kernel_size=f_sz_1, padding=pad)))
            if g_bn:
                g_layers.append(('bn{}'.format(l + 1), nn.BatchNorm2d(n_g_channels[l + 1])))
            if l < n_g_layers - 1:
                g_layers.append(('relu{}'.format(l + 1), nn.ReLU()))
        self.branch_g = nn.Sequential(OrderedDict(g_layers))

        # upsampling layers
        p, op = int((f_sz_2 - 1) // 2), (f_sz_2 % 2)
        self.up_convts = nn.ModuleList()
        self.up_bns = nn.ModuleList()
        n_f_channels = (n_t_channels[-1],) + n_f_filters + (channels,)
        for l in range(num_ups):
            self.up_convts.append(PacConvTranspose2d(n_f_channels[l], n_f_channels[l + 1],
                                                     kernel_size=f_sz_2, stride=2, padding=p, output_padding=op))
            if u_bn:
                self.up_bns.append(nn.BatchNorm2d(n_f_channels[l + 1]))

        # final prediction branch
        f_layers = []
        for l in range(n_f_layers):
            f_layers.append(('conv{}'.format(l + 1), nn.Conv2d(n_f_channels[l + num_ups], n_f_channels[l + num_ups + 1],
                                                               kernel_size=f_sz_1, padding=pad)))
            if f_bn:
                f_layers.append(('bn{}'.format(l + 1), nn.BatchNorm2d(n_f_channels[l + num_ups + 1])))
            if l < n_f_layers - 1:
                f_layers.append(('relu{}'.format(l + 1), nn.ReLU()))
        self.branch_f = nn.Sequential(OrderedDict(f_layers))

    def forward(self, target_low, guide):
        target_low, ch0 = convert_to_single_channel(target_low)
        x = self.branch_t(target_low)
        guide = self.branch_g(guide)
        for i in range(len(self.up_convts)):
            scale = math.pow(2, i+1) / self.factor
            guide_cur = guide[:, (i*self.k_ch):((i+1)*self.k_ch)]
            if scale != 1:
                guide_cur = F.interpolate(guide_cur, scale_factor=scale, align_corners=False, mode='bilinear')
            guide_cur = repeat_for_channel(guide_cur, ch0)
            x = self.up_convts[i](x, guide_cur)
            if self.up_bns:
                x = self.up_bns[i](x)
            x = F.relu(x)
        x = self.branch_f(x)
        x = recover_from_single_channel(x, ch0)
        return x


class PacJointUpsampleLite(PacJointUpsample):
    def __init__(self, factor, channels=1, guide_channels=3):
        if factor == 4:
            args = dict(n_g_filters=(12, 22), n_t_filters=(12, 16, 22), n_f_filters=(12, 16, 22), k_ch=12)
        elif factor == 8:
            args = dict(n_g_filters=(12, 16), n_t_filters=(12, 16, 16), n_f_filters=(12, 16, 16, 20), k_ch=12)
        elif factor == 16:
            args = dict(n_g_filters=(8, 16), n_t_filters=(8, 16, 16), n_f_filters=(8, 16, 16, 16, 16), k_ch=10)
        else:
            raise ValueError('factor can only be 4, 8, or 16.')
        super(PacJointUpsampleLite, self).__init__(factor, channels, guide_channels, **args)

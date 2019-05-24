"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch as th
from torch import nn

from . import fcn8s
import paccrf


def create_model(model_name, num_classes):
    # backbone models
    if '_' not in model_name:
        if model_name == 'fcn8s':
            return FCN8s(num_classes)
        elif model_name == 'fcn8saltpad':
            return FCN8sAltPad(num_classes)
        elif model_name == 'fcn8spac':
            return FCN8sPac(num_classes)

    backbone, crf_str = model_name.split('_')
    if backbone.endswith('frozen'):
        backbone = create_model(backbone[:-6], num_classes)
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        backbone = create_model(backbone, num_classes)

    if crf_str in {'convcrf', 'convcrfv1'}:
        return ModelWithConvCrf(backbone, compat='2d')
    else:
        # note that 'p', 'i', 'l' are reserved letters in model names
        if 'p' not in crf_str:
            crf_str += 'p4d5161'
        crf_type, crf_edges = crf_str[:crf_str.find('p')], crf_str[crf_str.find('p'):]
        if 'i' not in crf_type:
            loose = False
            num_steps = 5
        else:
            loose = crf_type.endswith('l')
            num_steps = int(crf_type[(crf_type.find('i') + 1):(-1 if loose else None)])
            crf_type = crf_type[:crf_type.find('i')]
        assert crf_type in {'crf', 'crfv7'}
        return ModelWithCrf(backbone,
                            num_steps=num_steps,
                            pairwise=crf_edges,
                            loose=loose,
                            use_yx=False,
                            shared_scales=False)


def kernel_transform_init(diag_values, size=1, nch=None, perturb_range=0.001):
    nch0 = len(diag_values)
    if nch is None:
        nch = nch0
    init = th.zeros(nch, nch0, size, size, dtype=th.float32)
    ctr = size // 2
    init[:nch0, :, ctr, ctr] = th.diag(th.tensor(diag_values))
    init.add_((th.rand_like(init) - 0.5) * perturb_range)
    return init


class FCN8s(fcn8s.FCN8s):
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__(num_classes)
        self.num_classes = num_classes

    def forward(self, x, out_crop=None):
        out = super(FCN8s, self).forward(x)
        in_h, in_w = x.shape[2:]
        assert out.shape[2] == in_h and out.shape[3] == in_w
        if out_crop is not None and (out_crop[0] != in_h or out_crop[1] != in_w):
            out = out[:, :, :out_crop[0], :out_crop[1]]
        return out


class FCN8sAltPad(fcn8s.FCN8sAltPad):
    def __init__(self, num_classes=21):
        super(FCN8sAltPad, self).__init__(num_classes)
        self.num_classes = num_classes

    def forward(self, x, out_crop=None):
        out = super(FCN8sAltPad, self).forward(x)
        in_h, in_w = x.shape[2:]
        assert out.shape[2] == in_h and out.shape[3] == in_w
        if out_crop is not None and (out_crop[0] != in_h or out_crop[1] != in_w):
            out = out[:, :, :out_crop[0], :out_crop[1]]
        return out


class FCN8sPac(fcn8s.FCN8sPac):
    def __init__(self, num_classes=21):
        super(FCN8sPac, self).__init__(num_classes)
        self.num_classes = num_classes

    def forward(self, x, out_crop=None):
        out = super(FCN8sPac, self).forward(x)
        in_h, in_w = x.shape[2:]
        assert out.shape[2] == in_h and out.shape[3] == in_w
        if out_crop is not None and (out_crop[0] != in_h or out_crop[1] != in_w):
            out = out[:, :, :out_crop[0], :out_crop[1]]
        return out


class ModelWithCrf(nn.Module):
    def __init__(self, backbone, num_steps=5, pairwise=('4d_5_16_1',), loose=False,
                 use_yx=False, shared_scales=True, adaptive_init=True):
        super(ModelWithCrf, self).__init__()
        self.backbone = backbone
        self.num_classes = self.backbone.num_classes
        self.use_yx = use_yx
        self.shared_scales = shared_scales
        if isinstance(pairwise, str):
            pw_strs = []
            for s in pairwise.split('p')[1:]:
                l_ = 3 if s[2] == 's' else 2
                pw_strs.append('_'.join((s[:l_], s[l_], s[(l_ + 1):-1], s[-1])))
        else:
            pw_strs = pairwise
        crf_params = dict(num_steps=num_steps,
                          perturbed_init=True,
                          fixed_weighting=False,
                          unary_weight=1.0,
                          pairwise_kernels=[])
        for pw_str in pw_strs:
            t_, k_, d_, b_ = pw_str.split('_')
            pairwise_param = dict(kernel_size=int(k_),
                                  dilation=int(d_),
                                  blur=int(b_),
                                  compat_type=('potts' if t_.startswith('0d') else t_[:2]),
                                  spatial_filter=t_.endswith('s'),
                                  pairwise_weight=1.0)
            crf_params['pairwise_kernels'].append(pairwise_param)

        CRF = paccrf.PacCRFLoose if loose else paccrf.PacCRF
        self.crf = CRF(self.num_classes, **crf_params)
        self.feat_scales = nn.ParameterList()
        for s in pw_strs:
            fs, dilation = float(s.split('_')[1]), float(s.split('_')[2])
            p_sc = (((fs - 1) * dilation + 1) / 4.0) if adaptive_init else 100.0
            c_sc = 30.0
            if use_yx:
                scales = th.tensor([p_sc, c_sc] if shared_scales else ([p_sc] * 2 + [c_sc] * 3), dtype=th.float32)
            else:
                scales = th.tensor(c_sc if shared_scales else [c_sc] * 3, dtype=th.float32)
            self.feat_scales.append(nn.Parameter(scales))

    def forward(self, x, out_crop=None):
        unary = self.backbone(x, out_crop)
        in_h, in_w = x.shape[2:]
        if out_crop is not None and (out_crop[0] != in_h or out_crop[1] != in_w):
            x = x[:, :, :out_crop[0], :out_crop[1]]
        if self.use_yx:
            if self.shared_scales:
                edge_feat = [paccrf.create_YXRGB(x, yx_scale=sc[0], rgb_scale=sc[1]) for sc in self.feat_scales]
            else:
                edge_feat = [paccrf.create_YXRGB(x, scales=sc) for sc in self.feat_scales]
        else:
            edge_feat = [x * (1.0 / rgb_scale.view(-1, 1, 1)) for rgb_scale in self.feat_scales]
        out = self.crf(unary, edge_feat)

        return out

    def load_state_dict(self, state_dict, strict=True):
        if isinstance(self.crf, paccrf.PacCRFLoose) and not strict:
            if not any(s.startswith('crf.steps.') for s in state_dict.keys()):
                for p in list(filter(lambda s: s.startswith('crf.'), state_dict.keys())):
                    for step in range(self.crf.num_steps):
                        state_dict['crf.steps.{}.{}'.format(step, p[4:])] = state_dict[p]

        super(ModelWithCrf, self).load_state_dict(state_dict, strict=strict)


class ModelWithConvCrf(nn.Module):
    def __init__(self, backbone, compat='potts', kernel_size=11, blur=4, dilation=1):
        super(ModelWithConvCrf, self).__init__()
        self.backbone = backbone
        self.num_classes = self.backbone.num_classes
        crf_params = dict(num_steps=5, perturbed_init=True, fixed_weighting=False, unary_weight=0.8, pairwise_kernels=[
            dict(kernel_size=kernel_size, dilation=dilation, blur=blur, compat_type=compat, spatial_filter=False,
                 pairwise_weight=2.0),
            dict(kernel_size=kernel_size, dilation=dilation, blur=blur, compat_type=compat, spatial_filter=False,
                 pairwise_weight=0.6)
        ])
        self.crf = paccrf.PacCRF(self.num_classes, **crf_params)

    def forward(self, x, out_crop=None):
        unary = self.backbone(x, out_crop)
        in_h, in_w = x.shape[2:]
        if out_crop is not None and (out_crop[0] != in_h or out_crop[1] != in_w):
            x = x[:, :, :out_crop[0], :out_crop[1]]
        edge_feat = [paccrf.create_YXRGB(x, 80.0, 13.0),
                     paccrf.create_position_feats(x.shape[2:], 3.0, bs=x.shape[0], device=x.device)]
        out = self.crf(unary, edge_feat)

        return out

"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import random
import collections
import tarfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_url

from . import prepare_sbd


mean_rgb = [122.675, 116.669, 104.008]
sbd_url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'
sbd_md5 = '82b4d87ceb2ed10f6038a1cba92111cb'
voc12seg_url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
voc12seg_md5 = '6cd6e144f989b92b3379bac3b3de84fd'
voc12segtest_url = ''  # No publicly available url for the test split
voc12segtest_md5 = '9065beb292b6c291fad82b2725749fda'
meta_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'meta')
voc11splits = {'train11': os.path.join(meta_dir, 'voc11seg/train.txt'),
               'val11': os.path.join(meta_dir, 'voc11seg/val.txt'),
               'trainval11': os.path.join(meta_dir, 'voc11seg/trainval.txt'),
               'val11_sbd': os.path.join(meta_dir, 'voc11seg/seg11valid.txt')
              }

pascal_voc_seg_palette = [255] * (256 * 3)
pascal_voc_seg_palette[:(21 * 3)] = [0, 0, 0,
                                     128, 0, 0,
                                     0, 128, 0,
                                     128, 128, 0,
                                     0, 0, 128,
                                     128, 0, 128,
                                     0, 128, 128,
                                     128, 128, 128,
                                     64, 0, 0,
                                     192, 0, 0,
                                     64, 128, 0,
                                     192, 128, 0,
                                     64, 0, 128,
                                     192, 0, 128,
                                     64, 128, 128,
                                     192, 128, 128,
                                     0, 64, 0,
                                     128, 64, 0,
                                     0, 192, 0,
                                     128, 192, 0,
                                     0, 64, 128]


def download_extract_tar(url, md5, root):
    os.makedirs(root, exist_ok=True)
    download_url(url, root, os.path.basename(url), md5)
    with tarfile.open(os.path.join(root, os.path.basename(url))) as t:
        t.extractall(root)


class DeepLabInputs(object):
    def __init__(self, size=None, use_bgr=True, pad_label=False, ignore_label=255, seed=None,
                 aug_flip=False, aug_scale=None, aug_color=None):
        self.size = size
        self.use_bgr = use_bgr
        self.aug_flip = aug_flip
        self.aug_scale = aug_scale
        self.aug_color = aug_color
        self.pad_label = pad_label
        self.ignore_label = ignore_label
        assert self.aug_scale is None or len(self.aug_scale) == 2  # (scale_low, scale_high)
        assert self.aug_color is None or len(self.aug_color) == 4  # (brightness, contrast, saturation, hue)
        random.seed(seed)

    def __call__(self, img, lbl, img_id):
        if self.aug_flip and random.random() < 0.5:
            img, lbl = img.transpose(Image.FLIP_LEFT_RIGHT), lbl.transpose(Image.FLIP_LEFT_RIGHT)
        if self.aug_scale:
            sc = self.aug_scale[0] + random.random() * (self.aug_scale[1] - self.aug_scale[0])
            hw = int(sc * img.size[1]), int(sc * img.size[0])
            img, lbl = transforms.Resize(hw)(img), transforms.Resize(hw)(lbl)
        if self.aug_color:
            img = transforms.ColorJitter(*self.aug_color)(img)

        img = np.transpose(np.asarray(img) - mean_rgb, (2, 0, 1)).astype(np.float32)
        if self.use_bgr:
            img = img[::-1, :, :]
        lbl = np.asarray(lbl, dtype=np.int64)
        in_h, in_w = img.shape[1:]

        if self.size is not None and self.size > 0:
            valid_h, valid_w = min(self.size, in_h), min(self.size, in_w)
            idx_h, idx_w = random.randint(0, max(0, in_h - self.size)), random.randint(0, max(0, in_w - self.size))

            img_tmp = np.zeros((3, self.size, self.size), dtype=np.float32)
            img_tmp[:, :valid_h, :valid_w] = img[:, idx_h:(idx_h + valid_h), idx_w:(idx_w + valid_w)]
            img = img_tmp
            if self.pad_label:
                label_tmp = np.ones((self.size, self.size), dtype=np.int64) * self.ignore_label
                label_tmp[:valid_h, :valid_w] = lbl[idx_h:(idx_h + valid_h), idx_w:(idx_w + valid_w)]
                lbl = label_tmp
            else:
                lbl = lbl[idx_h:(idx_h + valid_h), idx_w:(idx_w + valid_w)]
        elif self.use_bgr:
            img = img.copy()  # as of pytorch 0.4, negative-stride numpy array cannot be converted to pytorch tensor

        img = torch.from_numpy(img)
        lbl = torch.from_numpy(lbl)

        return img, lbl


class DeepLabEvalInputs(object):
    def __init__(self, size=None, use_bgr=True):
        self.size = size
        self.use_bgr = use_bgr

    def __call__(self, img, lbl, img_id):
        img = np.transpose(np.asarray(img) - mean_rgb, (2, 0, 1)).astype(np.float32)
        if self.use_bgr:
            img = img[::-1, :, :]
        in_h, in_w = img.shape[1:]

        if self.size is not None and self.size > 0:
            valid_h, valid_w = min(self.size, in_h), min(self.size, in_w)
            idx_h, idx_w = random.randint(0, max(0, in_h - self.size)), random.randint(0, max(0, in_w - self.size))

            img_tmp = np.zeros((3, self.size, self.size), dtype=np.float32)
            img_tmp[:, :valid_h, :valid_w] = img[:, idx_h:(idx_h + valid_h), idx_w:(idx_w + valid_w)]
            img = img_tmp
        elif self.use_bgr:
            img = img.copy()  # as of pytorch 0.4, negative-stride numpy array cannot be converted to pytorch tensor

        img = torch.from_numpy(img)
        img_id = torch.tensor([ord(s) for s in img_id])
        img_hw = torch.tensor([in_h, in_w])

        return img, img_id, img_hw


class PascalVOC(Dataset):
    def __init__(self, root, sbd_root='', split='train', transform=None, download=False):
        """
        splits:
        - train, val, trainval, test (1464, 1449, 2913, 1456)
        - train11, val11, trainval11 (1112, 1111, 2223)
        - train_aug: voc12_train + sbd_trainval - voc12_val (10582)
        - trainval_aug: voc12_trainval + sbd_trainval (12031)
        - train11_sbd: voc11_train + sbd_train (8825)
        - val11_sbd: voc11_val - sbd_train (736)
        """
        self.root = os.path.expanduser(root).rstrip(os.path.sep)
        self.sbd_root = os.path.expanduser(sbd_root).rstrip(os.path.sep) if sbd_root else ''
        self.split = split
        self.transform = transform
        self.files = collections.defaultdict(list)
        if '_rand' in split:  # e.g. 'train_aug_rand1500' would sample 1500 images from 'train_aug'
            split, subset = split.split('_rand')
            subset = int(subset) if subset else 0
        else:
            subset = -1
        if not os.path.exists(self.root) and download:
            assert os.path.basename(self.root) == 'VOCdevkit'
            download_extract_tar(voc12seg_url, voc12seg_md5, os.path.dirname(self.root))
        if not os.path.exists(self.sbd_root) and download and split in ('train_aug', 'trainval_aug', 'train11_sbd'):
            assert os.path.basename(self.sbd_root) == 'benchmark_RELEASE'
            download_extract_tar(sbd_url, sbd_md5, os.path.dirname(self.sbd_root))
            print('Converting SBD annotations')
            prepare_sbd.sbd_annotation_to_png(self.sbd_root)

        assert os.path.exists(self.root), 'Dataset not found, consider setting download=True.'
        for s in ['train', 'val', 'trainval']:
            file_list = tuple(open(os.path.join(self.root, 'VOC2012', 'ImageSets', 'Segmentation', s + '.txt'), 'r'))
            file_list = sorted([id_.rstrip() for id_ in file_list])
            self.files[s] = file_list
        if split == 'test':
            assert os.path.exists(os.path.join(self.root, 'VOC2012', 'ImageSets', 'Segmentation', 'test.txt')), \
            'test split needs to be downloaded manually'
            file_list = tuple(open(os.path.join(self.root, 'VOC2012', 'ImageSets', 'Segmentation', 'test.txt'), 'r'))
            file_list = sorted([id_.rstrip() for id_ in file_list])
            self.files['test'] = file_list
        for s in ['train11', 'val11', 'trainval11', 'val11_sbd']:
            file_list = tuple(open(voc11splits[s], 'r'))
            file_list = sorted([id_.rstrip() for id_ in file_list])
            self.files[s] = file_list
        if split in ('train_aug', 'trainval_aug', 'train11_sbd'):
            assert sbd_root, 'SBD is required for augmented data splits.'
            assert os.path.exists(self.sbd_root), 'Dataset not found, consider setting download=True.'
            sbd_train_list = tuple(open(os.path.join(self.sbd_root, 'dataset', 'train.txt'), 'r'))
            sbd_val_list = tuple(open(os.path.join(self.sbd_root, 'dataset', 'val.txt'), 'r'))
            sbd_train_list = [id_.rstrip() for id_ in sbd_train_list]
            sbd_val_list = [id_.rstrip() for id_ in sbd_val_list]
            sbd_list = list(set(sbd_train_list + sbd_val_list))
            self.files['train_aug'] = sorted(list(set(self.files['train'] + sbd_list) - set(self.files['val'])))
            self.files['trainval_aug'] = sorted(self.files['train_aug'] + self.files['val'])
            self.files['train11_sbd'] = sorted(set(self.files['train11'] + sbd_train_list))
        if subset >= 0:
            self.files[self.split] = random.sample(self.files[split], subset if subset >0 else len(self.files[split]))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_id = self.files[self.split][index]
        img_path = os.path.join(self.root, 'VOC2012', 'JPEGImages', img_id + '.jpg')
        lbl_path = os.path.join(self.root, 'VOC2012', 'SegmentationClass', img_id + '.png')
        if not os.path.exists(lbl_path) and not 'test' in self.split:
            img_path = os.path.join(self.sbd_root, 'dataset', 'img', img_id + '.jpg')
            lbl_path = os.path.join(self.sbd_root, 'dataset', 'cls_png', img_id + '.png')
            assert os.path.exists(lbl_path), 'SBD is not downloaded or pre-processed'
        im = Image.open(img_path)
        lbl = Image.open(lbl_path) if os.path.exists(lbl_path) else None
        if self.transform is not None:
            outputs = self.transform(im, lbl, img_id)
        else:
            outputs = im, lbl, img_id
        return outputs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='root data directory')
    parser.add_argument('--voc12seg', action='store_true', help='download voc12seg')
    parser.add_argument('--sbd', action='store_true', help='download sbd')
    parser.add_argument('--all', action='store_true', help='download all')
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    os.makedirs(root, exist_ok=True)
    if args.voc12seg or args.all:
        download_extract_tar(voc12seg_url, voc12seg_md5, root)
        download_extract_tar(voc12segtest_url, voc12segtest_md5, root)
    if args.sbd or args.all:
        download_extract_tar(sbd_url, sbd_md5, root)
        print('Converting SBD annotations')
        prepare_sbd.sbd_annotation_to_png(os.path.join(root, 'benchmark_RELEASE'))

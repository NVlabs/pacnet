"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import argparse
import tqdm
from PIL import Image
import numpy as np
import scipy.io

pascal_voc_seg_palette = [255] * (256 * 3)
pascal_voc_seg_palette[:(21 * 3)] = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
                                     128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0,
                                     128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0,
                                     64, 128]


def sbd_annotation_to_png(sbd_dir, target_dir=None):
    sbd_dir = os.path.expanduser(sbd_dir)
    if target_dir is None:
        target_dir = os.path.join(sbd_dir, 'dataset', 'cls_png')
    else:
        target_dir = os.path.expanduser(target_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    sbd_train_list = tuple(open(os.path.join(sbd_dir, 'dataset', 'train.txt'), 'r'))
    sbd_val_list = tuple(open(os.path.join(sbd_dir, 'dataset', 'val.txt'), 'r'))
    sbd_train_list = [id_.rstrip() for id_ in sbd_train_list]
    sbd_val_list = [id_.rstrip() for id_ in sbd_val_list]
    sbd_trainval_list = sorted(list(set(sbd_train_list + sbd_val_list)))

    for ii in tqdm.tqdm(sbd_trainval_list):
        lbl_path = os.path.join(sbd_dir, 'dataset', 'cls', ii + '.mat')
        data = scipy.io.loadmat(lbl_path)
        lbl = data['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        img = Image.fromarray(lbl, 'P')
        img.putpalette(pascal_voc_seg_palette)
        img.save(os.path.join(target_dir, ii + '.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sbd_dir', help='Path to SBD directory')
    parser.add_argument('--out', default=None, help='Output directory')
    args = parser.parse_args()
    sbd_annotation_to_png(args.sbd_dir, args.out)

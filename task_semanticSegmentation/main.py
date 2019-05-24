"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse
import os
import time
import math
import random
import glob
from PIL import Image
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from . import datasets, models


def fast_conf(pred, gt, nclasses):
    if pred.ndim > 1:
        pred = pred.flatten()
    if gt.ndim > 1:
        gt = gt.flatten()
    k = (gt >= 0) & (gt < nclasses)
    return np.bincount(nclasses * gt[k] + pred[k], minlength=nclasses ** 2).reshape(nclasses, nclasses)


def seg_measures(conf, measures=('miou', 'acc', 'macc')):
    if isinstance(measures, str):
        return seg_measures(conf, (measures,))[0]
    scores = []
    for m in measures:
        if m == 'miou':
            iou = np.diag(conf) / (conf.sum(1) + conf.sum(0) - np.diag(conf))
            scores.append(float(iou.mean()))
        elif m == 'acc':
            scores.append(float(np.diag(conf).sum() / conf.sum()))
        elif m == 'macc':
            cacc = np.diag(conf) / conf.sum(1)
            scores.append(float(cacc.mean()))
    return scores


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConfMeter(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.val = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.sum = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, gt):
        val = fast_conf(pred, gt, self.num_classes)
        self.val = val
        self.sum += val

    def score_val(self, score_type):
        return seg_measures(self.val, score_type)

    def score_avg(self, score_type):
        return seg_measures(self.sum, score_type)


def _compute_loss(output, target, loss_type, ignore_index=-100):
    if loss_type == 'ce':
        loss = F.cross_entropy(output, target, ignore_index=ignore_index)
    elif loss_type.startswith('fce'):
        forgiving_threshold = float(loss_type.split('_')[-1])
        loss_none = F.cross_entropy(output, target, ignore_index=ignore_index, reduction='none')
        loss_none[loss_none < - math.log(forgiving_threshold)] = 0.0
        loss = loss_none.mean()
    else:
        raise ValueError('loss type {} not supported'.format(loss_type))
    return loss


def train(model, train_loader, optimizer, device, epoch, lr, perf_measures, num_classes, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    conf = ConfMeter(num_classes)
    model.train()
    log = []
    end = time.time()
    if callable(train_loader):
        train_loader = train_loader()
    for batch_idx, sample in enumerate(train_loader):
        img, target = sample[0].to(device), sample[1].to(device)
        data_time.update((time.time() - end) / img.shape[0], img.shape[0])
        optimizer.zero_grad()
        output = model(img, target.shape[-2:])
        loss = _compute_loss(output, target, args.loss, ignore_index=255)
        loss.backward()
        optimizer.step()
        pred, gt = output.argmax(dim=1).cpu().numpy(), sample[1].numpy()
        batch_time.update((time.time() - end) / img.shape[0], img.shape[0])
        losses.update(loss.item(), img.shape[0])
        conf.update(pred, gt)
        end = time.time()
        batch_cnt = batch_idx + 1
        sample_cnt = batch_idx * args.batch_size + len(img)
        progress = sample_cnt / len(train_loader.dataset)
        if batch_cnt == len(train_loader) or batch_cnt % args.log_interval == 0:
            log_row = [progress + epoch - 1, lr, batch_time.avg, data_time.avg, losses.val, losses.avg]
            for m in perf_measures:
                log_row.extend([conf.score_val(m), conf.score_avg(m)])
            log.append(log_row)
        if batch_cnt == len(train_loader) or batch_cnt % args.print_interval == 0:
            msg = 'Train Epoch {} [{}/{} ({:3.0f}%)]\tLR {:g}\tTime {:.3f}\tData {:.3f}\t' \
                  'Loss {:.4f} ({:.4f})'.format(epoch, sample_cnt, len(train_loader.dataset), 100. * progress, lr,
                                                batch_time.avg, data_time.avg, losses.val, losses.avg)
            print(msg)
    msg = '\nTraining (#epochs={})\n'.format(epoch)
    msg += 'Average loss: {:.6f}\n'.format(losses.avg)
    msg += 'Average speed: {:.2f} samples/sec\n'.format(1.0 / batch_time.avg)
    msg += ''.join('{}: {:.6f}\n'.format(m, conf.score_avg(m)) for m in perf_measures)
    print(msg)
    return log


def test(model, test_loader, device, epoch, lr, perf_measures, num_classes, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    conf = ConfMeter(num_classes)
    if isinstance(model, torch.nn.DataParallel) and args.test_batch_size == 1:
        model = model.module
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, sample in enumerate(test_loader):
            img, target = sample[0].to(device), sample[1].to(device)
            data_time.update((time.time() - end) / img.shape[0], img.shape[0])
            output = model(img, target.shape[-2:])
            loss = _compute_loss(output, target, args.loss, ignore_index=255)
            pred, gt = output.argmax(dim=1).cpu().numpy(), sample[1].numpy()

            batch_time.update((time.time() - end) / img.shape[0], img.shape[0])
            losses.update(loss.item(), img.shape[0])
            conf.update(pred, gt)
            end = time.time()

    log = [float(epoch), lr, batch_time.avg, data_time.avg, losses.avg] + conf.score_avg(perf_measures)
    msg = '\nTesting (#epochs={})\n'.format(epoch)
    msg += 'Average loss: {:.6f}\n'.format(losses.avg)
    msg += 'Average speed: {:.2f} samples/sec ({}+{}ms /sample)\n'.format(1.0 / batch_time.avg,
                                                                          int(1000 * (batch_time.avg - data_time.avg)),
                                                                          int(1000 * data_time.avg))
    msg += ''.join('{}: {:.6f}\n'.format(m, conf.score_avg(m)) for m in perf_measures)
    print(msg)
    return [log]


def evaluate(model, test_loader, device, batch_size, out_type, save_dir, palette):
    os.makedirs(save_dir, exist_ok=True)
    batch_time = AverageMeter()
    if isinstance(model, torch.nn.DataParallel) and batch_size == 1:
        model = model.module
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, sample in enumerate(test_loader):
            imgs = sample[0].to(device)
            img_ids = sample[1]
            img_hw = sample[2]
            outputs = model(imgs)

            for out, img_id, (im_h, im_w) in zip(outputs, img_ids, img_hw):
                if im_h <= out.shape[1] and im_w <= out.shape[2]:
                    out = out[:, :im_h, :im_w]
                img_id = ''.join([chr(s) for s in img_id])
                if out_type == 'raw':
                    np.save(os.path.join(save_dir, img_id + '.npy'),
                            out.cpu().numpy().astype(np.float32))
                elif out_type == 'pred':
                    pred = out.argmax(dim=0).cpu().numpy().astype(np.uint8)
                    pred = Image.fromarray(pred)
                    pred.putpalette(palette)
                    pred.save(os.path.join(save_dir, img_id + '.png'))

            batch_time.update((time.time() - end) / imgs.shape[0], imgs.shape[0])
            end = time.time()

    msg = '\nEvaluation\n'
    msg += 'Average speed: {:.2f} samples/sec\n'.format(1.0 / batch_time.avg)
    print(msg)


def prepare_log(log_path, header, last_epoch=0):
    # keep all existing log lines up to epoch==last_epoch (included)
    try:
        log = np.genfromtxt(log_path, delimiter=',', skip_header=1, usecols=(0,))
    except:
        log = []
    if log != [] and log.size > 0:
        idxs = np.where(log <= last_epoch)[0]
        if len(idxs) > 0:
            lines_to_keep = max(idxs) + 2
            with open(log_path) as f:
                lines = f.readlines()
            with open(log_path, 'w') as f:

                f.writelines(lines[:lines_to_keep])
            return

    with open(log_path, 'w') as f:
        f.write(header + '\n')


def main():
    parser = argparse.ArgumentParser(description='Semantic segmentation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root', type=str, default='data', metavar='D',
                        help='place to find (or download) data')
    parser.add_argument('--exp-root', type=str, default='exp', metavar='E',
                        help='place to save results')
    parser.add_argument('--download', default=False, action='store_true',
                        help='download dataset if not found locally')
    parser.add_argument('--load-weights', type=str, default='', metavar='L',
                        help='file with pre-trained weights')
    parser.add_argument('--load-weights-backbone', type=str, default='', metavar='L',
                        help='file with pre-trained weights for backbone network')
    parser.add_argument('--model', type=str, default='fcn8s', metavar='M',
                        help='network model type')
    parser.add_argument('--dataset', type=str, default='VOC2012', metavar='D',
                        help='dataset')
    parser.add_argument('--num-data-worker', type=int, default=4, metavar='W',
                        help='number of subprocesses for data loading')
    parser.add_argument('--gpu', type=int, default=None, metavar='GPU',
                        help='GPU id to use')
    parser.add_argument('--val-ratio', type=float, default=0.0, metavar='V',
                        help='use this portion of training set for validation')
    parser.add_argument('--train-split', type=str, default='', metavar='TRAIN',
                        help='specify a subset for training')
    parser.add_argument('--test-split', type=str, default='', metavar='TEST',
                        help='specify a subset for testing')
    parser.add_argument('--eval', type=str, default='', metavar='EVAL', choices=('', 'pred', 'raw'),
                        help='do evaluation instead of train/test')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--train-crop', type=int, default=449, metavar='CROP',
                        help='input crop size during training')
    parser.add_argument('--test-crop', type=int, default=513, metavar='CROP',
                        help='input crop size during testing')
    parser.add_argument('--train-aug-scale', type=float, default=0.0,
                        help='random scaling as data augmentation')
    parser.add_argument('--train-aug-color', default=False, action='store_true',
                        help='random color jittering as data augmentation')
    parser.add_argument('--epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='Adam', metavar='O',
                        help='pick which optimizer to use')
    parser.add_argument('--loss', type=str, default='ce', metavar='L',
                        help='pick which loss function to use')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr-steps', nargs='+', default=None, metavar='S',
                        help='decrease lr by 10 at these epochs')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='WD',
                        help='Adam/SGD weight decay')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--seed', type=int, default=-1, metavar='S',
                        help='random seed')
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='ignore existing log files and snapshots')
    parser.add_argument('--print-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before displaying training status')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test-interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before a testing')
    parser.add_argument('--snapshot-interval', type=int, default=1, metavar='N',
                        help='snapshot intermediate models')
    args = parser.parse_args()

    seed_value = ord(os.urandom(1)) if args.seed == -1 else args.seed
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    assert(torch.cuda.is_available())
    if args.gpu is not None:
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cuda:0")
    dl_kwargs = {'num_workers': args.num_data_worker, 'pin_memory': True}

    # find existing snapshots
    os.makedirs(args.exp_root, exist_ok=True)
    snapshots_found = sorted([int(s.split('_')[-1].rstrip('.pth'))
                              for s in glob.glob(os.path.join(args.exp_root, 'weights_epoch_*.pth'))])
    load_weights = args.load_weights
    if snapshots_found and not args.overwrite:
        last_epoch = max(snapshots_found) if args.epochs > max(snapshots_found) else args.epochs
        assert last_epoch in snapshots_found
        if load_weights:
            print('Warning: parameter (load_weights={}) ignored!'.format(load_weights))
        load_weights = os.path.join(args.exp_root, 'weights_epoch_{}.pth'.format(last_epoch))
    else:
        last_epoch = 0
    test_only = (args.epochs <= last_epoch)
    if args.eval:
        assert test_only

    # dataset
    if args.dataset == 'VOC2012':
        perf_measures = ('miou','acc','macc')
        num_classes = 21
        palette = datasets.pascal_voc_seg_palette
        train_split = 'train_aug' if not args.train_split else args.train_split
        test_split = 'val' if not args.test_split else args.test_split
        aug_flip = True
        aug_scale = (1.0 - args.train_aug_scale, 1.0 + args.train_aug_scale) if args.train_aug_scale > 0 else None
        aug_color = (0.1, 0.1, 0.1, 0.01) if args.train_aug_color else None
        train_transform = datasets.DeepLabInputs(args.train_crop, pad_label=True,
                                                 aug_flip=aug_flip, aug_scale=aug_scale, aug_color=aug_color)
        if args.eval:
            test_transform = datasets.DeepLabEvalInputs(args.test_crop)
        else:
            test_transform = datasets.DeepLabInputs(args.test_crop, aug_flip=False, pad_label=(args.test_batch_size > 1))
        if ',' in args.data_root:
            voc_root, sbd_root = args.data_root.split(',')
        else:
            voc_root = os.path.join(args.data_root, 'VOCdevkit')
            sbd_root = os.path.join(args.data_root, 'benchmark_RELEASE')
        if args.epochs > 0:
            if '_rand' in train_split:
                train_dset = lambda : datasets.PascalVOC(voc_root, sbd_root, split=train_split,
                                                         transform=train_transform, download=args.download)
            else:
                train_dset = datasets.PascalVOC(voc_root, sbd_root, split=train_split, transform=train_transform,
                                                download=args.download)
        else:
            train_dset = None
        test_dset = datasets.PascalVOC(voc_root, sbd_root, split=test_split, transform=test_transform,
                                       download=args.download)
    else:
        raise ValueError('Dataset ({}) not supported.'.format(args.dataset))

    # data loader
    if test_only:
        train_loader = None
    else:
        if callable(train_dset):
            train_loader = lambda : torch.utils.data.DataLoader(train_dset(), batch_size=args.batch_size, shuffle=True,
                                                                **dl_kwargs)
        else:
            train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.test_batch_size, shuffle=False, **dl_kwargs)

    # model
    model = models.create_model(args.model, num_classes)
    has_backbone = '_' in args.model
    frozen_backbone = has_backbone and args.model.split('_')[0].endswith('frozen')

    if has_backbone and args.load_weights_backbone:
        try:
            model.backbone.load_state_dict(torch.load(args.load_weights_backbone))
        except Exception:
            print('Warning: strict weight loading fails!')
            model.backbone.load_state_dict(torch.load(args.load_weights_backbone), strict=False)
        print('\nBackbone model weights initialized from: {}'.format(args.load_weights_backbone))
    if load_weights:
        try:
            model.load_state_dict(torch.load(load_weights))
        except Exception:
            print('Warning: strict weight loading fails!')
            model.load_state_dict(torch.load(load_weights), strict=False)
        print('\nModel weights initialized from: {}'.format(load_weights))
    if args.gpu is None and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # optimizer, scheduler, and logs
    if not test_only:
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError('Optimizer type ({}) is not supported.'.format(args.optimizer))

        load_optim_state = os.path.join(args.exp_root, '{}_epoch_{}.pth'.format(args.optimizer.lower(), last_epoch))
        if os.path.exists(load_optim_state):
            optimizer.load_state_dict(torch.load(load_optim_state))
            print('\nOptimizer state initialized from: {}'.format(load_optim_state))
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       [] if not args.lr_steps else [int(v) for v in args.lr_steps],
                                                       gamma=0.1,
                                                       last_epoch=last_epoch-1)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       [] if not args.lr_steps else [int(v) for v in args.lr_steps],
                                                       gamma=0.1,
                                                       last_epoch=-1)
            for _ in range(last_epoch):
                scheduler.step()

        # log files
        fmt_train = '{:.6f},{:g},{:.3f},{:.3f},{:.6f},{:.6f}'
        fmt_test = '{:.6f},{:g},{:.3f},{:.3f},{:.6f}'
        csvheader_train = 'epoch,lr,time,time-data,loss,loss-avg'
        csvheader_test = 'epoch,lr,time,time-data,loss'
        for m in perf_measures:
            fmt_train += ',{:.6f},{:.6f}'
            fmt_test += ',{:.6f}'
            csvheader_train += ',{},{}-avg'.format(m, m)
            csvheader_test += ',{}'.format(m)
        train_log_path = os.path.join(os.path.join(args.exp_root, 'train.log'))
        test_log_path = os.path.join(os.path.join(args.exp_root, 'test.log'))
        prepare_log(train_log_path, csvheader_train, last_epoch)
        prepare_log(test_log_path, csvheader_test, last_epoch)

    # main computation
    init_lr = 0 if test_only else scheduler.get_lr()[0]
    if args.eval:
        pred_dir = os.path.join(args.exp_root, 'outputs_{}_{}'.format(test_split, args.eval))
        evaluate(model, test_loader, device, args.test_batch_size, args.eval, pred_dir, palette)
    else:
        log_test = test(model, test_loader, device, last_epoch, init_lr, perf_measures, num_classes, args)
    if last_epoch == 0 and not test_only:
        with open(test_log_path, 'a') as f:
            f.writelines([','.join([('' if arg == -1 else fmt.format(arg)) for fmt, arg in zip(fmt_test.split(','), l)])
                          + '\n' for l in log_test])
    for epoch in range(last_epoch + 1, args.epochs + 1):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        log_train = train(model, train_loader, optimizer, device, epoch, lr, perf_measures, num_classes, args)
        with open(train_log_path, 'a') as f:
            f.writelines([','.join([('' if arg == -1 else fmt.format(arg)) for fmt, arg in zip(fmt_train.split(','), l)])
                          + '\n' for l in log_train])
        if epoch % args.test_interval == 0:
            log_test = test(model, test_loader, device, epoch, lr, perf_measures, num_classes, args)
            with open(test_log_path, 'a') as f:
                f.writelines(
                    [','.join([('' if arg == -1 else fmt.format(arg)) for fmt, arg in zip(fmt_test.split(','), l)])
                     + '\n' for l in log_test])
        if (args.snapshot_interval > 0 and epoch % args.snapshot_interval == 0) or (epoch == args.epochs):
            save_weights = os.path.join(args.exp_root, 'weights_epoch_{}.pth'.format(epoch))
            save_optim_state = os.path.join(args.exp_root, '{}_epoch_{}.pth'.format(args.optimizer.lower(), epoch))
            model.to('cpu')
            if isinstance(model, torch.nn.DataParallel):
                weights_dict = model.module.state_dict()
            else:
                weights_dict = model.state_dict()
            if frozen_backbone:
                weights_dict = OrderedDict((k, v) for k, v in weights_dict.items() if not k.startswith('backbone'))
            torch.save(weights_dict, save_weights)
            torch.save(optimizer.state_dict(), save_optim_state)
            model.to(device)
            print('Snapshot saved to: {}, {}\n'.format(save_weights, save_optim_state))


if __name__ == '__main__':
    main()

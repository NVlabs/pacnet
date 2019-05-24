"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from . import models, datasets


def apply_model(net, lres, guide, factor):
    h0, w0 = lres.shape[-2:]
    h, w = guide.shape[-2:]
    if h0 * factor != h or w0 * factor != w:
        guide = F.interpolate(guide, size=(h0 * factor, w0 * factor), align_corners=False, mode='bilinear')
        out = net(lres, guide)
        out = F.interpolate(out, size=(h, w), align_corners=False, mode='bilinear')
    else:
        out = net(lres, guide)
    return out


def train(model, train_loader, optimizer, device, epoch, lr, loss_type, perf_measures, args):
    model.train()
    log = []
    for batch_idx, sample in enumerate(train_loader):
        lres, guide, target = sample[0].to(device), sample[1].to(device), sample[2].to(device)
        if len(sample) >= 4:
            raw_range = sample[3].to(device)
            _ch = raw_range.shape[1]
            raw_min = raw_range[:, :, 0].view(-1, _ch, 1, 1)
            raw_scale = raw_range[:, :, 1].view(-1, _ch, 1, 1) - raw_min
        else:
            raw_min, raw_scale = 0.0, 1.0
        optimizer.zero_grad()
        output = apply_model(model, lres, guide, args.factor)
        crop = tuple(((o - t) // 2, t) for o, t in zip(output.shape[-2:], target.shape[-2:]))
        output = output[:, :, crop[0][0]:crop[0][0]+crop[0][1], crop[1][0]:crop[1][0]+crop[1][1]]
        if loss_type == 'l2':
            loss = F.mse_loss(output, (target - raw_min) / raw_scale)
        elif loss_type == 'epe':
            loss = models.th_epe((output * raw_scale) + raw_min, target)
        elif loss_type == 'rmse':
            loss = models.th_rmse((output * raw_scale) + raw_min, target)
        else:
            raise ValueError('Loss type ({}) not supported.'.format(args.loss))
        loss.backward()
        optimizer.step()
        batch_cnt = batch_idx + 1
        sample_cnt = batch_idx * args.batch_size + len(lres)
        progress = sample_cnt / len(train_loader.dataset)
        if batch_cnt == len(train_loader) or batch_cnt % args.log_interval == 0:
            log_row = [progress + epoch - 1, lr, loss.item()]
            for m in perf_measures:
                if m == 'epe':
                    log_row.append(models.th_epe((output * raw_scale) + raw_min, target).item())
                elif m == 'rmse':
                    log_row.append(models.th_rmse((output * raw_scale) + raw_min, target).item())
            log.append(log_row)
        if batch_cnt == len(train_loader) or batch_cnt % args.print_interval == 0:
            print('Train Epoch {} [{}/{} ({:3.0f}%)]\tLR: {:g}\tLoss: {:.6f}\t'.format(
                epoch, sample_cnt, len(train_loader.dataset), 100. * progress, lr, loss.item()))
    return log


def test(model, test_loader, device, epoch, lr, loss_type, perf_measures, args):
    model.eval()
    loss_accum = 0
    perf_measures_accum = [0.0] * len(perf_measures)
    with torch.no_grad():
        for sample in test_loader:
            lres, guide, target = sample[0].to(device), sample[1].to(device), sample[2].to(device)
            if len(sample) >= 4:
                raw_range = sample[3].to(device)
                _ch = raw_range.shape[1]
                raw_min = raw_range[:, :, 0].view(-1, _ch, 1, 1)
                raw_scale = raw_range[:, :, 1].view(-1, _ch, 1, 1) - raw_min
            else:
                raw_min, raw_scale = 0.0, 1.0
            output = apply_model(model, lres, guide, args.factor)
            crop = tuple(((o - t) // 2, t) for o, t in zip(output.shape[-2:], target.shape[-2:]))
            output = output[:, :, crop[0][0]:crop[0][0]+crop[0][1], crop[1][0]:crop[1][0]+crop[1][1]]

            if loss_type == 'l2':
                loss = F.mse_loss(output, (target - raw_min) / raw_scale)
            elif loss_type == 'epe':
                loss = models.th_epe((output * raw_scale) + raw_min, target)
            elif loss_type == 'rmse':
                loss = models.th_rmse((output * raw_scale) + raw_min, target)
            else:
                raise ValueError('Loss type ({}) not supported.'.format(args.loss))

            loss_accum += loss.item() * len(output)

            for i, m in enumerate(perf_measures):
                if m == 'epe':
                    perf_measures_accum[i] += models.th_epe((output * raw_scale) + raw_min, target).item() * len(output)
                elif m == 'rmse':
                    perf_measures_accum[i] += models.th_rmse((output * raw_scale) + raw_min, target).item() * len(output)

    test_loss = loss_accum / len(test_loader.dataset)
    log = [float(epoch), lr, test_loss]
    msg = 'Average loss: {:.6f}\n'.format(test_loss)
    for m, mv in zip(perf_measures, perf_measures_accum):
        avg = mv / len(test_loader.dataset)
        msg += '{}: {:.6f}\n'.format(m, avg)
        log.append(avg)
    print('\nTesting (#epochs={})'.format(epoch))
    print(msg)
    return [log]


def prepare_log(log_path, header, last_epoch=0):
    # keep all existing log lines up to epoch==last_epoch (included)
    try:
        log = np.genfromtxt(log_path, delimiter=',', skip_header=1, usecols=(0,))
    except:
        log = []
    if len(log) > 0:
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
    parser = argparse.ArgumentParser(description='Joint upsampling',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--factor', type=int, default=8, metavar='R',
                        help='upsampling factor')
    parser.add_argument('--data-root', type=str, default='data', metavar='D',
                        help='place to find (or download) data')
    parser.add_argument('--exp-root', type=str, default='exp', metavar='E',
                        help='place to save results')
    parser.add_argument('--download', default=False, action='store_true',
                        help='download dataset if not found locally')
    parser.add_argument('--load-weights', type=str, default='', metavar='L',
                        help='file with pre-trained weights')
    parser.add_argument('--model', type=str, default='PacJointUpsample', metavar='M',
                        help='network model type')
    parser.add_argument('--dataset', type=str, default='NYUDepthV2', metavar='D',
                        help='dataset')
    parser.add_argument('--lowres-mode', type=str, default='', metavar='LM',
                        help='overwrite how lowres samples are generated')
    parser.add_argument('--zero-guidance', default=False, action='store_true',
                        help='use zeros for guidance')
    parser.add_argument('--loss', type=str, default='l2', metavar='L',
                        help='choose a loss function type')
    parser.add_argument('--measures', nargs='+', default=None, metavar='M',
                        help='performance measures to be reported during training and testing')
    parser.add_argument('--num-data-worker', type=int, default=4, metavar='W',
                        help='number of subprocesses for data loading')
    parser.add_argument('--val-ratio', type=float, default=0.0, metavar='V',
                        help='use this portion of training set for validation')
    parser.add_argument('--train-split', type=str, default='', metavar='TRAIN',
                        help='specify a subset for training')
    parser.add_argument('--test-split', type=str, default='', metavar='TEST',
                        help='specify a subset for testing')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--train-crop', type=int, default=256, metavar='CROP',
                        help='input crop size in training')
    parser.add_argument('--eval-border', type=int, default=-1, metavar='EB',
                        help='specify a border that is excluded from evaluating loss and error')
    parser.add_argument('--epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='Adam', metavar='O',
                        help='pick which optimizer to use')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr-steps', nargs='+', default=None, metavar='S',
                        help='decrease lr by 10 at these epochs')
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='ignore existing log files and snapshots')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='WD',
                        help='Adam/SGD weight decay')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--print-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before displaying training status')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test-interval', type=int, default=10, metavar='N',
                        help='how many epochs to wait before a testing')
    parser.add_argument('--snapshot-interval', type=int, default=100, metavar='N',
                        help='snapshot intermediate models')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    assert(torch.cuda.is_available())
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    dl_kwargs = {'num_workers': args.num_data_worker, 'pin_memory': True} if use_cuda else {}

    # find existing snapshots
    os.makedirs(args.exp_root, exist_ok=True)
    snapshots_found = sorted([int(s.split('_')[-1].rstrip('.pth'))
                              for s in glob.glob(os.path.join(args.exp_root, 'weights_epoch_*.pth'))])
    load_weights = args.load_weights
    if snapshots_found and not args.overwrite:
        last_epoch = max(snapshots_found) if args.epochs > max(snapshots_found) else args.epochs
        assert last_epoch in snapshots_found
        assert not load_weights
        load_weights = os.path.join(args.exp_root, 'weights_epoch_{}.pth'.format(last_epoch))
    else:
        last_epoch = 0
    test_only = (args.epochs <= last_epoch)

    # dataset
    if args.dataset == 'NYUDepthV2':
        ch, guide_ch = 1, 3
        eval_border = 6 if args.eval_border < 0 else args.eval_border
        perf_measures = ('rmse',) if not args.measures else args.measures
        train_split = 'train' if not args.train_split else args.train_split
        test_split = 'test' if not args.test_split else args.test_split
        lowres_mode = 'center' if not args.lowres_mode else args.lowres_mode
        train_transform = datasets.AssembleJointUpsamplingInputs(args.factor, flip=True, lowres_mode=lowres_mode,
                                                                 zero_guidance=args.zero_guidance,
                                                                 output_crop=eval_border, crop=(
                None if args.train_crop <= 0 else args.train_crop))
        test_transform = datasets.AssembleJointUpsamplingInputs(args.factor, flip=False, lowres_mode=lowres_mode,
                                                                zero_guidance=args.zero_guidance,
                                                                output_crop=eval_border)
        if args.epochs > 0:
            train_dset = datasets.NYUDepthV2(args.data_root, transform=train_transform, download=args.download,
                                             split=train_split, val_ratio=args.val_ratio, cache_all=True)
        else:
            train_dset = None
        test_dset = datasets.NYUDepthV2(args.data_root, transform=test_transform, download=args.download,
                                        split=test_split, val_ratio=args.val_ratio, cache_all=True)
    elif args.dataset in ('Sintel', 'Sintel-clean', 'Sintel-final', 'Sintel-albedo'):
        render_pass = 'clean' if args.dataset == 'Sintel' else args.dataset.split('-')[1]
        ch, guide_ch = 1, 3
        eval_border = 0 if args.eval_border < 0 else args.eval_border
        perf_measures = ('epe',) if not args.measures else args.measures
        train_split = 'train' if not args.train_split else args.train_split
        test_split = 'val' if not args.test_split else args.test_split
        lowres_mode = 'bilinear' if not args.lowres_mode else args.lowres_mode
        train_transform = datasets.AssembleJointUpsamplingInputs(args.factor, flip=True, lowres_mode=lowres_mode,
                                                                 zero_guidance=args.zero_guidance,
                                                                 output_crop=eval_border, crop=(
                None if args.train_crop <= 0 else args.train_crop))
        test_transform = datasets.AssembleJointUpsamplingInputs(args.factor, flip=False, lowres_mode=lowres_mode,
                                                                zero_guidance=args.zero_guidance,
                                                                output_crop=eval_border)
        if args.epochs > 0:
            train_dset = datasets.Sintel(args.data_root, transform=train_transform, download=args.download,
                                         cache_all=True, fields=(render_pass + '_1', 'flow'), split=train_split)
        else:
            train_dset = None
        test_dset = datasets.Sintel(args.data_root, transform=test_transform, download=args.download, cache_all=True,
                                    fields=(render_pass + '_1', 'flow'), split=test_split)
    else:
        raise ValueError('Dataset ({}) not supported.'.format(args.dataset))

    # data loader
    if test_only:
        train_loader = None
    else:
        train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.test_batch_size, shuffle=True, **dl_kwargs)

    # model
    if args.model.startswith('JBU'):
        ks, s_color, s_spatial = args.model.split('_')[1:]
        model = models.JointBilateral(channels=ch, factor=args.factor, kernel_size=int(ks),
                                      scale_color=float(s_color), scale_space=float(s_spatial))
    else:
        model = models.__dict__[args.model](channels=ch, guide_channels=guide_ch, factor=args.factor)
    if load_weights:
        model.load_state_dict(torch.load(load_weights))
        print('\nModel weights initialized from: {}'.format(load_weights))
    model = model.to(device)

    # optimizer, scheduler, and logs
    if not test_only:
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError('Optimizer type ({}) is not supported.'.format(args.optimizer))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   [] if not args.lr_steps else [int(v) for v in args.lr_steps],
                                                   gamma=0.1,
                                                   last_epoch=-1)
        for s in range(last_epoch):
            scheduler.step()  # TODO： a temporary workaround -- ideally should recover optimizer from checkpoint instead

        # log files
        fmtstr = '{:.6f},{:g},{:.6f},{:.6f}'
        csv_header = 'epoch,lr,loss'
        for m in perf_measures:
            csv_header += (',' + m)
        train_log_path = os.path.join(os.path.join(args.exp_root, 'train.log'))
        test_log_path = os.path.join(os.path.join(args.exp_root, 'test.log'))
        prepare_log(train_log_path, csv_header, last_epoch)
        prepare_log(test_log_path, csv_header, last_epoch)

    # main computation
    init_lr = 0 if test_only else scheduler.get_lr()[0]
    log_test = test(model, test_loader, device, last_epoch, init_lr, args.loss, perf_measures, args)
    if last_epoch == 0 and not test_only:
        with open(test_log_path, 'a') as f:
            f.writelines([','.join([('' if arg == -1 else fmt.format(arg)) for fmt, arg in zip(fmtstr.split(','), l)])
                          + '\n' for l in log_test])
    for epoch in range(last_epoch + 1, args.epochs + 1):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        log_train = train(model, train_loader, optimizer, device, epoch, lr, args.loss, perf_measures, args)
        with open(train_log_path, 'a') as f:
            f.writelines([','.join([('' if arg == -1 else fmt.format(arg)) for fmt, arg in zip(fmtstr.split(','), l)])
                          + '\n' for l in log_train])
        if epoch % args.test_interval == 0:
            log_test = test(model, test_loader, device, epoch, lr, args.loss, perf_measures, args)
            with open(test_log_path, 'a') as f:
                f.writelines(
                    [','.join([('' if arg == -1 else fmt.format(arg)) for fmt, arg in zip(fmtstr.split(','), l)]) + '\n'
                     for l in log_test])
        if (args.snapshot_interval > 0 and epoch % args.snapshot_interval == 0) or (epoch == args.epochs):
            save_weights = os.path.join(args.exp_root, 'weights_epoch_{}.pth'.format(epoch))
            torch.save(model.to('cpu').state_dict(), save_weights)
            print('Snapshot saved to: {}\n'.format(save_weights))
            model = model.to(device)


if __name__ == '__main__':
    main()

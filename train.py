# -*- coding:UTF-8 -*-
import argparse
import os
import copy
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from datasets import TrainDataset, EvalDataset
from nets.RDASNet import RDASNet
from nets.my_model_4 import MODEL
from utils import AverageMeter, denormalize, calc_psnr

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='./data/Color_train_rgb.h5')
    parser.add_argument('--eval-file', type=str,  default='./data/Color_val_rgb.h5')
    parser.add_argument('--outputs-dir', type=str,  default='./models/RGB')
    parser.add_argument('--Gray', type=bool, default=True)
    parser.add_argument('--weights-file', type=str, default=None)
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--patch-size', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--val_noiseL', type=int, default=50)
    parser.add_argument("--noiseIntL", nargs=2, type=int, default=[0, 75], help="Noise training interval")
    args = parser.parse_args()

    args.val_noiseL /= 255.
    args.noiseIntL[0] /= 255.
    args.noiseIntL[1] /= 255.

    writer = SummaryWriter(comment='tensorboard')

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    in_channels = 3
    if args.Gray == True:  in_channels = 1
    net = RDASNet(scale_factor=args.scale,
                num_channels=in_channels,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers)

    if args.weights_file is not None:
        state_dict = net.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            n = n.replace('module.', '')
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

    criterion = nn.L1Loss()

    # Move to GPU
    device_ids = [0, 1, 2, 3]
    model = nn.DataParallel(net, device_ids=device_ids).cuda(0)
    criterion.cuda(0)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = TrainDataset(args.train_file, patch_size=args.patch_size, scale=args.scale)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (0.1 ** (epoch // int(args.num_epochs * 0.8)))

        model.train()
        epoch_losses = AverageMeter()
        train_epoch_psnr = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data
                noise = torch.zeros(inputs.size())
                stdn = np.random.uniform(args.noiseIntL[0], args.noiseIntL[1], size=noise.size()[0])
                for nx in range(noise.size()[0]):
                    sizen = noise[0, :, :, :].size()
                    noise[nx, :, :, :] = torch.FloatTensor(sizen).normal_(mean=0, std=stdn[nx])
                inputs = inputs + noise
                inputs = inputs.cuda(0)
                labels = labels.cuda(0)

                preds = model(inputs)

                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))
                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
                preds = denormalize(preds.squeeze(0))
                labels = denormalize(labels.squeeze(0))

                train_epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()
        
        psnr_test = 0

        for data in eval_dataloader:
            inputs, labels = data
            noise = torch.FloatTensor(inputs.size()).normal_(mean=0, std=args.val_noiseL)
            inputs = inputs + noise
            inputs = inputs.cuda(0)
            labels = labels.cuda(0)

            with torch.no_grad():
                preds = model(inputs)

            preds = denormalize(preds.squeeze(0))
            labels = denormalize(labels.squeeze(0))

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        writer.add_scalars('psnr', {"train_epoch_psnr":  train_epoch_psnr.avg.item(),
                                    "val_epoch_psnr": epoch_psnr.avg.item()}, epoch + 1)
        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))

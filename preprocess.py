# -*- coding:UTF-8 -*-
import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
import os

def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    image_list = sorted(glob.glob('{}/*'.format(args.images_dir)))
    mask_list = sorted(glob.glob('{}/*'.format(args.masks_dir)))

    patch_idx = 0
    patch_idx2 = 0

    for i, path in enumerate(image_list):
        if args.Gray == True:
            hr = pil_image.open(path).convert('L')
            hr = np.expand_dims(hr, 2)
        else:
            hr = pil_image.open(path).convert('RGB')
        hr = np.array(hr)
        hr_group.create_dataset(str(patch_idx), data=hr)
        print("input:", i, patch_idx, path, hr.shape)
        patch_idx += 1

    for j, path in enumerate(mask_list):
        if args.Gray == True:
            lr = pil_image.open(path).convert('L')
            lr = np.expand_dims(lr, 2)
        else:
            lr = pil_image.open(path).convert('RGB')
        lr = np.array(lr)
        lr_group.create_dataset(str(patch_idx2), data=lr)
        patch_idx2 += 1
        print("label:", i, patch_idx, path)
    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default='./datasets/Waterloo4744')  # img
    parser.add_argument('--masks-dir', type=str, default='./datasets/Waterloo4744')  # GT
    parser.add_argument('--output-path', type=str, default='./data/Waterloo4744.h5')
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--Gray', type=bool, default=False)
    args = parser.parse_args()

    train(args)

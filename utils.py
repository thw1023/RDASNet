# -*- coding:UTF-8 -*-
import torch
import numpy as np
from skimage.measure import compare_ssim
from torch.autograd import Variable
from skimage import measure
import cv2

def denormalize(img):
    img = img.mul(255.0).clamp(0.0, 255.0)

    return img

def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

def calc_psnr(img1, img2, max=255.0):
    return 10. * ((max ** 2) / ((img1 - img2) ** 2).mean()).log10()

def batch_psnr(img, imclean, data_range=1.):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += measure.compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def calc_ssim(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    ssim = compare_ssim(img1, img2, multichannel=True)

    return ssim

class AverageMeter(object):
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

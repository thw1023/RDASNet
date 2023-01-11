import glob
import os

from skimage.measure import compare_ssim, compare_psnr
import numpy as np
import cv2
import re

def calcuate():
    files_source = glob.glob(os.path.join('datasets/', 'Set12/', '*'))
    files_source.sort()

    ssim_sum = 0
    psnr_sum = 0
    for f in files_source:
        file_name = f.split('/')[-1].split('\\')[-1].split('.')[0]
        # print(file_name)
        f_result_path = os.path.join('results/', 'Set12/15/', file_name + '_denoise.png')
        print(f_result_path)
        img1 = cv2.imread(f)
        img2 = cv2.imread(f_result_path)

        img1 = np.array(img1)
        img2 = np.array(img2)
        psnr = compare_psnr(img1, img2)
        psnr_sum += psnr
        ssim = compare_ssim(img1, img2, multichannel=True)
        ssim_sum += ssim
        print('{} {} {}'.format(f, psnr, ssim))
    psnr_avg = psnr_sum / len(files_source)
    ssim_avg = ssim_sum / len(files_source)
    print('AVG {} {}'.format(psnr_avg, ssim_avg))

if __name__ == "__main__":
    # calculate SSIM
    calcuate()

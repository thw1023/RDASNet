import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch import nn
from torchvision import utils as vutils
import PIL.Image as pil_image
import os
import glob
import time
from nets.RDASNet import RDASNet
from nets.my_model_4 import MODEL
from utils import denormalize, calc_psnr, calc_ssim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-file', type=str, default='./models/Gray/best.pth')
    parser.add_argument('--image-file', type=str, default='Set12/')
    parser.add_argument('--result-file', type=str, default='results/')
    parser.add_argument('--Gray', type=bool, default=True)
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--test-noiseL', type=int, default=50)
    args = parser.parse_args()

    cudnn.benchmark = True

    in_channels = 3
    if args.Gray == True:  in_channels = 1
    net = MODEL(scale_factor=args.scale,
                  num_channels=in_channels,
                  num_features=args.num_features,
                  growth_rate=args.growth_rate,
                  num_blocks=args.num_blocks,
                  num_layers=args.num_layers)
    device_ids = [3]
    model = nn.DataParallel(net, device_ids=device_ids).cuda(3)
    state_dict = model.state_dict()

    # add model
    for n, p in torch.load(args.models_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('datasets/', args.image_file, '*'))
    files_source.sort()

    out_dir = os.path.join(args.result_file, args.image_file, str(args.test_noiseL))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_file = open(out_dir+"/log.txt", "a+")

    psnr_test = 0
    for f in files_source:
        if args.Gray:
            ISource = pil_image.open(f).convert('L')
            In_img = ISource
            ISource = np.expand_dims(ISource, 2)
        else:
            ISource = pil_image.open(f).convert('RGB')
            In_img = ISource
        ISource = np.expand_dims(np.array(ISource).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        ISource = torch.from_numpy(ISource).cuda(3)

        INoise = ISource
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=args.test_noiseL / 255.).cuda(3)
        INoise = ISource + noise

        with torch.no_grad():
            start = time.time()
            IOut = model(INoise).squeeze(0)
            time_elapsed = time.time() - start
        print('Running complete in {:,.4f}s'.format(time_elapsed))

        IOut_img = denormalize(IOut)
        ISource_img = denormalize(ISource.squeeze(0))
        INoise_img = denormalize(INoise.squeeze(0))

        out_file = os.path.join(out_dir, f.split('/')[-1])
        vutils.save_image(denormalize(INoise.squeeze(0)), out_file.replace('.', '_noise.'), normalize=True)
        output = denormalize(IOut).permute(1, 2, 0).byte().cpu().numpy()
        if args.Gray:
            output = pil_image.fromarray(output.squeeze(2)).convert('L')
        else:
            output = pil_image.fromarray(output).convert('RGB')

        psnr = calc_psnr(ISource_img, IOut_img)
        psnr_noise = calc_psnr(ISource_img, INoise_img)
        psnr_test += psnr
        # print(In_img.shape)
        # print(output.shape)

        print('{} PSNR: {:.4f}'.format(f, psnr))
        log_file.write(f + " PSNR: " + str(psnr.item()) + "\n")
        log_file.write('Running complete in {:,.4f}s \n'.format(time_elapsed))
        output.save(out_file.replace('.', '_denoise.'.format(args.scale)))

    psnr_test /= len(files_source)
    log_file.write("PSNR: " + str(psnr_test.item()) + "\n\n")
    log_file.close()
    print("PSNR on test data %f" % (psnr_test))

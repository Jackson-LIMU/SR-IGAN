import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestMyDataset, display_transform, TestMyDatasetChop

from module.srcnn import SRCNN

parser = argparse.ArgumentParser(description='Test Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_best_psnr_4.pth', type=str, help='generator model epoch name')
# parser.add_argument('--model_name', default='netG_epoch_4_last.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name

model = SRCNN().eval()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

# test_set = TestMyDatasetChop('/media/b227/加量不加价/data/Code/SR-Code/111/SRGAN-master/data/benchmark/Set5/HR', upscale_factor=UPSCALE_FACTOR)
test_set = TestMyDatasetChop('/media/cumtailab227/加量不加价/data/Code/SR-Code/111/SRGAN-master/data/coal_test', upscale_factor=UPSCALE_FACTOR)
# test_set = TestMyDatasetChop('F:/data/Code/SR-Code/111/SRGAN-master/data/mytest', upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

results = {'psnr': [], 'ssim': []}
out_path = 'test_results/SRF_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

testing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
test_images = []

for lr_image, hr_restore_img, hr_image in test_bar:
    batch_size = lr_image.size(0)
    testing_results['batch_sizes'] += batch_size
    lr_image = Variable(lr_image, volatile=True)
    hr_image = Variable(hr_image, volatile=True)
    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()
        hr_restore_img = hr_restore_img.cuda()

    sr_image = model(hr_restore_img)
    mse = ((hr_image - sr_image) ** 2).data.mean()
    testing_results['mse'] += mse * batch_size
    ssim = pytorch_ssim.ssim(sr_image, hr_image).data.item()
    testing_results['ssims'] += ssim * batch_size
    testing_results['psnr'] = 10 * log10((hr_image.max() ** 2) / (testing_results['mse'] / testing_results['batch_sizes']))
    testing_results['ssim'] = testing_results['ssims'] / testing_results['batch_sizes']
    test_bar.set_description(
        desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
            testing_results['psnr'], testing_results['ssim']))

    test_images = torch.stack(
        [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
         display_transform()(sr_image.data.cpu().squeeze(0))])
    image = utils.make_grid(test_images, nrow=3, padding=5)
    utils.save_image(image, out_path + '_psnr_%.4f_ssim_%.4f..png' % (testing_results['psnr'], testing_results['ssim']), padding=5)


    # test_images = torch.stack(
    #     [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
    #      display_transform()(sr_image.data.cpu().squeeze(0))])
    # image = utils.make_grid(test_images, nrow=3, padding=5
csv_path = 'test_csv/'
if not os.path.exists(csv_path):
    os.makedirs(csv_path)

results['psnr'].append(testing_results['psnr'])
results['ssim'].append(testing_results['ssim'])

data_frame = pd.DataFrame(data={'PSNR': results['psnr'], 'SSIM': results['ssim']})
data_frame.to_csv(csv_path + '×' + str(UPSCALE_FACTOR) + '_test_results.csv')

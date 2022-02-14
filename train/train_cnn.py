import argparse
import warnings
import os
from math import log10
import hiddenlayer as hl
# from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, TestMyDatasetChop
from loss import GeneratorLoss
from module.srcnn import SRCNN

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=700, type=int, help='train epoch number')
bce = nn.BCELoss()

if __name__ == '__main__':
    opt = parser.parse_args()
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    # writer = SummaryWriter(log_dir='/media/b227/加量不加价/data/Code/SR-Code/111/SRGAN-master/logs')
    # train_set = TrainDatasetFromFolder(r'/media/b227/加量不加价/data/Code/SR-Code/SRGAN-master/data/coal_train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    train_set = TrainDatasetFromFolder(r'/media/cumtailab227/加量不加价/data/Code/SR-Code/111/SRGAN-master/data/coal_train',
                                       crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    # train_set = TrainDatasetFromFolder(/media/b227/加量不加价/data/Code/SR-Code/111/SRGAN-master/show/coal_panet_adafm_8/weights/32.5974/netD_epoch_4_last.pth
    # /media/b227/加量不加价/data/Code/SR-Code/111/SRGAN-master/show/coal_panet_adafm_8/weights/32.5974/netG_best_psnr_4.pth
    # /media/b227/加量不加价/data/Code/SR-Code/111/SRGAN-master/show/coal_panet_adafm_8/weights/32.5974/netG_epoch_4_last.pthr'/media/b227/加量不加价/data/Code/SR-Code/111/SRGAN-master/data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = TestMyDatasetChop(r'/media/cumtailab227/加量不加价/data/Code/SR-Code/111/SRGAN-master/data/val_val',
                                upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    model = SRCNN()
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    generator_criterion = nn.MSELoss()
    history1 = hl.History()
    canvas1 = hl.Canvas()

    if torch.cuda.is_available():
        model.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(model.parameters())


    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    best_psnr = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0, 'psnr': 0, 'mse': 0, 'ssim': 0, 'ssims': 0}

        model.train()
        for data, target, restore_hr in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(restore_hr)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = model(z)

            batch_mse = ((fake_img - real_img) ** 2).data.mean()
            running_results['mse'] += batch_mse * batch_size
            running_results['psnr'] = 10 * log10(
                (real_img.max() ** 2) / (running_results['mse'] / running_results['batch_sizes']))
            batch_ssim = pytorch_ssim.ssim(fake_img, real_img).item()
            running_results['ssims'] += batch_ssim * batch_size
            running_results['ssim'] = running_results['ssims'] / running_results['batch_sizes']

            real_out = model(real_img)
            model.zero_grad()

            g_loss = generator_criterion(fake_img, real_img)
            g_loss.backward()

            optimizerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            # running_results['d_score'] += real_out.item() * batch_size
            # running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] trian_Loss_D: %.4f train_Loss_G: %.4f PSNR:%.4f SSIM:%.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],running_results['psnr'],running_results['ssim']))
            # running_results['d_score'] / running_results['batch_sizes'],
            # running_results['g_score'] / running_results['batch_sizes']))

        model.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_hr_restore
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = model(lr)


                val_g_loss = generator_criterion(sr, hr)

                valing_results['g_loss'] += val_g_loss.item() * batch_size

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(
                    (hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[result] val_Loss_D: %.4f val_Loss_G: %.4f PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['d_loss'] / valing_results['batch_sizes'],valing_results['g_loss'] / valing_results['batch_sizes'],
                        valing_results['psnr'], valing_results['ssim']))

                history1.log(epoch, train_g_loss=running_results['g_loss'] / running_results['batch_sizes'],
                                    val_g_loss=valing_results['g_loss'] / valing_results['batch_sizes'],
                                    train_psnr=running_results['psnr'],
                                    val_psnr=valing_results['psnr'])

                with canvas1:
                    canvas1.draw_plot([history1['train_g_loss'],history1['val_g_loss']])
                    canvas1.draw_plot([history1['train_psnr'],history1['val_psnr']])


                is_best = max(valing_results['psnr'], best_psnr)
                if is_best > best_psnr:
                    best_psnr = is_best
                    torch.save(model.state_dict(), 'epochs/netG_best_psnr_%d.pth' % (UPSCALE_FACTOR))

                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1
        # save model parameters
        # torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        # results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        # results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            out_path = 'benchmark_csv/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'],
                      'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')

    torch.save(model.state_dict(), 'epochs/netG_epoch_%d_last.pth' % (UPSCALE_FACTOR))


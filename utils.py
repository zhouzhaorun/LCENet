# --- Imports --- #
import random
import numpy as np
import time
import torch
import torch.nn.functional as F
from math import log10
from skimage import measure
from PIL import Image
from skimage.metrics import structural_similarity

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)
    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [structural_similarity(dehaze_list_np[ind], gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]
    return ssim_list


def print_log(epoch, num_epochs, val_psnr, val_ssim):
    print('Epoch [{0}/{1}], Val_PSNR:{2:.2f}, Val_SSIM:{3:.4f}'.format(epoch, num_epochs, val_psnr, val_ssim))
    # --- Write the training log --- #
    with open('./result/training_log/log.txt', 'a') as f:
        print('Date: {0}s, Epoch: [{1}/{2}], Val_PSNR: {3:.2f}, Val_SSIM: {4:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch, num_epochs, val_psnr, val_ssim), file=f)


def transform_invert(img_):
    img_ = img_.squeeze(0).transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = img_.detach().cpu().numpy() * 255.0

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_


def validation(net, val_data_loader, device, epoch, save_valid_dir):
    psnr_list = []
    ssim_list = []
    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad():
            lowlight, reallight = val_data
            lowlight = lowlight.to(device)
            reallight = reallight.to(device)
            tin, tout, lin, lout, delow = net(lowlight)
        psnr_list.extend(to_psnr(delow, reallight))
        ssim_list.extend(to_ssim_skimage(delow, reallight))
        img_ = transform_invert(delow)
        img_.save(save_valid_dir + 'image_out/' + str(epoch) + '_' + str(batch_id) + '.png')
        tin_ = transform_invert(tin)
        tin_.save(save_valid_dir + 'gradient_in/' + str(epoch) + '_' + str(batch_id) + '.png')
        tout_ = transform_invert(tout)
        tout_.save(save_valid_dir + 'gradient_out/' + str(epoch) + '_' + str(batch_id) + '.png')
        lin_ = transform_invert(lin)
        lin_.save(save_valid_dir + 'light_in/' + str(epoch) + '_' + str(batch_id) + '.png')
        lout_ = transform_invert(lout)
        lout_.save(save_valid_dir + 'light_out/' + str(epoch) + '_' + str(batch_id) + '.png')
    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)

    return avr_psnr, avr_ssim

def t_model(net, val_data_loader, device, save_dir):
    timelist = 0.0
    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad():
            lowlight, img_name = val_data
            lowlight = lowlight.to(device)
            print(lowlight.size())
            start = time.time()
            tin, tout, lin, lout, delow = net(lowlight)
            print(delow.size())
            end = time.time()
            if batch_id != 0:
                timelist += end - start
            print(end-start)

        print(batch_id)
        img_ = transform_invert(delow)
        img_.save(save_dir + 'image_out' + '/' + img_name[0])  # + 'image_out'
        # tin_ = transform_invert(tin)
        # tin_.save(save_dir + 'gradient_in' + '/' + img_name[0])
        # tout_ = transform_invert(tout)
        # tout_.save(save_dir + 'gradient_out' + '/' + img_name[0])
        # lin_ = transform_invert(lin)
        # lin_.save(save_dir + 'light_in' + '/' + img_name[0])
        # lout_ = transform_invert(lout)
        # lout_.save(save_dir + 'light_out' + '/' + img_name[0])
    print("time mean :")
    print(timelist / 14.0)



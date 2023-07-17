import sys

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import random
import math
import PIL
import matplotlib.pyplot as plt
import cfg

args = cfg.parse_args()
def generate_click_prompt(img, msk, pt_label = 1):
    # return: prompt, prompt mask
    pt_list = []
    msk_list = []
    b, c, h, w, d = msk.size()
    msk = msk[:,0,:,:,:]
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j,:,:,i]
            indices = torch.nonzero(msk_s)
            if indices.size(0) == 0:
                # generate a random array between [0-h, 0-h]:
                random_index = torch.randint(0, h, (2,)).to(device = msk.device)
                new_s = msk_s
            else:
                random_index = random.choice(indices)
                label = msk_s[random_index[0], random_index[1]]
                new_s = torch.zeros_like(msk_s)
                # convert bool tensor to int
                new_s = (msk_s == label).to(dtype = torch.float)
                # new_s[msk_s == label] = 1
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)
        pts = torch.stack(pt_list_s, dim=0)
        msks = torch.stack(msk_list_s, dim=0)
        pt_list.append(pts)
        msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1)
    msk = torch.stack(msk_list, dim=-1)

    msk = msk.unsqueeze(1)

    return img, pt, msk #[b, 2, d], [b, c, h, w, d]


def vis_image(imgs, pred_masks, gt_masks, save_path, reverse=False, points=None):
    b, c, h, w = pred_masks.size()
    dev = pred_masks.get_device()
    row_num = min(b, 4)

    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    if reverse == True:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks
    if c == 2:
        pred_disc, pred_cup = pred_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w), pred_masks[:, 1, :, :].unsqueeze(
            1).expand(b, 3, h, w)
        gt_disc, gt_cup = gt_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w), gt_masks[:, 1, :, :].unsqueeze(
            1).expand(b, 3, h, w)
        tup = (
        imgs[:row_num, :, :, :], pred_disc[:row_num, :, :, :], pred_cup[:row_num, :, :, :], gt_disc[:row_num, :, :, :],
        gt_cup[:row_num, :, :, :])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat((pred_disc[:row_num, :, :, :], pred_cup[:row_num, :, :, :], gt_disc[:row_num, :, :, :],
                             gt_cup[:row_num, :, :, :]), 0)
        vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)
    else:
        imgs = torchvision.transforms.Resize((h, w))(imgs)
        if imgs.size(1) == 1:
            imgs = imgs[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        pred_masks = pred_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        gt_masks = gt_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        if points != None:
            for i in range(b):
                if args.thd:
                    p = np.round(points.cpu() / args.roi_size * args.out_size).to(dtype=torch.int)
                else:
                    p = np.round(points.cpu() / args.image_size * args.out_size).to(dtype=torch.int)
                # gt_masks[i,:,points[i,0]-5:points[i,0]+5,points[i,1]-5:points[i,1]+5] = torch.Tensor([255, 0, 0]).to(dtype = torch.float32, device = torch.device('cuda:' + str(dev)))
                gt_masks[i, 0, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.5
                gt_masks[i, 1, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.1
                gt_masks[i, 2, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.4
        tup = (imgs[:row_num, :, :, :], pred_masks[:row_num, :, :, :], gt_masks[:row_num, :, :, :])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat(tup, 0)
        vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)

    return


def eval_seg(pred, true_mask_p, threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0, 0, 0, 0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:, 0, :, :].numpy().astype('int32')
            cup_pred = vpred_cpu[:, 1, :, :].numpy().astype('int32')

            disc_mask = gt_vmask_p[:, 0, :, :].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p[:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')

            '''iou for numpy'''
            iou_d += iou(disc_pred, disc_mask)
            iou_c += iou(cup_pred, cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()
            cup_dice += dice_coeff(vpred[:, 1, :, :], gt_vmask_p[:, 1, :, :]).item()

        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    else:
        eiou, edice = 0, 0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:, 0, :, :].numpy().astype('int32')

            disc_mask = gt_vmask_p[:, 0, :, :].squeeze(1).cpu().numpy().astype('int32')

            '''iou for numpy'''
            eiou += iou(disc_pred, disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()

        return eiou / len(threshold), edice / len(threshold)

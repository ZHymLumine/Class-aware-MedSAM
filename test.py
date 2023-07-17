"""
test the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from monai.losses import DiceCELoss, DiceLoss
import surface_distance
from surface_distance import metrics
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from dataset.datasets import *
from utils import *
import logging
from einops import rearrange
import cfg
from utils.utils import setup_logger
from utils.SurfaceDice import *

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()


# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

args = cfg.parse_args()
device = torch.device(args.device)
class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
                np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
                "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )


if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(project=args.task_name, config={"lr": args.lr, "batch_size": args.batch_size,
                                               "data_path": args.tr_npy_path,
                                               "model_type": args.model_type,
                                               })


class MedSAM(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

def main():
    sam_model = sam_model_registry[args.model_type](args, checkpoint=args.checkpoint)
    medsam_model = MedSAM(image_encoder=sam_model.image_encoder,
                          mask_decoder=sam_model.mask_decoder,
                          prompt_encoder=sam_model.prompt_encoder,
                          ).to(device)
    seg_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    test_dataset = NpyDataset(args.ts_npy_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    validate(test_dataloader, medsam_model, seg_loss, ce_loss)

    return



def validate(test_dataloader, medsam_model, seg_loss, ce_loss):
    os.makedirs('./log', exist_ok=True)
    setup_logger(logger_name="test", root='./log', screen=True, tofile=True)
    logger = logging.getLogger(f"test")
    logger.info(str(args))
    medsam_model.eval()
    n_val = len(test_dataloader)
    tot_loss = 0
    dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    sam_trans = ResizeLongestSide(medsam_model.image_encoder.img_size)
    medsam_segs = []
    with torch.no_grad():
        loss_summary = []
        loss_nsd = []
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(test_dataloader)):
            # predict the segmentation mask using the fine-tuned model
            image, gt2D = image.to(device), gt2D.to(device)
            H, W = image.shape[:2]
            resize_img = medsam_model.apply_image(image)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
            input_image = medsam_model.preprocess(resize_img_tensor[None, :, :, :])  # (1, 3, 1024, 1024)

            bbox = medsam_model.apply_boxes(boxes, (H, W))

            image_embedding = medsam_model.image_encoder(input_image.to(device))  # (1, 256, 64, 64)
            # convert box to 1024x1024 grid
            bbox = sam_trans.apply_boxes(bbox, (H, W))
            box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            medsam_seg_prob, _ = medsam_model.mask_decoder(
                image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
                image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
            # convert soft mask to hard mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            medsam_segs.append(medsam_seg)
            print("gt size ", gt2D.size(), '\n', gt2D)
            print("seg size ", medsam_seg.shape, '\n', medsam_seg)
            medsam_dsc = compute_dice_coefficient(gt2D>0, medsam_seg>0)
            loss_summary.append(medsam_dsc)

            ssd = compute_surface_distances((gt2D == 1)[0, 0].cpu().numpy(),
                                            (medsam_seg == 1)[0, 0].cpu().numpy(),
                                            spacing_mm=[1, 1])
            nsd = compute_surface_dice_at_tolerance(ssd, args.tolerance)
            logger.info(
                " Case {} - Dice {:.6f} | NSD {:.6f}".format(
                    step, medsam_dsc, nsd
                ))
        logging.info("- Test metrics Dice: " + str(np.mean(loss_summary)))
        logging.info("- Test metrics NSD: " + str(np.mean(loss_nsd)))

def get_bbox_from_mask(mask):
    '''Returns a bounding box from a mask'''
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))

    return np.array([x_min, y_min, x_max, y_max])



if __name__ == "__main__":
    main()
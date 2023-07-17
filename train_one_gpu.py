"""
train the image encoder and mask decoder
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


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0, 0, 0, 0), lw=2))


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

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + '-' + run_id)
device = torch.device(args.device)


# %% set up model

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
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(__file__, join(model_save_path, run_id + '_' + os.path.basename(__file__)))

    sam_model = sam_model_registry[args.model_type](args, checkpoint=args.checkpoint)
    medsam_model = MedSAM(image_encoder=sam_model.image_encoder,
                          mask_decoder=sam_model.mask_decoder,
                          prompt_encoder=sam_model.prompt_encoder,
                          ).to(device)

    ##---------------------_! ----------------
    medsam_model.train()

    for n, value in medsam_model.image_encoder.named_parameters():
        if "Adapter" not in n:
            value.requires_grad = False
    print('Number of total parameters: ', sum(p.numel() for p in medsam_model.parameters()))  # 93735472
    print('Number of trainable parameters: ',
          sum(p.numel() for p in medsam_model.parameters() if p.requires_grad))  # 93729252

    img_mask_encdec_params = list(
        medsam_model.image_encoder.parameters()
    ) + list(
        medsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print('Number of image encoder and mask decoder parameters: ',
          sum(p.numel() for p in img_mask_encdec_params if p.requires_grad))  # 93729252
    seg_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10
    # train_loader, val_loader = get_dataset(args)

    train_dataset = NpyDataset(args.tr_npy_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            medsam_model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    medsam_model.train()
    for epoch in range(start_epoch, num_epochs):
        epoch_loss, tot_step = train_epoch(epoch, train_dataloader, medsam_model,
                                           optimizer, args, seg_loss, ce_loss, losses, best_loss)

        epoch_loss /= tot_step

        losses.append(epoch_loss)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        # save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        # save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()

    """ evaluation """
    test_dataset = NpyDataset(args.ts_npy_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    validate(test_dataloader, medsam_model, seg_loss, ce_loss)


def train_epoch(epoch, train_dataloader, medsam_model, optimizer, args, seg_loss, ce_loss, losses, best_loss):
    epoch_loss = 0
    tot_step = 0
    print(f'Epoch: {epoch}')
    for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        boxes_np = boxes.detach().cpu().numpy()
        image, gt2D = image.to(device), gt2D.to(device)
        if args.use_amp:
            ## AMP
            scaler = torch.cuda.amp.GradScaler()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                medsam_pred = medsam_model(image, boxes_np)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                    medsam_pred, gt2D.float()
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            medsam_pred = medsam_model(image, boxes_np)
            loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        tot_step += 1

    return loss.item(), tot_step


def validate(test_dataloader, medsam_model, seg_loss, ce_loss):
    setup_logger(logger_name="test", root='./log', screen=True, tofile=True)
    logger = logging.getLogger(f"test")
    logger.info(str(args))
    medsam_model.eval()
    n_val = len(test_dataloader)
    tot_loss = 0
    dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')

    with torch.no_grad():
        loss_summary = []
        loss_nsd = []
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(test_dataloader)):
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            if args.use_amp:
                ## AMP
                scaler = torch.cuda.amp.GradScaler()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    masks = (medsam_pred > 0.5).astype(np.uint8)
                    # dice_loss = (medsam_pred, gt2D)
                    # dice = dice_loss + ce_loss(
                    #     medsam_pred, gt2D.float()
                    # )
            else:
                medsam_pred = medsam_model(image, boxes_np)
                masks = (medsam_pred > 0.5).astype(np.uint8)
                # loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())

            loss = 1 - dice_loss(masks, gt2D)
            loss_summary.append(loss.detach().cpu().numpy())

            masks = medsam_pred > 0.5
            ssd = surface_distance.compute_surface_distances((gt2D == 1)[0, 0].cpu().numpy(),
                                                             (masks == 1)[0, 0].cpu().numpy())
            nsd = metrics.compute_surface_dice_at_tolerance(ssd, args.tolerance)  # kits
            loss_nsd.append(nsd)

            logger.info(
                " Case {} - Dice {:.6f} | NSD {:.6f}".format(
                    step, loss.item(), nsd
                ))
        logging.info("- Test metrics Dice: " + str(np.mean(loss_summary)))
        logging.info("- Test metrics NSD: " + str(np.mean(loss_nsd)))


if __name__ == "__main__":
    main()

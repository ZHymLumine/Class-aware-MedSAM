import glob
import math
import os
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam, SGD
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import (
    CacheDataset,
    DataLoader,
    ThreadDataLoader,
    Dataset,
    decollate_batch,
    set_track_meta,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers import Act, Norm
from monai.networks.nets import UNet
from monai.transforms import (
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureTyped,
    FgBgToIndicesd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.utils import set_determinism

# for profiling
from monai.utils.nvtx import Range
import contextlib  # to improve code readability (combining training/validation loop with and without profiling)

profiling = False

# to see the trend in training curve and dice results, set max_epochs to be larger (300)
# note that before optimization, training can be quite a bit slower
if profiling:
    max_epochs = 6
else:
    max_epochs = 300

# to improve readability


def range_func(x, y):
    return Range(x)(y) if profiling else y


no_profiling = contextlib.nullcontext()


def transformations(fast=False, device="cuda:0"):
    train_transforms = [
        range_func("LoadImage", LoadImaged(keys=["image", "label"])),
        range_func("EnsureChannelFirst", EnsureChannelFirstd(keys=["image", "label"])),
        range_func("Orientation", Orientationd(keys=["image", "label"], axcodes="RAS")),
        range_func(
            "Spacing",
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
        ),
        range_func(
            "ScaleIntensityRange",
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ),
        range_func("CropForeground", CropForegroundd(keys=["image", "label"], source_key="image")),
        # pre-compute foreground and background indexes
        # and cache them to accelerate training
        range_func(
            "Indexing",
            FgBgToIndicesd(
                keys="label",
                fg_postfix="_fg",
                bg_postfix="_bg",
                image_key="image",
            ),
        ),
    ]

    if fast:
        # convert the data to Tensor without meta, move to GPU and cache to avoid CPU -> GPU sync in every epoch
        train_transforms.append(
            range_func("EnsureType", EnsureTyped(keys=["image", "label"], device=device, track_meta=False))
        )

    train_transforms.append(
        # randomly crop out patch samples from big
        # image based on pos / neg ratio
        # the image centers of negative samples
        # must be in valid image area
        range_func(
            "RandCrop",
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                fg_indices_key="label_fg",
                bg_indices_key="label_bg",
            ),
        ),
    )

    val_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
    if fast:
        # convert the data to Tensor without meta, move to GPU and cache to avoid CPU -> GPU sync in every epoch
        val_transforms.append(EnsureTyped(keys=["image", "label"], device=device, track_meta=False))

    return Compose(train_transforms), Compose(val_transforms)

def get_dataset(args):
    data_root = args.data_path
    fast = args.fast
    device = args.device
    print_config()

    train_images = sorted(glob.glob(os.path.join(data_root, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_root, "labelsTr", "*.nii.gz")))

    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]

    train_trans, val_trans = transformations(fast=fast, device=device)
    if fast:
        # as `RandCropByPosNegLabeld` crops from the cached content and `deepcopy`
        # the crop area instead of modifying the cached value, we can set `copy_cache=False`
        # to avoid unnecessary deepcopy of cached content in `CacheDataset`
        train_ds = CacheDataset(
            data=train_files,
            transform=train_trans,
            cache_rate=1.0,
            num_workers=8,
            copy_cache=False,
        )
        val_ds = CacheDataset(data=val_files, transform=val_trans, cache_rate=1.0, num_workers=5, copy_cache=False)
        # disable multi-workers because `ThreadDataLoader` works with multi-threads
        train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=4, shuffle=True)
        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

        set_track_meta(False)
    else:
        train_ds = Dataset(data=train_files, transform=train_trans)
        val_ds = Dataset(data=val_files, transform=val_trans)
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    return train_loader, val_loader
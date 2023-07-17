import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def create_loader(dataset, batch_size, is_training=False, num_workers=1, pin_memory=False):
    # 简化版本的数据预处理，你可能需要根据你的任务来调整这些变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset.transform = transform

    # 简化版本的数据加载器，省略了一些特性如预加载、多周期加载等
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_training,
    )

    return loader

import torch
from torch.utils.data import DataLoader

from src.dataset import PlantDataset
from src.preprocessing.transforms import train_transform, eval_transform


def create_dataloaders(train_df, val_df, test_df, class_to_idx,
                       batch_size=32, num_workers=0):

    train_dataset = PlantDataset(
        train_df, class_to_idx, train_transform
    )

    val_dataset = PlantDataset(
        val_df, class_to_idx, eval_transform
    )

    test_dataset = PlantDataset(
        test_df, class_to_idx, eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
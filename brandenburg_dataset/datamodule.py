import os
from typing import Optional
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from configparser import NoOptionError
from brandenburg_dataset.dataset import BrandenburgDataset

"""
Trainer args (accelerator, devices, num_nodes, etc…)
Data args (sequence length, stride, etc...)
Model specific args (layer_dim, num_layers, learning_rate, etc…)
Program arguments (data_path, cluster_email, etc…)
"""


class BrandenburgDataModule(LightningDataModule):
    def __init__(self, cfg):

        self.data_dir = data_dir = cfg.get("program", "data_dir")
        
        self.sequence_len = cfg.getint("dataset", "sequence_len")
        self.sample_itvl = cfg.getint("dataset", "sample_itvl")
        self.stride = cfg.getint("dataset", "stride")

        # self.transform = transform
        self.batch_size = cfg.getint("loader", "batch_size")
        self.num_workers = cfg.getint("loader", "num_workers")
        self.train_shuffle = cfg.getboolean("loader", "train_shuffle")
        self.test_shuffle = cfg.getboolean("loader", "test_shuffle")
        self.pin_memory = cfg.getboolean("loader", "pin_memory")
        self.sampler = cfg.get("loader", "sampler")

        try:
            self.which_classes = cfg.get("dataset", "classes")
        except NoOptionError:
            self.which_classes = None

        super().__init__()

    def setup(self, stage: Optional[str] = None):
        
        self.spatial_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((244, 244)),
            ]
        )
        self.temporal_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((244, 244)),
            ]
        )

        if stage == "fit" or stage is None:

            self.train_dataset = BrandenburgDataset(
                data_dir=os.path.join(self.data_dir, "train"),
                sequence_len=self.sequence_len,
                sample_itvl=self.sample_itvl,
                stride=self.stride,
                transform=self.spatial_transform,
            )

            self.validation_dataset = BrandenburgDataset(
                data_dir=os.path.join(self.data_dir, "validation"),
                sequence_len=self.sequence_len,
                sample_itvl=self.sample_itvl,
                stride=self.stride,
                transform=self.temporal_transform,
            )

        if stage == "test" or stage is None:

            self.test_dataset = BrandenburgDataset(
                data_dir=os.path.join(self.data_dir, "test"),
                sequence_len=self.sequence_len,
                sample_itvl=self.sample_itvl,
                stride=self.stride,
                transform=self.spatial_transform,
            )

    def train_dataloader(self):

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        return train_loader

    def val_dataloader(self):
        validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=self.test_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        return validation_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.test_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        return test_loader

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        """
        Custom dataset for loading driver images with data augmentation.

        Args:
            path (str): Path to dataset.
            train (bool): True if loading training data, False for test data.
            transform (callable, optional): Transform to be applied to the images.
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        if train:
            for class_index in range(10):
                class_path = os.path.join(path, f'c{class_index}')
                if not os.path.exists(class_path):
                    raise FileNotFoundError(f"Class folder '{class_path}' not found.")
                
                files = os.listdir(class_path)
                for filename in files:
                    self.image_paths.append(os.path.join(class_path, filename))
                    self.labels.append(class_index)
        else:
            self.image_paths = os.listdir(path)
            self.labels = None
        self.train = train
        self.path = path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.train:
            image_path = self.image_paths[idx]
            label = self.labels[idx]
        else:
            image_path = os.path.join(self.path, self.image_paths[idx])
            label = self.image_paths[idx]

        img = Image.open(image_path).convert('RGB')

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        if self.train:
            return img, label
        else:
            return img, label


def load_datasets(train_path, test_path, batch_size=128):
    """
    Load datasets and create DataLoaders for training, validation, and testing.

    Args:
        train_path (str): Path to training dataset.
        test_path (str): Path to testing dataset.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = val_transform

    # Training dataset with augmentation
    train_dataset = CustomDataset(train_path, train=True, transform=train_transform)

    # Split into training and validation datasets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Apply validation transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Testing dataset
    test_dataset = CustomDataset(test_path, train=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples, and {len(test_dataset)} test samples.")
    return train_loader, val_loader, test_loader

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class OnTheFlyDataset(Dataset):
    """Dataset class to load and preprocess images on the fly."""
    def __init__(self, root_dir, labels=None, transform=None, return_filenames=False):
        self.root_dir = root_dir.replace("\\", "/")
        self.labels = labels
        self.transform = transform
        self.return_filenames = return_filenames
        self.image_paths = []

        if labels is not None:  # For training/validation
            for class_index in range(10):  # Assuming classes are named c0, c1, ..., c9
                class_dir = f"{self.root_dir}/c{class_index}"
                if os.path.exists(class_dir):
                    self.image_paths.extend([(f"{class_dir}/{img}", class_index)
                                             for img in os.listdir(class_dir)])
                else:
                    raise FileNotFoundError(f"Class directory not found: {class_dir}")
        else:  # For testing
            if os.path.exists(self.root_dir):
                self.image_paths = [(f"{self.root_dir}/{img}", None) for img in os.listdir(self.root_dir)]
            else:
                raise FileNotFoundError(f"Test directory not found: {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        img_path = img_path.replace("\\", "/")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.return_filenames:
            filename = os.path.basename(img_path)
            return (image, filename) if label is None else (image, label, filename)
        if label is not None:
            return image, torch.tensor(label, dtype=torch.long)
        return image

def load_datasets(train_path, test_path, batch_size=128, include_filenames=False):
    """Create DataLoaders for training, validation, and testing with ResNet-style augmentations."""
    # Training dataset transform with ResNet-style augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Random crop and resize to 224x224
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Random color jitter
        transforms.RandomRotation(degrees=15),  # Random rotation within 15 degrees
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for ResNet
    ])

    # Validation and test dataset transform
    eval_transform = transforms.Compose([
        transforms.Resize(256),  # Resize shorter side to 256
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Training dataset
    train_dataset = OnTheFlyDataset(root_dir=train_path, labels=True, transform=train_transform)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    val_dataset.dataset.transform = eval_transform  # Apply evaluation transform to validation set

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Test dataset
    if include_filenames:
        test_dataset = OnTheFlyDataset(root_dir=test_path, labels=None, transform=eval_transform, return_filenames=True)
    else:
        test_dataset = OnTheFlyDataset(root_dir=test_path, labels=None, transform=eval_transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader
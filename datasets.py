import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_datasets(train_images_path, train_labels_path, test_images_path, test_ids_path, batch_size=128):
    """
    Load preprocessed datasets and prepare DataLoaders for training, validation, and testing.

    Args:
        train_images_path (str): Path to preprocessed train images.
        train_labels_path (str): Path to preprocessed train labels.
        test_images_path (str): Path to preprocessed test images.
        test_ids_path (str): Path to test image IDs.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        train_loader, val_loader, test_images_tensor, test_ids (tuple): DataLoaders and test tensors.
    """
    # Load preprocessed data
    if all(os.path.exists(f) for f in [train_images_path, train_labels_path, test_images_path, test_ids_path]):
        X_train_all = np.load(train_images_path)
        y_train_all = np.load(train_labels_path)
        test_images = np.load(test_images_path)
        test_ids = np.load(test_ids_path)
        print("Loaded preprocessed data.")
    else:
        raise FileNotFoundError("Preprocessed data files not found. Run 'processed.py' first.")

    # Prepare training and validation datasets
    X_train_all = torch.tensor(X_train_all, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    y_train_all = torch.tensor(y_train_all, dtype=torch.long)
    dataset = TensorDataset(X_train_all, y_train_all)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Prepare test data
    test_images_tensor = torch.tensor(test_images, dtype=torch.float32).permute(0, 3, 1, 2)
    test_ids = test_ids

    return train_loader, val_loader, test_images_tensor, test_ids

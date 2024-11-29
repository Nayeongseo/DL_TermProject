import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm  # tqdm 추가
from model import AlexNet
from datasets import load_datasets
import numpy as np

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AlexNet Training and Testing")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--train_images', type=str, default='train_images.npy', help='Path to train images file')
    parser.add_argument('--train_labels', type=str, default='train_labels.npy', help='Path to train labels file')
    parser.add_argument('--test_images', type=str, default='test_images.npy', help='Path to test images file')
    parser.add_argument('--test_ids', type=str, default='test_ids.npy', help='Path to test IDs file')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for training and evaluation')
    return parser.parse_args()


def train(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm 사용하여 학습 상태 표시
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Accuracy 계산
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        progress_bar.set_postfix(loss=loss.item(), accuracy=f"{accuracy:.2f}%")

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy


def evaluate(model, val_loader, device, criterion):
    """Evaluate the model for one epoch."""
    model.eval()
    y_val_true = []
    y_val_pred = []
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm 사용하여 검증 상태 표시
    progress_bar = tqdm(val_loader, desc="Evaluating", unit="batch")
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Accuracy 계산
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            y_val_true.extend(labels.cpu().numpy())
            y_val_pred.extend(preds.cpu().numpy())

            accuracy = 100 * correct / total
            progress_bar.set_postfix(accuracy=f"{accuracy:.2f}%")

    val_loss = running_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy, y_val_true, y_val_pred

def predict_in_batches(model, test_images_tensor, batch_size, device):
    """Batch-wise prediction to avoid OOM errors."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_images_tensor), batch_size), desc="Predicting"):
            batch = test_images_tensor[i:i + batch_size].to(device)
            outputs = model(batch)
            batch_predictions = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions.append(batch_predictions)
    return np.vstack(predictions)

def main():
    args = parse_args()

    # Device setting
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load datasets
    train_loader, val_loader, test_images_tensor, test_ids = load_datasets(
        train_images_path=args.train_images,
        train_labels_path=args.train_labels,
        test_images_path=args.test_images,
        test_ids_path=args.test_ids,
        batch_size=args.batch_size
    )

    # Define model
    model = AlexNet(num_classes=10).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    

    # Train and evaluate for multiple epochs
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Train for one epoch
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Evaluate for one epoch
        val_loss, val_accuracy, y_val_true, y_val_pred = evaluate(model, val_loader, device, criterion)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        
    # Confusion Matrix
    cm = confusion_matrix(y_val_true, y_val_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.jpg')
    
    # Test predictions with batching
    test_predictions = predict_in_batches(model, test_images_tensor, batch_size=16, device=device)

    # Submission file
    submission = pd.DataFrame(test_predictions, columns=[f'c{i}' for i in range(10)])
    submission.insert(0, 'img', test_ids)
    # Extreme probability clipping
    submission.iloc[:, 1:] = np.clip(submission.iloc[:, 1:], 1e-15, 1 - 1e-15)

    submission.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")


if __name__ == '__main__':
    main()

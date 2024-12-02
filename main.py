import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from model import ResNet as Model
from datasets import load_datasets
import pandas as pd
import numpy as np


class EarlyStopping:
    def __init__(self, patience=3, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.best_accuracy = 0
        self.counter = 0
        self.stop = False

    def update(self, val_accuracy):
        if val_accuracy > self.best_accuracy + self.delta:
            self.best_accuracy = val_accuracy
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet Training and Testing")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--train_path', type=str, default='C:/Users/ASUS/Desktop/driver/imgs/train', help='Path to train images directory')
    parser.add_argument('--test_path', type=str, default='C:/Users/ASUS/Desktop/driver/imgs/test', help='Path to test images directory')
    parser.add_argument('--submission_file', type=str, default='submission.csv', help='Output submission file name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training and evaluation')
    return parser.parse_args()


def train(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # Mixed precision
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        progress_bar.set_postfix(loss=loss.item(), accuracy=f"{100 * correct / total:.2f}%")

    return running_loss / len(train_loader), 100 * correct / total


def evaluate(model, val_loader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    progress_bar = tqdm(val_loader, desc="Evaluating", unit="batch")
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return running_loss / len(val_loader), 100 * correct / total


def predict_and_save_results(model, test_loader, device, submission_file):
    """Predict on the test set and save results in the required format."""
    model.eval()
    results = []

    print("Predicting and preparing submission file...")
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

            # Create submission rows
            for prob, filename in zip(probabilities, filenames):
                row = {'img': filename}
                row.update({f'c{i}': prob[i] for i in range(10)})
                results.append(row)

    # Convert to DataFrame and save as CSV
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(submission_file, index=False)
    print(f"Submission file saved to {submission_file}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load datasets with optimized DataLoader
    train_loader, val_loader, test_loader = load_datasets(
    train_path=args.train_path,
    test_path=args.test_path,
    batch_size=args.batch_size,
    include_filenames=True
)

    # Define model
    model = Model(num_classes=10).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    best_val_accuracy = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Training loop
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, scaler)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation loop
        val_loss, val_accuracy = evaluate(model, val_loader, device, criterion)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model saved with accuracy: {best_val_accuracy:.2f}%")

        # Early stopping or 100% accuracy check
        if train_accuracy >= 100.0:
            print("Train accuracy reached 100%. Stopping training.")
            break
        if val_accuracy >= 100.0:
            print("Validation accuracy reached 100%. Stopping training.")
            break

        early_stopping.update(val_accuracy)
        if early_stopping.stop:
            print("Early stopping triggered. Stopping training.")
            break

    # Test predictions
    model.load_state_dict(torch.load("best_model.pth"))
    predict_and_save_results(model, test_loader, device, args.submission_file)


if __name__ == '__main__':
    main()

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from model import AlexNet
from torch.utils.data import DataLoader, Dataset, random_split
import os
import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import load_datasets

# Early Stopping 클래스
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")
        else:
            self.best_loss = val_loss
            self.counter = 0


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        """
        Args:
            num_classes (int): 총 클래스 수
            smoothing (float): 라벨 스무딩 정도 (0일 경우 일반 CrossEntropy와 동일)
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): 모델 예측값 (배치 크기, num_classes)
            target (Tensor): 실제 라벨 (배치 크기)
        """
        log_probs = F.log_softmax(pred, dim=-1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
        return loss
    
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AlexNet Training and Testing")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--device', type=str, default='mps', help='Device to use for training and evaluation')
    return parser.parse_args()


def train(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch with loss scaling."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    scale_factor = 128  # 손실 스케일링 팩터

    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 손실 스케일링 적용
        scaled_loss = loss * scale_factor
        scaled_loss.backward()
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

def predict_in_batches(model, test_loader, device):
    """-wise prediction to avoid OOM errors."""
    model.eval()
    predictions = []
    test_ids = []
    with torch.no_grad():
        for inputs, ids in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            batch_predictions = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions.append(batch_predictions)
            test_ids.extend(ids)
    return np.vstack(predictions), test_ids

def main():
    args = parse_args()

    # Device setting
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print("Using device:", device)

    # 데이터셋 로드
    train_path = '/Users/limgayoung/Downloads/driver/imgs/train'
    test_path = '/Users/limgayoung/Downloads/driver/imgs/test'
    train_loader, val_loader, test_loader = load_datasets(
        train_path=train_path,
        test_path=test_path,
        batch_size=args.batch_size
    )

    # 모델 정의
    model = AlexNet(num_classes=10).to(device)

    # 손실 함수와 옵티마이저 정의
    criterion = LabelSmoothingLoss(num_classes=10, smoothing=1e-1)  # Smoothing 값은 조정 가능
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    
    # 학습률 스케줄러 정의
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    early_stopping = EarlyStopping(patience=3, verbose=True)
    
    # 학습 및 검증 루프
    # 학습 및 검증 루프
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Evaluate
        val_loss, val_accuracy, y_val_true, y_val_pred = evaluate(model, val_loader, device, criterion)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # 스케줄러 업데이트 (학습률 감소)
        scheduler.step(val_loss)

        # Early Stopping 체크
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
        if train_accuracy >= 100.0:
            print("Stop")
            break
        if val_accuracy >= 100.0:
            print("Stop")
            break
        

    # Confusion Matrix
    cm = confusion_matrix(y_val_true, y_val_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.jpg')
    
    # Test predictions with batching
    test_predictions, test_ids = predict_in_batches(model, test_loader, device=device)

    # Submission file
    submission = pd.DataFrame(test_predictions, columns=[f'c{i}' for i in range(10)])
    submission.insert(0, 'img', test_ids)
    # Extreme probability clipping
    submission.iloc[:, 1:] = np.clip(submission.iloc[:, 1:], 1e-15, 1 - 1e-15)

    submission.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")

if __name__ == '__main__':
    main()

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
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import load_datasets


# Early Stopping 클래스
class EarlyStopping:
    """
    조기 종료를 위한 클래스.
    검증 손실이 개선되지 않을 경우 학습을 중단
    """
    def __init__(self, patience=5, verbose=False):
        self.patience = patience #개선되지 않은 epoch 수 허용치
        self.verbose = verbose #True일 경우 종료 메시지를 출력
        self.counter = 0 #개선되지 않은 epoch 수
        self.best_loss = None #차상의 검증 손실 값
        self.early_stop = False #조기 종료 여부

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss #첫 번째 epoch 검증
        elif val_loss > self.best_loss: #손실이 개선되지 않은 경우
            self.counter += 1
            if self.counter >= self.patience: #허용치 초과시 종료
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")
        else: #손실이 개선된 경우
            self.best_loss = val_loss
            self.counter = 0

# Label Smoothing 손실 함수
class LabelSmoothingLoss(nn.Module):
    """
    라벨 스무딩을 적용한 손실 함수.
    일반 CrossEntropyLoss 대비 더 나은 일반화 성능 제공
    """
    def __init__(self, num_classes, smoothing=0.1):
        """
        Args:
            num_classes (int): 총 클래스 수
            smoothing (float): 라벨 스무딩 정도 (0일 경우 일반 CrossEntropy와 동일)
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing #실제 라벨에 대한 confidence
        self.smoothing = smoothing #스무딩 정도
        self.num_classes = num_classes #클래스 개수

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): 모델 예측값 (배치 크기, num_classes)
            target (Tensor): 실제 라벨 (배치 크기)
        """
        log_probs = F.log_softmax(pred, dim=-1) #예측값에 log_softmax 적용
        true_dist = torch.zeros_like(log_probs) #초기화 된 라벨 분포
        true_dist.fill_(self.smoothing / (self.num_classes - 1)) #스무딩 값 할당
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence) #실제 라벨에 confidence 할당
        loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1)) #손실 계산
        return loss
       
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AlexNet Training and Testing")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--device', type=str, default='mps', help='Device to use for training and evaluation')
    return parser.parse_args()

#학습 함수
def train(model, train_loader, criterion, optimizer, device):
    """모델 학습 함수. 데이터 로더를 사용해 1 epoch 동안 학습 수행"""
    model.train() #학습 모드로 설정
    running_loss = 0.0 #누적 손실
    correct, total = 0, 0 #정확도 계산 변수
    scale_factor = 128  # 손실 스케일링 팩터

    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device) #데이터를 디바이스로 이동

        optimizer.zero_grad() #옵티마이저 초기화
        outputs = model(inputs) #모델 예측값 계산
        loss = criterion(outputs, labels) #손실 계산

        # 손실 스케일링 적용
        scaled_loss = loss * scale_factor
        scaled_loss.backward() #역전파
        optimizer.step() #가중치 업데이트

        running_loss += loss.item() #손실 누적

        # Accuracy 계산
        _, predicted = torch.max(outputs, 1) #가장 높은 확률의 클래스 선택
        total += labels.size(0) #전체 데이터 개수
        correct += (predicted == labels).sum().item() #맞춘 데이터 수 

        accuracy = 100 * correct / total #정확도 계산
        progress_bar.set_postfix(loss=loss.item(), accuracy=f"{accuracy:.2f}%")

    train_loss = running_loss / len(train_loader) #평균 손실
    train_accuracy = 100 * correct / total #최종 정확도
    return train_loss, train_accuracy

#검증 함수
def evaluate(model, val_loader, device, criterion):
    """
    모델 평가 함수
    검증 테이터에 대해 손실 및 정확도를 계산
    """
    model.eval() #평가모드로 설정
    y_val_true = [] #실제 라벨 저장
    y_val_pred = [] #예측 라벨 저장
    running_loss = 0.0 #초기화
    correct = 0
    total = 0

    # tqdm 사용하여 검증 상태 표시
    progress_bar = tqdm(val_loader, desc="Evaluating", unit="batch")
    with torch.no_grad(): #그래디언트 계산 비활성화
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) #모델 예측
            loss = criterion(outputs, labels) #손실 계산

            running_loss += loss.item() #손실 누적

            # Accuracy 계산
            _, preds = torch.max(outputs, 1) #예측값 계산
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            y_val_true.extend(labels.cpu().numpy()) #실제 라벨 저장
            y_val_pred.extend(preds.cpu().numpy()) #예측 라벨 저장

            accuracy = 100 * correct / total #정확도 계산
            progress_bar.set_postfix(accuracy=f"{accuracy:.2f}%") #진행 상황 출력

    val_loss = running_loss / len(val_loader) #평균 손실
    val_accuracy = 100 * correct / total #최종 정확도
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

#메인 함수
def main():
    """
    학습 및 검증 파이프라인을 실행하는 메인 함수
    """
    args = parse_args() #명령줄 인수 파싱

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
    model = AlexNet(num_classes=10).to(device) #알렉스넷 초기화
 
    # 손실 함수와 옵티마이저 정의
    criterion = LabelSmoothingLoss(num_classes=10, smoothing=1e-1)  #손실함수
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2) #옵티마이저
    
    # 학습률 스케줄러 정의
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    #조기 종료 설정
    early_stopping = EarlyStopping(patience=3, verbose=True)
    
    #학습 및 검증 루프
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        #학습
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        #검증
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
    
    # 배치를 통해 예측 테스트
    test_predictions, test_ids = predict_in_batches(model, test_loader, device=device)

    # 하위 권한 파일
    submission = pd.DataFrame(test_predictions, columns=[f'c{i}' for i in range(10)])
    submission.insert(0, 'img', test_ids)
    
    # 극단적인 확률 클리핑  
    submission.iloc[:, 1:] = np.clip(submission.iloc[:, 1:], 1e-15, 1 - 1e-15)

    submission.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")

if __name__ == '__main__':
    main()

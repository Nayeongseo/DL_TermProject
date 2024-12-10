import torch.nn as nn
from torchvision.models import alexnet

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Squeeze-and-Excitation Block.
        입력 특징 맵에 대한 채널별 중요도를 학습하고
        이를 이용해 특징 맵을 재조정하는 블록

        Args:
            channels (int): 입력 채널 수
            reduction (int): 채널 축소 비율(기본값: 16)
        """
        super(SEBlock, self).__init__()
        
        #Global Average Polling: 입력 특징 맵을 채널별 평균값으로 압축
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        #Fully Connected Layer를 통해 채널 중요도를 학습
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True), #활성화 함수
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid() #중요도를 0과 1사이 값으로 조정
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): 입력 텐서(batch_size, channels, height, width)

        Returns:
            Tensor: 채널별 중요도가 조정된 출력 텐서
        """
        batch, channels, _, _ = x.size()
        # Global Average Pooling
        y = self.global_avg_pool(x).view(batch, channels)
        # 중요도 계산
        y = self.fc(y).view(batch, channels, 1, 1)
        # 입력과 중요도를 곱하여 출력
        return x * y

class AlexNet(nn.Module):
    """
    Squeeze-and-Excitation Block을 추가한 AlexNet 모델
    """
    def __init__(self, num_classes=10, reduction=16):
        """
        Args:
            num_classes (int): 분류 클래스 수(기본값: 16)
            reduction (int): SEBlock의 채널 축소 비율
        """
        super(AlexNet, self).__init__()
        
        # 사전학습된 AlexNet 로드
        pretrained_model = alexnet(weights="AlexNet_Weights.IMAGENET1K_V1")
        
        # 특징 추출기: SEBlock을 포함하도록 수정
        self.features = nn.Sequential(
            nn.Sequential(
                pretrained_model.features[0],  # 첫번째 Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
                pretrained_model.features[1],  # ReLU
                pretrained_model.features[2],  # MaxPool2d(kernel_size=3, stride=2)
                SEBlock(channels=64, reduction=reduction) #SEBlock 추가
            ),
            nn.Sequential(
                pretrained_model.features[3],  # 두번째 Conv2d(64, 192, kernel_size=5, padding=2)
                pretrained_model.features[4],  # ReLU
                pretrained_model.features[5],  # MaxPool2d(kernel_size=3, stride=2)
                SEBlock(channels=192, reduction=reduction) #SEBlock 추가
            ),
            nn.Sequential(
                pretrained_model.features[6],  # Conv2d(192, 384, kernel_size=3, padding=1)
                pretrained_model.features[7],  # ReLU
                SEBlock(channels=384, reduction=reduction)
            ),
            nn.Sequential(
                pretrained_model.features[8],  # 세번째 Conv2d(384, 256, kernel_size=3, padding=1)
                pretrained_model.features[9],  # ReLU
                SEBlock(channels=256, reduction=reduction) #SEBlock 추가
            ),
            nn.Sequential(
                pretrained_model.features[10],  # 네번째 Conv2d(256, 256, kernel_size=3, padding=1)
                pretrained_model.features[11],  # ReLU
                pretrained_model.features[12],  # MaxPool2d(kernel_size=3, stride=2)
                SEBlock(channels=256, reduction=reduction) #SEBlock 추가
            )
        )

        #분류기: Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Flatten(), #특징 맵을 1차원 벡터로 전환
            nn.Linear(256 * 6 * 6, 4096), #첫번째 Fully connected layer
            nn.BatchNorm1d(4096), #배치 정규화
            nn.ReLU(inplace=True), #활성화 함수
            nn.Dropout(0.5), #드롭아웃
            
            nn.Linear(4096, 4096), #두번째 Fully connected layer
            nn.BatchNorm1d(4096), #배치 정규화
            nn.ReLU(inplace=True), #활성화 함수
            nn.Dropout(0.5), #드롭아웃
            
            nn.Linear(4096, num_classes) #출력층(클래스 개수)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): 입력 텐서(batch_size, channels, height, width)

        Returns:
            Tensor: 클래스별 점수를 나타내는 출력 텐서
        """
        #특징 추출기를 거쳐 특징 맵 생성
        for block in self.features:
            x = block(x)
        #분류기를 통해 최종 클래스 점수 생성
        x = self.classifier(x)
        return x

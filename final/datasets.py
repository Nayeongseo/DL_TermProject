import os
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from torchvision import transforms

#사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        """
        사용자 정의 데이터셋 클래스
        운전자의 이미지를 로드하고 데이터 증강 적용
        Args:
            path (str): 데이터셋 경로
            train (bool): True면 학습 데이터, False면 테스트 데이터
            transform (callable, optional): 이미지에 적용할 변환 함수
        """
        self.image_paths = [] #이미지 파일 경로 리스트
        self.labels = [] #이미지 라벨 리스트
        self.transform = transform #데이터 증강 함수

        if train:
            #학습 데이터 로드
            for class_index in range(10): #클래스: c0~c9
                class_path = os.path.join(path, f'c{class_index}')
                if not os.path.exists(class_path):
                    raise FileNotFoundError(f"Class folder '{class_path}' not found.")
                
                files = os.listdir(class_path) #클래스 디렉토리 내 파일리스트
                for filename in files:
                    self.image_paths.append(os.path.join(class_path, filename)) #이미지 경로 추가
                    self.labels.append(class_index) #라벨 추가
        else:
            #테스트 데이터 로드
            self.image_paths = os.listdir(path)
            self.labels = None #테스트 데이터는 라벨이 없음
        self.train = train #학습 데이터 여부 플래그
        self.path = path #데이터 경로

    def __len__(self):
        """
        데이터셋의 크기를 반환
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        데이터셋에서 특정 인덱스에 해당하는 데이터 반환

        Args:
            idx (int): 데이터 인덱스

        Returns:
            tuple: 이미지, 라벨
        """
        if self.train:
            image_path = self.image_paths[idx] #학습 이미지 경로
            label = self.labels[idx] #학습 이미지 라벨
        else:
            image_path = os.path.join(self.path, self.image_paths[idx]) #테스트 이미지 경로
            label = self.image_paths[idx] #테스트 파일명 

        img = Image.open(image_path).convert('RGB') #이미지를 RGB로 로드

        # 데이터 증강 적용
        if self.transform:
            img = self.transform(img)

        if self.train:
            return img, label #학습 데이터
        else:
            return img, label #테스트데이터

#데이터 로드 및 DataLoader 생성 함수
def load_datasets(train_path, test_path, batch_size=128):
    """
    학습, 검증, 테스트 데이터셋을 로드하고 DataLoader를 생성
    
    Args:
        train_path (str): 학습 데이터셋 경로
        test_path (str): 테스트 데이터셋 경로
        batch_size (int): DataLoader 배치 크기

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    #학습 데이터에 적용할 데이터 증강 변환 정의
    train_transform = transforms.Compose([
        transforms.Resize((227, 227)), #이미지 크기 조정
        transforms.RandomHorizontalFlip(p=0.5), #랜덤 수평 뒤집기
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05), #밝기, 대비, 채도 조정
        transforms.RandomRotation(degrees=15), #랜덤 회전
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5), #랜덤 투시 변환
        transforms.ToTensor(), #텐서 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #이미지 정규화
    ])
    
    #검증 및 테스트 데이터에 적용할 변환 정의
    val_transform = transforms.Compose([
        transforms.Resize((227, 227)), #이미지 크기 조정
        transforms.ToTensor(), #텐서로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #이미지 정규화
    ])

    test_transform = val_transform #테스트 데이터 변환은 검증 데이터 변환과 동일

    #학습 데이터셋 생성
    train_dataset = CustomDataset(train_path, train=True, transform=train_transform)

    #학습 데이터 셋을 학습/검증으로 분할
    train_size = int(0.8 * len(train_dataset)) #학습데이터 크기: 전체의 80%
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    #검증 데이터에 변환 적용
    val_dataset.dataset.transform = val_transform

    #DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    #테스트 데이터셋 및 DataLoader 생성
    test_dataset = CustomDataset(test_path, train=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    #데이터셋 크기 출력
    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples, and {len(test_dataset)} test samples.")
    return train_loader, val_loader, test_loader

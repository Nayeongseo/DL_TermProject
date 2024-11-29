import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# 경로 설정
train_path = '/workspace/T_NET/driver/state-farm-distracted-driver-detection/imgs/train'
test_path = '/workspace/T_NET/driver/state-farm-distracted-driver-detection/imgs/test'
train_images_file = 'train_images.npy'
train_labels_file = 'train_labels.npy'
test_images_file = 'test_images.npy'
test_ids_file = 'test_ids.npy'

# 데이터 전처리 및 저장
def preprocess_and_save():
    # 학습 데이터 전처리
    class_labels = []
    images = []

    for class_index in range(10):
        class_path = os.path.join(train_path, f'c{class_index}')
        files = os.listdir(class_path)
        print(f"Class {class_index}에 대한 파일 수: {len(files)}")

        for filename in tqdm(files, desc=f'Processing class {class_index}'):
            image_path = os.path.join(class_path, filename)
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize((227, 227))  # Resize
                img_array = np.array(img) / 255.0  # Normalize to 0-1
                images.append(img_array)
                class_labels.append(class_index)
            except Exception as e:
                print(f"이미지 처리 중 오류 발생: {image_path}, 오류: {e}")

    # numpy 배열로 변환
    images = np.array(images)
    class_labels = np.array(class_labels)
    print(f"학습 데이터: 총 이미지 수: {len(images)}, 총 라벨 수: {len(class_labels)}")

    # 학습 데이터 저장
    np.save(train_images_file, images)
    np.save(train_labels_file, class_labels)
    print("학습 데이터가 저장되었습니다.")

    # 테스트 데이터 전처리
    test_images = []
    test_ids = []

    for filename in tqdm(os.listdir(test_path), desc='Processing test images'):
        image_path = os.path.join(test_path, filename)
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((227, 227))  # Resize
            img_array = np.array(img) / 255.0  # Normalize to 0-1
            test_images.append(img_array)
            test_ids.append(filename)
        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {image_path}, 오류: {e}")

    # numpy 배열로 변환
    test_images = np.array(test_images)
    test_ids = np.array(test_ids)
    print(f"테스트 데이터: 총 이미지 수: {len(test_images)}")

    # 테스트 데이터 저장
    np.save(test_images_file, test_images)
    np.save(test_ids_file, test_ids)
    print("테스트 데이터가 저장되었습니다.")

if __name__ == "__main__":
    preprocess_and_save()

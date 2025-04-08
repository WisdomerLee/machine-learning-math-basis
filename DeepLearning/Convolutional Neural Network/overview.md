# CNN의 핵심 개념 정리

## 1. CNN이란?

- 이미지나 영상 같은 **공간적인 특징(패턴)**을 효과적으로 학습하는 신경망
- 주로 **이미지 분류, 객체 탐지, 얼굴 인식** 등에 활용됨

---

## 2. 주요 구성 요소

| 구성 요소 | 설명 |
|-----------|------|
| **Convolution Layer (합성곱 층)** | 필터(또는 커널)를 사용해 이미지의 특징 추출 |
| **ReLU (활성화 함수)** | 비선형성 부여 (보통 ReLU 사용) |
| **Pooling Layer (풀링층)** | 이미지 크기를 줄여 계산량 감소 (보통 MaxPooling 사용) |
| **Fully Connected Layer (완전 연결층)** | 마지막 단계에서 분류기 역할 수행 |

---

## CNN 구조 예시

```
[입력 이미지] 
 → [Conv → ReLU → Pool] 
 → [Conv → ReLU → Pool] 
 → [Flatten] 
 → [Fully Connected Layer] 
 → [출력]
```

---

# PyTorch로 구현해보기 (MNIST 숫자 분류 예시)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# 1. 하이퍼파라미터 설정
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# 2. 데이터셋 로드 (MNIST)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 3. CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 입력 채널 1개, 출력 채널 16개
        self.pool = nn.MaxPool2d(2, 2)                           # 2x2 풀링
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 출력 채널 32개
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28x28 → 14x14
        x = self.pool(F.relu(self.conv2(x)))   # 14x14 → 7x7
        x = x.view(-1, 32 * 7 * 7)             # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                        # 10 클래스 출력
        return x

# 4. 모델, 손실 함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 5. 학습 루프
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 6. 테스트 정확도 확인
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')
```

---

# 핵심 정리

- CNN은 이미지의 **공간 구조(위치, 패턴)**를 잘 인식하는 구조
- `Conv2d`, `ReLU`, `MaxPool2d`를 반복하며 특징을 추출
- 마지막은 `Linear` 층으로 분류 수행
- `MNIST`처럼 흑백 이미지(1채널)는 `Conv2d(1, ...)`로 시작

---

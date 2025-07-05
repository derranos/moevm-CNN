import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import time
from torch.amp import autocast, GradScaler

# Устройство и проверка CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}, CUDA version: {torch.version.cuda}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    torch.cuda.empty_cache()  # Очистка памяти GPU
else:
    print("Warning: Running on CPU, install PyTorch with CUDA for faster training")

# Путь к Imagenette2-160
target_dir = "./imagenette2-160"

# Маппинг классов Imagenette к ImageNet
imagenette_to_imagenet = {
    0: 0, 1: 207, 2: 485, 3: 509, 4: 497, 5: 566, 6: 569, 7: 571, 8: 574, 9: 701
}

# Трансформации
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Загрузка датасетов
train_dataset = datasets.ImageFolder(os.path.join(target_dir, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(target_dir, 'val'), transform=val_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Диагностика маппинга и данных
print("Val class to index mapping:", val_dataset.class_to_idx)
print("Imagenette to ImageNet mapping:", imagenette_to_imagenet)
for images, _ in train_loader:
    print("Train data shape:", images.shape, "Min:", images.min().item(), "Max:", images.max().item())
    break
for images, _ in val_loader:
    print("Val data shape:", images.shape, "Min:", images.min().item(), "Max:", images.max().item())
    break

# Загрузка модели
model = models.efficientnet_v2_m(weights='IMAGENET1K_V1').to(device)
print("Model: EfficientNetV2-M loaded with IMAGENET1K_V1 weights")

# Оптимизатор, функция потерь и mixed precision
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler('cuda') if device == "cuda" else None

# Дообучение
model.train()
num_epochs = 1
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        labels = torch.tensor([imagenette_to_imagenet[label.item()] for label in labels]).to(device)
        optimizer.zero_grad()
        if device == "cuda":
            with autocast('cuda'):  # Mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 10:.4f}")
            running_loss = 0.0
print(f"Finished training for {num_epochs} epochs")

# Очистка памяти перед инференсом
if device == "cuda":
    torch.cuda.empty_cache()
    print(f"GPU memory allocated after training: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

# Инференс на валидационной выборке
model.eval()
start_time = time.time()
predictions = []
with torch.no_grad():
    for images, _ in val_loader:
        images = images.to(device)
        if device == "cuda":
            with autocast('cuda'):
                outputs = model(images)
        else:
            outputs = model(images)
        pred_classes = torch.argmax(outputs, dim=1).cpu().numpy()
        predictions.extend(pred_classes)
inference_time = time.time() - start_time

# Преобразование меток
true_labels = [imagenette_to_imagenet[label] for label in val_dataset.targets]

# Точность
accuracy = np.mean(np.array(predictions) == np.array(true_labels))

# Сохранение модели
torch.save(model.state_dict(), "efficientnetv2m_finetuned.pth")

print(f"Top-1 Accuracy: {accuracy:.4f}, Inference Time: {inference_time:.2f} seconds")
print("Model saved to: efficientnetv2m_finetuned.pth")
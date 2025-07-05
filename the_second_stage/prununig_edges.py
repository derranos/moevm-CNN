import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.utils.prune as prune
import time
from torch.amp import autocast

# Пользовательский метод прунинга
class ThresholdPruning(prune.BasePruningMethod):
    def __init__(self, threshold):
        self.threshold = threshold
    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) >= self.threshold

# Устройство
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"  
# Путь к модели и датасету
model_path = "efficientnetv2m_finetuned.pth"
target_dir = "./imagenette2-160"
imagenette_to_imagenet = {0: 0, 1: 207, 2: 485, 3: 509, 4: 497, 5: 566, 6: 569, 7: 571, 8: 574, 9: 701}

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Загрузка тестового набора
test_dataset = datasets.ImageFolder(os.path.join(target_dir, 'val'), transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Загрузка модели
model = models.efficientnet_v2_m()
model.load_state_dict(torch.load(model_path, weights_only=True))
model = model.to(device)
model.eval()

# Очистка памяти GPU
if device == "cuda":
    torch.cuda.empty_cache()

def count_pruned_weights(model):
    total = 0
    pruned = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data.cpu().numpy()
            total += weight.size
            pruned += np.sum(weight == 0)
    return pruned, total

# --- ДОБАВЛЕНО: Инференс и метрики до прунинга ---
start_time = time.time()
predictions_before = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        with autocast('cuda') if device == "cuda" else torch.no_grad():
            outputs = model(images)
        pred_classes = torch.argmax(outputs, dim=1).cpu().numpy()
        predictions_before.extend(pred_classes)
inference_time_before = time.time() - start_time

true_labels = [imagenette_to_imagenet[label] for label in test_dataset.targets]
accuracy_before = np.mean(np.array(predictions_before) == np.array(true_labels))

# --- Прунинг с ограничением слоёв ---
threshold = 0.15
k = 0
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        ThresholdPruning.apply(module, 'weight', threshold=threshold)
        prune.remove(module, 'weight')
        k += 1
        if k % 2 == 1:
            threshold *= 0.5

# Очистка памяти GPU
if device == "cuda":
    torch.cuda.empty_cache()

# --- Инференс после прунинга ---
start_time = time.time()
predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        with autocast('cuda') if device == "cuda" else torch.no_grad():
            outputs = model(images)
        pred_classes = torch.argmax(outputs, dim=1).cpu().numpy()
        predictions.extend(pred_classes)
inference_time = time.time() - start_time

accuracy = np.mean(np.array(predictions) == np.array(true_labels))

# --- Подсчет количества обнуленных весов ---
pruned_count, total_count = count_pruned_weights(model)
pruned_percent = 100 * pruned_count / total_count if total_count > 0 else 0

# --- Сохранение ---
pruned_model_path = "efficientnetv2m_pruned_threshold.pth"
with open("results_pruned.txt", "w") as f:
    f.write(f"Top-1 Accuracy (before pruning): {accuracy_before:.4f}\nInference Time (before pruning): {inference_time_before:.2f} seconds\n")
    f.write(f"Top-1 Accuracy (pruned): {accuracy:.4f}\nInference Time (pruned): {inference_time:.2f} seconds\n")
    f.write(f"Pruned weights: {pruned_percent:.2f}% ({pruned_count}/{total_count})\n")
torch.save(model.state_dict(), pruned_model_path)

print(f"Pruned model saved to: {pruned_model_path}")
print(f"Results saved to: results_pruned.txt")
print(f"Top-1 Accuracy (before pruning): {accuracy_before:.4f}, Inference Time (before pruning): {inference_time_before:.2f} seconds")
print(f"Top-1 Accuracy (pruned): {accuracy:.4f}, Inference Time (pruned): {inference_time:.2f} seconds")
print(f"Pruned weights: {pruned_percent:.2f}% ({pruned_count}/{total_count})")
import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import time

# Устройство
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

# Прунинг нейронов (фильтров) по L1-норме
def prune_neurons(model, prune_ratio=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Вычисление L1-нормы каждого фильтра
            l1_norm = torch.norm(module.weight.data, p=1, dim=(1, 2, 3))
            num_filters = module.out_channels
            num_pruned = int(num_filters * prune_ratio)
            if num_pruned > 0:
                # Определяем индексы фильтров для сохранения
                _, indices = torch.topk(l1_norm, num_filters - num_pruned, largest=True)
                keep_indices = indices.sort()[0]
                # Обнуляем фильтры, которые не вошли в keep_indices
                for i in range(num_filters):
                    if i not in keep_indices:
                        module.weight.data[i] = 0
    return model

print("Start pruning")
pruned_model = prune_neurons(model, prune_ratio=0.05)

# Инференс после прунинга
start_time = time.time()
predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = pruned_model(images)
        pred_classes = torch.argmax(outputs, dim=1).cpu().numpy()
        predictions.extend(pred_classes)
inference_time = time.time() - start_time

# Преобразование меток
true_labels = [imagenette_to_imagenet[label] for label in test_dataset.targets]

# Расчёт точности
accuracy = np.mean(np.array(predictions) == np.array(true_labels))

# Подсчёт удалённых нейронов
def count_pruned_neurons(model):
    total_neurons = 0
    pruned_neurons = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            total_neurons += module.out_channels
            weights = module.weight.data
            # Фильтр считается удалённым, если все его веса равны нулю
            pruned_neurons += sum(torch.all(weights[i] == 0) for i in range(module.out_channels))
    return pruned_neurons, total_neurons

pruned_neurons, total_neurons = count_pruned_neurons(pruned_model)
pruned_neurons_percent = 100 * pruned_neurons / total_neurons if total_neurons > 0 else 0

# Сохранение результатов
pruned_model_path = "efficientnetv2m_pruned_neurons.pth"
with open("results_pruned_neurons.txt", "w") as f:
    f.write(f"Top-1 Accuracy (pruned): {accuracy:.4f}\n")
    f.write(f"Inference Time (pruned): {inference_time:.2f} seconds\n")
    f.write(f"Pruned neurons: {pruned_neurons_percent:.2f}% ({pruned_neurons}/{total_neurons})\n")
torch.save(pruned_model.state_dict(), pruned_model_path)

print(f"Pruned model saved to: {pruned_model_path}")
print(f"Results saved to: results_pruned_neurons.txt")
print(f"Top-1 Accuracy (pruned): {accuracy:.4f}, Inference Time (pruned): {inference_time:.2f} seconds")
print(f"Pruned neurons: {pruned_neurons_percent:.2f}% ({pruned_neurons}/{total_neurons})")
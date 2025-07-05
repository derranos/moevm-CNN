import os
from torchvision.datasets.utils import download_and_extract_archive

# Куда скачать
target_dir = "./imagenette2-160"

if not os.path.exists(target_dir):
    print("Скачиваем Imagenette...")
    download_and_extract_archive(
        url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
        download_root=".",
        filename="imagenette2-160.tgz"
    )
    print("Готово!")
else:
    print("Imagenette уже скачан.")

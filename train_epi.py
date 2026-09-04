# %%
# Célula 1 — Patch do torch.load e confirmação da GPU
import torch

_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

print("CUDA disponível:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# %%
# Célula 2 — Treinamento com GPU 
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    results = model.train(
        data="dataset/exports/epi-v1/data.yaml",
        epochs=50,
        imgsz=640,
        device=0,
        project="runs",
        name="epi-v1",
        workers=0,
    )
    print("Pesos salvos em:", results.save_dir)

# %%
# Célula 3 — Métricas finais (mAP, precisão, recall)
metrics = model.val(data="dataset/exports/epi-v1/data.yaml", workers=0)
print("mAP@0.5:      ", metrics.box.map50)
print("mAP@0.5:0.95: ", metrics.box.map)
print("Precisão:     ", metrics.box.mp)
print("Recall:       ", metrics.box.mr)

# %%
# Célula 4 — Exportar imagem de resultado com bounding boxes 
import glob
val_images = glob.glob("dataset/exports/epi-v1/valid/images/*.jpg") + \
             glob.glob("dataset/exports/epi-v1/valid/images/*.png")
if not val_images:
    raise FileNotFoundError("Nenhuma imagem encontrada em valid/images/")

sample_image = val_images[0]
print("Imagem escolhida para teste:", sample_image)

detect_results = model.predict(source=sample_image, save=True, project="runs", name="epi-v1-predict")
print("Imagem anotada salva em:", detect_results[0].save_dir)

# %%

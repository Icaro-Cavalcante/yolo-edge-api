"""
preprocessing/utils/evaluate.py

Função de avaliação de pipeline de pré-processamento para YOLOv8.
Grava os frames pré-processados em disco antes de chamar model.val() para garantir
que o efeito das transformações (espaço de cor, resize, filtros, equalização)
seja medido de forma precisa pelo validador do Ultralytics.
"""
from pathlib import Path
import shutil
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import yaml

# Patch para PyTorch 2.6+ (desativa weights_only temporariamente)
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from ultralytics import YOLO


def evaluate_pipeline(
    preprocess_fn: Optional[Callable[[np.ndarray], any]] = None,
    label: str = "eval",
    data_yaml: str = "dataset/exports/epi-v1/data.yaml",
    model_path: str = "models/yolov8n.pt",
    imgsz: int = 320,
    batch: int = 1,
    device: str = "cpu",
) -> float:
    """
    Avalia um pipeline de pré-processamento no dataset de validação.

    Args:
        preprocess_fn: Função que recebe um frame BGR (np.ndarray) e retorna
                       um np.ndarray ou PreprocessResult. Se None, usa o frame original.
        label: Identificador único do experimento.
        data_yaml: Caminho para o data.yaml base.
        model_path: Caminho dos pesos YOLOv8.
        imgsz: Tamanho de inferência para validação.
        batch: Batch size.
        device: Device de execução ('cpu' ou int de GPU).

    Returns:
        map50: Valor float de mAP@0.5 medido no conjunto de validação.
    """
    yaml_path = Path(data_yaml).resolve()
    with open(yaml_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    dataset_root = Path(base_cfg.get("path", yaml_path.parent))
    val_images_rel = base_cfg.get("val", "valid/images")
    val_images_dir = (dataset_root / val_images_rel).resolve()

    if "images" in val_images_dir.parts:
        val_labels_dir = val_images_dir.parent / "labels"
    else:
        val_labels_dir = dataset_root / "valid" / "labels"

    tmp_base = Path("preprocessing/outputs/_tmp_eval").resolve() / label
    tmp_images_dir = tmp_base / "images"
    tmp_labels_dir = tmp_base / "labels"

    if tmp_base.exists():
        shutil.rmtree(tmp_base)
    tmp_images_dir.mkdir(parents=True, exist_ok=True)
    tmp_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [p for p in val_images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
    )
    if not image_files:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {val_images_dir}")

    for img_path in image_files:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        orig_h, orig_w = img_bgr.shape[:2]
        res = preprocess_fn(img_bgr) if preprocess_fn is not None else img_bgr

        if hasattr(res, "frame"):
            frame = res.frame
            scale = getattr(res, "scale", 1.0)
            pad_w = getattr(res, "pad_w", 0)
            pad_h = getattr(res, "pad_h", 0)
        else:
            frame = res
            scale = 1.0
            pad_w = 0
            pad_h = 0

        if frame.dtype != np.uint8:
            frame_to_save = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        else:
            frame_to_save = frame

        out_dest = tmp_images_dir / img_path.name
        cv2.imwrite(str(out_dest), frame_to_save)

        lbl_source = val_labels_dir / (img_path.stem + ".txt")
        lbl_dest = tmp_labels_dir / (img_path.stem + ".txt")
        if lbl_source.exists():
            if pad_w > 0 or pad_h > 0:
                infer_size = max(frame_to_save.shape[0], frame_to_save.shape[1])
                new_lines = []
                with open(lbl_source, "r") as f_in:
                    for line in f_in:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = parts[0]
                            xc, yc, w, h = map(float, parts[1:5])
                            abs_xc = xc * orig_w * scale + pad_w
                            abs_yc = yc * orig_h * scale + pad_h
                            abs_w = w * orig_w * scale
                            abs_h = h * orig_w * scale
                            new_lines.append(
                                f"{cls_id} {abs_xc / infer_size:.6f} {abs_yc / infer_size:.6f} {abs_w / infer_size:.6f} {abs_h / infer_size:.6f}"
                            )
                with open(lbl_dest, "w") as f_out:
                    f_out.write(chr(10).join(new_lines) + chr(10))
            else:
                shutil.copy(str(lbl_source), str(lbl_dest))

    tmp_yaml = tmp_base / "data.yaml"
    tmp_cfg = {
        "path": str(tmp_base),
        "train": "images",
        "val": "images",
        "test": "images",
        "names": base_cfg["names"],
    }
    with open(tmp_yaml, "w") as f:
        yaml.safe_dump(tmp_cfg, f)

    model = YOLO(model_path)
    val_results = model.val(
        data=str(tmp_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
        save=False,
        plots=False,
    )

    map50 = float(val_results.box.map50)
    print(f"[{label}] mAP@0.5: {map50:.4f}")
    return map50

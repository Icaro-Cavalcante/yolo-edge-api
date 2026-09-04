"""
preprocessing/experiments/e4_generate_dark.py

Passo 7: Gera versão escurecida do dataset de validação para simular
condições adversas de iluminação (subexposição/ambiente escuro).
Salva em dataset/exports/epi-v1-dark/valid/images e copia as labels correspondentes.
"""
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def generate_dark_dataset(factor: float = 0.35):
    src_dir = Path("dataset/exports/epi-v1/valid").resolve()
    dst_dir = Path("dataset/exports/epi-v1-dark/valid").resolve()

    images_src = src_dir / "images"
    labels_src = src_dir / "labels"

    images_dst = dst_dir / "images"
    labels_dst = dst_dir / "labels"

    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    img_files = list(images_src.glob("*.jpg")) + list(images_src.glob("*.png"))
    print(f"[e4_generate_dark] Escurecendo {len(img_files)} imagens (fator {factor})...")

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        # Aplica atenuação de brilho simulando baixa luminosidade
        dark = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        cv2.imwrite(str(images_dst / img_path.name), dark)

        lbl_file = labels_src / (img_path.stem + ".txt")
        if lbl_file.exists():
            shutil.copy(str(lbl_file), str(labels_dst / lbl_file.name))

    print(f"[e4_generate_dark] Dataset escurecido gerado com sucesso em: {dst_dir}")


if __name__ == "__main__":
    generate_dark_dataset()

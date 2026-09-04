"""
preprocessing/experiments/e1_visualize.py

Passo 4: Visualização comparativa dos canais e do efeito BGR vs RGB.
Salva a figura em preprocessing/outputs/e1_visualize.png.
"""
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def main():
    val_dir = Path("dataset/exports/epi-v1/valid/images")
    sample_imgs = list(val_dir.glob("*.jpg"))
    if not sample_imgs:
        print("Nenhuma imagem encontrada para visualização.")
        return

    sample_path = sample_imgs[0]
    bgr = cv2.imread(str(sample_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    out_dir = Path("preprocessing/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "e1_visualize.png"

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Linha 1: Imagem como BGR (interpretada como RGB pelo visualizador) vs Imagem convertida RGB
    axes[0, 0].imshow(bgr)
    axes[0, 0].set_title("Interpretado direto como BGR (Canais Trocados)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb)
    axes[0, 1].set_title("Convertido BGR -> RGB (Cores Reais)")
    axes[0, 1].axis("off")

    diff = cv2.absdiff(bgr, rgb)
    axes[0, 2].imshow(diff)
    axes[0, 2].set_title("Diferença Absoluta |BGR - RGB|")
    axes[0, 2].axis("off")

    # Linha 2: Canais R, G, B individuais
    axes[1, 0].imshow(rgb[:, :, 0], cmap="Reds")
    axes[1, 0].set_title("Canal R (Vermelho / Laranja)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(rgb[:, :, 1], cmap="Greens")
    axes[1, 1].set_title("Canal G (Verde)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(rgb[:, :, 2], cmap="Blues")
    axes[1, 2].set_title("Canal B (Azul)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Visualização salva em: {out_path}")


if __name__ == "__main__":
    main()

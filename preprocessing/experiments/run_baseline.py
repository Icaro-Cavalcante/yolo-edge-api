"""
preprocessing/experiments/run_baseline.py

Passo 3: Avalia o baseline sem pré-processamento adicional no dataset epi-v1.
Referência para o cálculo de deltas em todos os experimentos subsequentes.
"""
from pathlib import Path
import sys

# Garante raiz do projeto no path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.utils.evaluate import evaluate_pipeline


def main():
    print("=" * 60)
    print("Executando Baseline Geral (epi-v1, yolov8n.pt, imgsz=320)")
    print("=" * 60)

    # Avaliação do baseline sem pré-processamento
    map50 = evaluate_pipeline(
        preprocess_fn=None,
        label="baseline",
        data_yaml="dataset/exports/epi-v1/data.yaml",
        model_path="models/yolov8n.pt",
        imgsz=320,
    )

    print("-" * 60)
    print(f"Resultado Baseline mAP@0.5: {map50:.4f}")
    print("=" * 60)
    return map50


if __name__ == "__main__":
    main()

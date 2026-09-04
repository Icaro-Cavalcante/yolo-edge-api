"""
preprocessing/experiments/e2_resize.py

Passo 5: Avaliação do redimensionamento: Resize ingênuo vs Letterbox.
- E2-A: resize ingênuo (use_letterbox=False)
- E2-B: letterbox (use_letterbox=True)
Demonstração da transformação matemática de coordenadas com demo_bbox_adjustment().
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.preprocessor import Preprocessor, PreprocessConfig
from preprocessing.utils.evaluate import evaluate_pipeline


def demo_bbox_adjustment():
    print()
    print("-" * 60)
    print("DEMONSTRAÇÃO DE AJUSTE DE COORDENADAS (demo_bbox_adjustment)")
    print("-" * 60)

    # Simula um frame retangular de câmera 640x480 sendo processado com letterbox para 320x320
    orig_h, orig_w = 480, 640
    frame_fake = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    cfg = PreprocessConfig(infer_size=320, use_letterbox=True)
    pp = Preprocessor(cfg)
    res = pp.process(frame_fake)

    print(f"Resolução original : {orig_w}x{orig_h}")
    print(f"infer_size         : {cfg.infer_size}")
    print(f"scale              : {res.scale:.4f}")
    print(f"pad_w              : {res.pad_w} px")
    print(f"pad_h              : {res.pad_h} px")

    # Suponha uma detecção no espaço letterboxed (320x320): [x1, y1, x2, y2]
    bbox_letterbox = np.array([[50.0, 70.0, 270.0, 250.0]])
    bbox_original = pp.adjust_boxes(bbox_letterbox, res)

    print(f"BBox antes (espaço letterbox 320x320) : {bbox_letterbox[0].tolist()}")
    print(f"BBox depois (espaço original 640x480) : {[round(v, 2) for v in bbox_original[0].tolist()]}")
    print("-" * 60)
    print()


def main():
    print("=" * 60)
    print("EXPERIMENTO E2: REDIMENSIONAMENTO (Resize Ingênuo vs Letterbox)")
    print("=" * 60)

    # E2-A: Resize ingênuo
    cfg_a = PreprocessConfig(infer_size=320, convert_rgb=True, use_letterbox=False)
    pp_a = Preprocessor(cfg_a)
    map_a = evaluate_pipeline(pp_a.process, label="e2_a_resize")

    # E2-B: Letterbox correto
    cfg_b = PreprocessConfig(infer_size=320, convert_rgb=True, use_letterbox=True)
    pp_b = Preprocessor(cfg_b)
    map_b = evaluate_pipeline(pp_b.process, label="e2_b_letterbox")

    print("-" * 60)
    print(f"E2-A: resize ingênuo : mAP@0.5 = {map_a:.4f}")
    print(f"E2-B: letterbox      : mAP@0.5 = {map_b:.4f}")
    print(f"Delta (E2-B - E2-A)  : {map_b - map_a:+.4f}")
    print("=" * 60)

    demo_bbox_adjustment()


if __name__ == "__main__":
    main()

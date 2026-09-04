"""
preprocessing/experiments/e4_contrast.py

Passo 7: Avaliação de técnicas de melhoria de contraste em baixa luminosidade.
- E4-A: RGB apenas (ilum. ruim, sem equalização)
- E4-B: equalizeHist (global no canal V do HSV)
- E4-C: CLAHE clip=2 tile=8 (LAB)
- E4-F: CLAHE clip=4 tile=8 (LAB)
"""
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.preprocessor import Preprocessor, PreprocessConfig
from preprocessing.utils.evaluate import evaluate_pipeline


def make_equalize_hist_v_fn(base_preprocessor):
    def _pipeline(frame_bgr: np.ndarray):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_eq = cv2.equalizeHist(v)
        bgr_eq = cv2.cvtColor(cv2.merge([h, s, v_eq]), cv2.COLOR_HSV2BGR)
        return base_preprocessor.process(bgr_eq)
    return _pipeline


def main():
    dark_yaml = "dataset/exports/epi-v1-dark/data.yaml"
    print("=" * 60)
    print("EXPERIMENTO E4: CONTRASTE E EQUALIZAÇÃO EM BAIXA LUZ")
    print("=" * 60)

    # E4-A: Sem equalização (RGB apenas em imagem escura)
    cfg_a = PreprocessConfig(infer_size=320, convert_rgb=True, use_letterbox=True, clahe=False)
    pp_a = Preprocessor(cfg_a)
    map_a = evaluate_pipeline(pp_a.process, label="e4_a_dark_none", data_yaml=dark_yaml)

    # E4-B: equalizeHist global (canal V-HSV)
    pp_base = Preprocessor(PreprocessConfig(infer_size=320, convert_rgb=True, use_letterbox=True, clahe=False))
    fn_b = make_equalize_hist_v_fn(pp_base)
    map_b = evaluate_pipeline(fn_b, label="e4_b_equalize_hist", data_yaml=dark_yaml)

    # E4-C: CLAHE clip=2, tile=8 (LAB)
    cfg_c = PreprocessConfig(
        infer_size=320, convert_rgb=True, use_letterbox=True,
        clahe=True, clahe_clip=2.0, clahe_tile=8, clahe_space="lab",
    )
    pp_c = Preprocessor(cfg_c)
    map_c = evaluate_pipeline(pp_c.process, label="e4_c_clahe_clip2", data_yaml=dark_yaml)

    # E4-F: CLAHE clip=4, tile=8 (LAB)
    cfg_f = PreprocessConfig(
        infer_size=320, convert_rgb=True, use_letterbox=True,
        clahe=True, clahe_clip=4.0, clahe_tile=8, clahe_space="lab",
    )
    pp_f = Preprocessor(cfg_f)
    map_f = evaluate_pipeline(pp_f.process, label="e4_f_clahe_clip4", data_yaml=dark_yaml)

    print("-" * 60)
    print(f"E4-A: RGB apenas (ilum. ruim)     : mAP@0.5 = {map_a:.4f}")
    print(f"E4-B: equalizeHist (V-HSV)         : mAP@0.5 = {map_b:.4f} (Delta vs E4-A: {map_b - map_a:+.4f})")
    print(f"E4-C: CLAHE clip=2 tile=8 (LAB)    : mAP@0.5 = {map_c:.4f} (Delta vs E4-A: {map_c - map_a:+.4f})")
    print(f"E4-F: CLAHE clip=4 tile=8 (LAB)    : mAP@0.5 = {map_f:.4f} (Delta vs E4-A: {map_f - map_a:+.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
preprocessing/experiments/e1_color_space.py

Passo 4: Avaliação do impacto do espaço de cor (BGR vs RGB).
- E1-A: BGR sem conversão
- E1-B: BGR->RGB (cvtColor)
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.preprocessor import Preprocessor, PreprocessConfig
from preprocessing.utils.evaluate import evaluate_pipeline


def main():
    print("=" * 60)
    print("EXPERIMENTO E1: ESPAÇO DE COR (BGR vs RGB)")
    print("=" * 60)

    # E1-A: BGR sem conversão
    cfg_a = PreprocessConfig(infer_size=320, convert_rgb=False, use_letterbox=True)
    pp_a = Preprocessor(cfg_a)
    map_a = evaluate_pipeline(pp_a.process, label="e1_a_bgr")

    # E1-B: BGR->RGB (cvtColor)
    cfg_b = PreprocessConfig(infer_size=320, convert_rgb=True, use_letterbox=True)
    pp_b = Preprocessor(cfg_b)
    map_b = evaluate_pipeline(pp_b.process, label="e1_b_rgb")

    print("-" * 60)
    print(f"E1-A: BGR sem conversão  : mAP@0.5 = {map_a:.4f}")
    print(f"E1-B: BGR->RGB (cvtColor): mAP@0.5 = {map_b:.4f}")
    print(f"Delta (E1-B - E1-A)      : {map_b - map_a:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

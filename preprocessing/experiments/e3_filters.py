"""
preprocessing/experiments/e3_filters.py

Passo 6: Avaliação de filtragem espacial e trade-off custo vs benefício:
- E3-A: RGB apenas (sem filtro)
- E3-B: GaussianBlur 3x3, sigma=0.8
- E3-C: GaussianBlur 5x5, sigma=1.5
- E3-D: medianBlur kernel=3
Benchmark de custo computacional com benchmark_filter_cost().
"""
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.preprocessor import Preprocessor, PreprocessConfig
from preprocessing.utils.evaluate import evaluate_pipeline


def benchmark_filter_cost(num_warmup: int = 20, num_iters: int = 100, frame_shape=(480, 640, 3)):
    print()
    print("-" * 60)
    print("BENCHMARK DE CUSTO COMPUTACIONAL DOS FILTROS (ms/frame)")
    print("-" * 60)

    frame = np.random.randint(0, 255, frame_shape, dtype=np.uint8)

    filters = {
        "GaussianBlur 3x3 (sigma=0.8)": lambda f: cv2.GaussianBlur(f, (3, 3), sigmaX=0.8),
        "GaussianBlur 5x5 (sigma=1.5)": lambda f: cv2.GaussianBlur(f, (5, 5), sigmaX=1.5),
        "medianBlur k=3              ": lambda f: cv2.medianBlur(f, 3),
    }

    results = {}
    for name, fn in filters.items():
        for _ in range(num_warmup):
            fn(frame)

        t0 = time.perf_counter()
        for _ in range(num_iters):
            fn(frame)
        total_time = (time.perf_counter() - t0) * 1000
        avg_ms = total_time / num_iters
        results[name.strip()] = avg_ms
        print(f"{name} : {avg_ms:6.2f} ms/frame ({1000.0 / avg_ms:5.1f} FPS equivalente)")

    print("-" * 60)
    print()
    return results


def main():
    print("=" * 60)
    print("EXPERIMENTO E3: FILTRAGEM ESPACIAL (Gaussiano vs Mediana)")
    print("=" * 60)

    # E3-A: Sem filtro (baseline)
    cfg_a = PreprocessConfig(infer_size=320, convert_rgb=True, use_letterbox=True, gaussian_blur=False, median_blur=False)
    pp_a = Preprocessor(cfg_a)
    map_a = evaluate_pipeline(pp_a.process, label="e3_a_nofilter")

    # E3-B: GaussianBlur 3x3, sigma=0.8
    cfg_b = PreprocessConfig(
        infer_size=320, convert_rgb=True, use_letterbox=True,
        gaussian_blur=True, gaussian_ksize=3, gaussian_sigma=0.8,
    )
    pp_b = Preprocessor(cfg_b)
    map_b = evaluate_pipeline(pp_b.process, label="e3_b_gauss3")

    # E3-C: GaussianBlur 5x5, sigma=1.5
    cfg_c = PreprocessConfig(
        infer_size=320, convert_rgb=True, use_letterbox=True,
        gaussian_blur=True, gaussian_ksize=5, gaussian_sigma=1.5,
    )
    pp_c = Preprocessor(cfg_c)
    map_c = evaluate_pipeline(pp_c.process, label="e3_c_gauss5")

    # E3-D: medianBlur k=3
    cfg_d = PreprocessConfig(
        infer_size=320, convert_rgb=True, use_letterbox=True,
        median_blur=True, median_ksize=3,
    )
    pp_d = Preprocessor(cfg_d)
    map_d = evaluate_pipeline(pp_d.process, label="e3_d_median3")

    print("-" * 60)
    print(f"E3-A: Sem filtro (baseline)  : mAP@0.5 = {map_a:.4f}")
    print(f"E3-B: GaussianBlur 3x3       : mAP@0.5 = {map_b:.4f} (Delta: {map_b - map_a:+.4f})")
    print(f"E3-C: GaussianBlur 5x5       : mAP@0.5 = {map_c:.4f} (Delta: {map_c - map_a:+.4f})")
    print(f"E3-D: medianBlur k=3         : mAP@0.5 = {map_d:.4f} (Delta: {map_d - map_a:+.4f})")
    print("=" * 60)

    benchmark_filter_cost()


if __name__ == "__main__":
    main()

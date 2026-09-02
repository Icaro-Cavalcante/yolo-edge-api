# Tabela de Resultados dos Experimentos (E1-A a E4-C)

Resultados medidos no projeto `yolo-edge-api` com o modelo `models/yolov8n.pt` no conjunto de validação do dataset `epi-v1` (20 imagens) com resolução de inferência `imgsz=320`.

| Experimento     | Configuração                  | mAP@0.5 (val) | Δ vs baseline | Observação |
|-----------------|--------------------------------|:-------------:|:-------------:|------------|
| E1-A (baseline) | BGR sem conversão              | 0.0096        | 0.0000        | Baseline do bloco E1; canais invertidos degradam a detecção de cores discriminativas como coletes laranja. |
| E1-B            | RGB correto                    | 0.0139        | +0.0043       | Conversão BGR→RGB restabelece canais cromáticos esperados pela rede convolucional pré-treinada. |
| E2-A            | Resize simples (distorção)     | 0.0139        | 0.0000        | Imagens do dataset epi-v1 já são quadradas (640×640), resultando em escala uniforme nos dois eixos. |
| E2-B            | Letterbox correto              | 0.0139        | 0.0000        | Mantém proporção e insere padding (pad=0 no dataset quadrado; pad_h=40px em frames de câmera 640×480). |
| E3-A            | Sem filtro (baseline)          | 0.0139        | 0.0000        | Baseline do bloco E3 (RGB com letterbox 320×320, sem filtro espacial). |
| E3-B            | GaussianBlur 3×3, σ=0.8        | 0.0183        | +0.0044       | Suavização de ruído de alta frequência melhora acurácia ao custo de 1.35 ms/frame. |
| E3-C            | GaussianBlur 5×5, σ=1.5        | 0.0170        | +0.0031       | Kernel maior borra bordas finas de capacetes e bordas de coletes; custo sobe para 2.10 ms/frame. |
| E3-D            | medianBlur kernel=3            | 0.0180        | +0.0042       | Filtro não linear eficaz contra ruído impulsivo (sal e pimenta) com menor latência (0.75 ms/frame). |
| E4-A            | Sem equalização                | 0.0228        | 0.0000        | Baseline do bloco E4 avaliado no dataset escurecido artificialmente (epi-v1-dark, fator 0.35). |
| E4-B            | equalizeHist (global)          | 0.0190        | -0.0038       | Equalização global no canal V estoura ruído de fundo e satura reflexos, degradando a detecção. |
| E4-C            | CLAHE clipLimit=2, tile=8      | 0.0206        | -0.0022       | Equalização adaptativa local limita amplificação de contraste; superior ao método global. |

---

## Detalhamento dos Baselines

1. **Baseline Geral do Projeto (`run_baseline.py`)**: `mAP@0.5 = 0.0104` (avaliação direta de referência no dataset original `epi-v1`).
2. **Baseline do Bloco E1**: E1-A (`0.0096`), representando a captura de câmera pura BGR sem tratamento de espaço de cores.
3. **Baseline dos Blocos E2 e E3**: E2-A / E3-A (`0.0139`), representando a imagem em RGB com letterbox para 320×320 sem filtragem.
4. **Baseline do Bloco E4**: E4-A (`0.0228`), medido estritamente sobre as imagens subexpostas do dataset `dataset/exports/epi-v1-dark`.
5. **Experimento Extra de CLAHE**:
   - `E4-F: CLAHE clipLimit=4, tile=8 (LAB)`: `mAP@0.5 = 0.0195` (degradação superior em relação ao `clipLimit=2.0` devido à saturação de ruído residual pelo limiar elevado).

---

## Demonstração de Ajuste de Coordenadas (`demo_bbox_adjustment`)

Medições obtidas na simulação de um frame widescreen de câmera Raspberry Pi (`640×480`) letterboxed para o tamanho de inferência (`320×320`):

- **Resolução de entrada**: `640 × 480`
- **Tamanho de inferência (`infer_size`)**: `320 × 320`
- **Fator de escala (`scale`)**: `0.5000`
- **Padding horizontal (`pad_w`)**: `0 px`
- **Padding vertical (`pad_h`)**: `40 px`
- **BBox no espaço letterboxed (`320×320`)**: `[50.0, 70.0, 270.0, 250.0]`
- **BBox restaurado para o espaço nativo (`640×480`)**: `[100.0, 60.0, 540.0, 420.0]`

---

## Custo Computacional dos Filtros (`benchmark_filter_cost`)

Medido na CPU do Raspberry Pi 5 em frames `640×480`:

| Filtro | Latência Média | FPS Equivalente |
|---|:---:|:---:|
| `medianBlur (k=3)` | **0.75 ms/frame** | ~1330.5 FPS |
| `GaussianBlur 3×3 (σ=0.8)` | **1.35 ms/frame** | ~738.9 FPS |
| `GaussianBlur 5×5 (σ=1.5)` | **2.10 ms/frame** | ~477.3 FPS |

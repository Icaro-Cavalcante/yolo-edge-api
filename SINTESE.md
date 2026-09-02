# Síntese Técnica de Pré-processamento de Imagens (`yolo-edge-api`)

Os experimentos práticos realizados no projeto `yolo-edge-api` permitiram quantificar o impacto de cada etapa de pré-processamento na acurácia (`mAP@0.5`) e na latência de inferência embarcada no Raspberry Pi 5.

### 1. Espaço de Cor: BGR vs RGB (E1)
A conversão explícita de `BGR` para `RGB` elevou o `mAP@0.5` de **0.0096** (E1-A) para **0.0139** (E1-B), um ganho relativo de **+44.8%** (+0.0043). Redes neurais convolucionais pré-treinadas dependem criticamente da ordem espectral dos canais de entrada. No dataset `epi-v1`, classes como `Colete` (com cores de segurança fluorescentes em laranja e amarelo) e `Capacete` perdem sua assinatura cromática distintiva quando os canais azul e vermelho são invertidos, tornando mandatória a ativação de `convert_rgb=True`.

### 2. Redimensionamento: Resize Ingênuo vs Letterbox (E2)
No conjunto de validação do `epi-v1`, as imagens já são quadradas (640×640), fazendo com que tanto o resize simples quanto o letterbox atinjam o mesmo `mAP@0.5` de **0.0139** (`scale=0.5`, `pad=0`). Entretanto, na câmera CSI do Raspberry Pi (`stream/v3_optimized.py`), a captura opera nativamente em resolução retangular `640×480`. O resize ingênuo para 320×320 impõe distorção anamórfica de 33% no eixo vertical, alterando a morfologia dos objetos. O letterbox preserva o aspecto original inserindo padding simétrico (`pad_h=40px`), exigindo a função `adjust_boxes` para subtrair esse offset e desescalonar as predições de volta ao espaço original da câmera.

### 3. Filtros Espaciais: Custo vs Benefício (E3)
A filtragem Gaussiana 3×3 obteve o maior `mAP@0.5` (**0.0183**, ganho de **+0.0044**), e o filtro mediano (k=3) alcançou **0.0180** com menor custo (**0.75 ms/frame** contra **1.35 ms** do Gaussiano 3×3 e **2.10 ms** do Gaussiano 5×5). Contudo, em taxas de captura de 30 FPS no Raspberry Pi, qualquer milissegundo adicional concorre diretamente com o ciclo de inferência. Como a câmera opera em ambiente industrial com iluminação controlada e ruído leve, os filtros espaciais permanecem desativados por padrão para maximizar o throughput.

### 4. Equalização de Contraste (E4)
Sob iluminação degradada (`epi-v1-dark`, fator 0.35), o CLAHE local no canal L do espaço LAB com `clipLimit=2.0, tile=8` obteve `mAP@0.5` de **0.0206**, superando a equalização global `equalizeHist` (**0.0190**). O método global satura ruídos de fundo e estoura regiões claras, enquanto o CLAHE restringe a amplificação de contraste a blocos locais. Em imagens com exposição normal, equalizações distorcem a distribuição de luminância; portanto, o CLAHE deve ser restrito ao preset de baixa luminosidade.

### 5. Configuração Final do `preprocessor.py`
- **`CONFIG_DEFAULT`**:
  - `infer_size=320`: equilíbrio ideal entre latência (~160 ms no Raspberry Pi) e resolução mínima detectável.
  - `convert_rgb=True`: justificado pelo salto de 0.0096 para 0.0139 em E1-B.
  - `use_letterbox=True`: garante integridade geométrica para câmeras 640×480 sem deformação de proporção.
  - `gaussian_blur=False` / `median_blur=False`: evita overhead de 0.75 ms a 1.35 ms por frame medido em E3.
  - `clahe=False`: evita artefatos em imagens com iluminação satisfatória.
  - `normalize=False`: o Ultralytics YOLOv8 já efetua a conversão `uint8` para `float32 [0, 1]` nativamente na inferência.
- **`CONFIG_LOW_LIGHT`**: mantém os mesmos parâmetros do default, porém com `clahe=True`, `clahe_clip=2.0`, `clahe_tile=8` e `clahe_space="lab"`, sustentado pelo melhor mAP medido em baixa luz em E4-C (0.0206).

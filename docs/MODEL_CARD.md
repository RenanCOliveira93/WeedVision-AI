# Model card · WeedVision `yolo11n_weed`

> Modelo de **pesquisa/demo**. **Não** usar para decidir aplicação de herbicida nem para controlar atuadores.

## Resumo
| Campo | Valor |
|---|---|
| Modelo | YOLO11n (Ultralytics), ajustado de pesos COCO; exportado para ONNX (`models/onnx/yolo11n_weed.onnx`, 10,6 MB) |
| Tarefa | **detecção** por caixas de **uma classe**: `weed` |
| Dados de treino | Sudars et al. 2020, split de treino (650 imagens, 4.462 caixas `weed`), CC BY 4.0 |
| Entrada / saída | RGB, letterbox 640×640 → caixas (xyxy em pixels da imagem original), confiança |
| Licença | **AGPL-3.0** (pesos e código Ultralytics) ou licença Enterprise; ver [10_licencas](pt-BR/10_licencas.md) |
| Runtime de inferência | `weedvision/onnx_infer.py` (onnxruntime; **não importa ultralytics**) |

## Desempenho medido (módulo de métricas próprio, `runs/eval`)
<!--T:EVAL-->
| Modelo | Conjunto | Domínio | Imgs | mAP50 | mAP50-95 | P@0,25 | R@0,25 | F1@0,25 |
|---|---|---|---|---|---|---|---|---|
| frcnn_mnv3 | test | mesmo | 134 | 0.612 | 0.326 | 54.3% | 68.4% | 60.5% |
| frcnn_mnv3 | test_origin | origem retida | 241 | 0.186 | 0.070 | 34.9% | 24.4% | 28.7% |
| frcnn_mnv3 | val | mesmo | 151 | 0.709 | 0.323 | 61.2% | 76.0% | 67.8% |
| frcnn_mnv3_on_maize | test | outro (zero-shot) | 500 | 0.000 | 0.000 | — | 0.0% | — |
| frcnn_mnv3_v2 | test | mesmo | 134 | 0.605 | 0.317 | 52.5% | 67.5% | 59.1% |
| frcnn_mnv3_v2 | test_origin | origem retida | 241 | 0.179 | 0.063 | 34.3% | 25.1% | 29.0% |
| frcnn_mnv3_v2 | val | mesmo | 151 | 0.689 | 0.312 | 58.1% | 74.4% | 65.3% |
| frcnn_mnv3_v2_on_maize | test | outro (zero-shot) | 500 | 0.000 | 0.000 | — | 0.0% | — |
| yolo11n | test | mesmo | 134 | 0.725 | 0.488 | 65.3% | 75.2% | 69.9% |
| yolo11n | test_origin | origem retida | 241 | 0.291 | 0.127 | 45.9% | 32.6% | 38.1% |
| yolo11n | val | mesmo | 151 | 0.821 | 0.535 | 72.9% | 81.5% | 76.9% |
| yolo11n_on_maize | test | outro (zero-shot) | 500 | 0.000 | 0.000 | — | 0.0% | — |
| yolo11n_onnx | test | mesmo | 134 | 0.723 | 0.487 | 64.1% | 75.9% | 69.5% |
| yolo11n_onnx | val | mesmo | 151 | 0.827 | 0.543 | 71.1% | 83.7% | 76.9% |
| yolo11n_onnx_coreml | test | mesmo | 134 | 0.723 | 0.485 | 63.9% | 76.8% | 69.7% |
| yolo11n_onnx_coreml | val | mesmo | 151 | 0.827 | 0.535 | 70.8% | 83.7% | 76.7% |
| yolo11s | test | mesmo | 134 | 0.727 | 0.512 | 69.2% | 71.7% | 70.4% |
| yolo11s | test_origin | origem retida | 241 | 0.217 | 0.092 | 38.3% | 22.2% | 28.1% |
| yolo11s | val | mesmo | 151 | 0.810 | 0.549 | 76.3% | 74.2% | 75.2% |
| yolo11s_on_maize | test | outro (zero-shot) | 500 | 0.000 | 0.000 | — | 0.0% | — |
| yolo11s_onnx | test | mesmo | 134 | 0.728 | 0.512 | 68.3% | 70.6% | 69.4% |
| yolo11s_onnx | val | mesmo | 151 | 0.813 | 0.554 | 75.7% | 74.9% | 75.3% |
<!--/T:EVAL-->
Incerteza (bootstrap por imagem, otimista):
<!--T:BOOT-->
| Conjunto | Modelo | mAP50 (IC 95%) | mAP50-95 (IC 95%) | Δ mAP50 vs yolo11n (IC 95%) |
|---|---|---|---|---|
| val (n=151) | yolo11n | 0.822 [0.796, 0.847] | [0.512, 0.560] | — |
| val (n=151) | yolo11s | 0.811 [0.783, 0.838] | [0.528, 0.575] | -0.011 [-0.027, +0.004] |
| val (n=151) | frcnn_mnv3_v2 | 0.689 [0.654, 0.727] | [0.292, 0.337] | -0.133 [-0.156, -0.112] |
| test (n=134) | yolo11n | 0.724 [0.668, 0.778] | [0.447, 0.532] | — |
| test (n=134) | yolo11s | 0.728 [0.665, 0.786] | [0.464, 0.555] | +0.003 [-0.027, +0.030] |
| test (n=134) | frcnn_mnv3_v2 | 0.604 [0.539, 0.667] | [0.277, 0.359] | -0.121 [-0.154, -0.088] |
| test_origin (n=241) | yolo11n | 0.298 [0.249, 0.343] | [0.107, 0.155] | — |
| test_origin (n=241) | yolo11s | 0.220 [0.173, 0.274] | [0.072, 0.119] | -0.078 [-0.105, -0.047] |
| test_origin (n=241) | frcnn_mnv3_v2 | 0.184 [0.148, 0.216] | [0.053, 0.079] | -0.114 [-0.154, -0.070] |
<!--/T:BOOT-->
Paridade PyTorch × ONNX medida no val e no teste (diferença de mAP50 ≤ 0,006; ver notebook 03).

## Uso pretendido
Pesquisa e demonstração do pipeline; apoio à **inspeção humana** em imagens próximas ao solo parecidas com o Sudars.

## Fora do escopo / não usar para
- **Qualquer decisão de pulverização** ou acionamento de atuador.
- **Café brasileiro** e qualquer cultura/câmera/altura diferente das de treino: **não validado**; o teste em outro dataset deu **mAP50 = 0**.
- Distinguir cultura de daninha: a classe `crop` **não foi treinada** (3 caixas) e o modelo **confunde parte da cultura com daninha** quando a origem muda (145 de 534 falsos positivos no grupo retido tinham o centro dentro de uma caixa `crop`).
- Segmentação, contagem exata ou densidade de infestação.

## Limitações conhecidas
- Treinado em ≈ 650 imagens, 3 sessões **inferidas**; mistura estufa e campo.
- Objetos minúsculos (mediana ≈ 0,1% da imagem; ≈ 15×15 px a 640).
- Sem imagens negativas no treino.
- Uma semente de treino; ICs otimistas.
- Benchmark só em Apple M4 Max.
- Coordenadas geográficas só com calibração e pose; validação apenas sintética.

## Riscos e mitigação
| Risco | Mitigação no projeto |
|---|---|
| Falso positivo sobre a cultura | medido; documentado; **sem integração com atuador**; revisão humana |
| Falso negativo (daninha não detectada) | análise de FN por tamanho/motivo; não usar a contagem como censo |
| Mudança de câmera/cultura/região | rótulo de domínio na API/UI; teste por origem; monitoramento de drift (doc 07) |
| Coordenada errada | API recusa sem calibração+pose; aviso de erro no GeoJSON |
| Licença | AGPL documentada; alternativa permissiva comparada (Faster R-CNN, pior neste conjunto) |

## Como foi avaliado
Split por **sessão inferida** + origem `IMG_*` retida; limiar de confiança 0,25 fixado antes da avaliação; todas as métricas do mesmo código, conferido contra `pycocotools`. Ver [06_treino_e_avaliacao](pt-BR/06_treino_e_avaliacao.md).

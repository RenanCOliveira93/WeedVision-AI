# 06 · Treino, avaliação e comparação de modelos

## O que foi treinado (tudo na CPU/GPU Apple M4 Max, MPS; uma semente por modelo)
| Modelo | Origem dos pesos iniciais | Receita | Tempo de treino |
|---|---|---|---|
| YOLO11n | COCO (Ultralytics) | 40 épocas, 640 px, lote 16, `patience=15`, augment padrão | 667 s |
| YOLO11s | COCO (Ultralytics) | idem | 953 s |
| Faster R-CNN MobileNetV3-Large FPN **v1** | COCO (torchvision) | 12 épocas, SGD lr 0,01, lote 4, flip | 446 s |
| Faster R-CNN MobileNetV3-Large FPN **v2** | COCO (torchvision) | idem com lr 0,005 + warmup de 100 iterações | ≈ 400 s |

Classe única `weed` (a classe `crop` não é aprendível neste split; ver doc 01). **Segmentação não foi treinada.**
*O modelo YOLO salvo é o `best.pt` selecionado pelo mAP50-95 do val do próprio Ultralytics; as métricas abaixo vêm do módulo próprio, não do relatório do Ultralytics.*

## Como medimos
`weedvision/metrics.py`: AP de 101 pontos (estilo COCO), casamento guloso por confiança, ≤ 100 detecções/imagem, mAP50-95 sobre IoU 0,50:0,05:0,95. **Conferido contra `pycocotools`** em 4 conjuntos aleatórios (tolerância 2·10⁻³). Precisão/revocação/F1 usam **confiança 0,25** e IoU 0,5 (limiar fixado de antemão, não ajustado no teste); a matriz de confusão também. O mesmo código avalia YOLO (PyTorch e ONNX) e Faster R-CNN.

## Resultados (gerados de `runs/eval/*.json`)
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

**Como ler:** "mesmo" = sessões do mesmo conjunto de origem, não vistas no treino; "origem retida" = grupo `IMG_*` nunca visto (câmera/origem distinta, **inclui 403 caixas `crop`**); "outro (zero-shot)" = milho da Nigéria, protocolo de anotação diferente.

## Incerteza (bootstrap por imagem, 200 reamostras, mesmos índices para todos os modelos)
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

> Os ICs **subestimam** a incerteza: as imagens de uma sessão são quase idênticas (correlacionadas) e o bootstrap as trata como independentes. Além disso o teste tem só 6 sessões pequenas e o val uma única sessão, e **uma só semente** por modelo.

### Leitura
- **YOLO11n ≈ YOLO11s** no val e no teste (diferença de mAP50 com IC que inclui 0). Não há base para dizer que o modelo maior é melhor aqui; no grupo de origem retida o **11s foi pior**.
- **Faster R-CNN MobileNetV3 ficou claramente abaixo** dos YOLO neste conjunto (ICs da diferença não incluem 0). Foi tentado **um** reajuste (v2), que **não melhorou**. A *loss* de treino somada **sobe** nas duas versões (0,58 → ~0,66–0,70), e **não diagnostiquei a causa**; pode ser instabilidade de otimização ou apenas o aumento de positivos amostrados — não verificado. Portanto "YOLO > Faster R-CNN" vale **para esta receita sem ajuste fino**, não como afirmação geral sobre as arquiteturas. A comparação também **não tem a mesma resolução de entrada** (Faster R-CNN vê 1138×640; YOLO 640 de lado maior), o que, se algo, favorece o Faster R-CNN.
- **Queda forte no grupo de origem retida** (mAP50 ≈ 0,2–0,3 contra ≈ 0,7 no teste): é a medida mais honesta de generalização para outra câmera/origem dentro do mesmo conjunto.
- **Zero-shot no milho: mAP50 = 0,000 para todos os modelos.**

## Falsos negativos e falsos positivos
Baldes de tamanho na escala dos dados (1280×720: 0,05% ≈ 21×21 px; 0,25% ≈ 48×48 px).
<!--T:FN-->
| Modelo / conjunto | FN / GT | muito pequeno | pequeno | médio | grande |
|---|---|---|---|---|---|
| yolo11n / test | 112/452 | 11/28 (39.3%) | 18/98 (18.4%) | 50/189 (26.5%) | 33/137 (24.1%) |
| yolo11n / test_origin | 944/1401 | 93/169 (55.0%) | 346/490 (70.6%) | 338/431 (78.4%) | 167/311 (53.7%) |
| yolo11s / test | 128/452 | 13/28 (46.4%) | 26/98 (26.5%) | 54/189 (28.6%) | 35/137 (25.5%) |
| yolo11s / test_origin | 1090/1401 | 121/169 (71.6%) | 387/490 (79.0%) | 362/431 (84.0%) | 220/311 (70.7%) |
| frcnn_mnv3_v2 / test | 147/452 | 12/28 (42.9%) | 31/98 (31.6%) | 63/189 (33.3%) | 41/137 (29.9%) |
| frcnn_mnv3_v2 / test_origin | 1049/1401 | 149/169 (88.2%) | 401/490 (81.8%) | 339/431 (78.7%) | 160/311 (51.4%) |
<!--/T:FN-->

- No **teste**, a taxa de FN **não cresce de forma regular com o tamanho** (objetos médios e grandes perdem tanto quanto os pequenos); só o balde "muito pequeno" é pior para o YOLO11n. Quase todos os FN são "**ausente**" (o modelo não propõe nada), não "localização ruim" nem "classe errada" (só há 1 classe).
- No **grupo de origem retida** a taxa de FN é alta em todos os tamanhos (YOLO11n: 54–78%; Faster R-CNN v2: 51–88%) e **não é monotônica** no YOLO11n: isso aponta para **mudança de domínio**, não para tamanho do objeto.
- A análise de **aglomeração** (vizinhos sobrepostos) é praticamente **vazia**: as anotações quase não se sobrepõem (4 e 42 objetos com vizinho sobreposto). Não dá para concluir nada sobre plantas agrupadas.
- **Falsos positivos sobre a cultura:** no grupo retido, dos 534 FP do YOLO11n (conf ≥ 0,25), **145 têm o centro dentro de uma caixa `crop` anotada** (≈ 27%); no teste de mesmo domínio, 1 de 181. O modelo, que nunca viu `crop`, **confunde parte da cultura com daninha** quando a origem muda. **Isso é o risco operacional central:** seria pulverizar a cultura.

## O que esta avaliação NÃO mostra
Desempenho em café ou em qualquer cultura brasileira; desempenho em outra câmera/altura; desempenho com negativos (sem mato); ganho do `augment.py` próprio; resultado de segmentação; estabilidade entre sementes de treino.

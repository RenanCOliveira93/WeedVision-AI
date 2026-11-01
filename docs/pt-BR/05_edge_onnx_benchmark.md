# 05 · Edge AI: exportação ONNX, runtime próprio e benchmark

> **Aviso central:** o benchmark foi feito num **Apple M4 Max**, que **não é hardware de borda**. Os números servem para comparar runtimes *entre si nesta máquina*; **não** estimam desempenho em Jetson, Raspberry Pi, celular ou câmera embarcada.

## Exportação e runtime
- `YOLO(best.pt).export(format="onnx", imgsz=640, opset=17, dynamic=False)` → `models/onnx/yolo11{n,s}_weed.onnx` (entrada fixa 1×3×640×640).
- `weedvision/onnx_infer.py` implementa **do zero** o letterbox (preenchimento 114, proporção preservada), o *decode* da saída `(1, 4+nc, 8400)` e a **NMS por classe** (IoU 0,6, ≤ 100 caixas). Depende só de `onnxruntime`, OpenCV e numpy: **não importa `ultralytics`**, mantendo a AGPL fora do código de inferência (os pesos continuam AGPL; doc 10).
- A imagem Docker da API (`infra/Dockerfile.backend`) instala só `requirements.txt` (sem torch): **escrita, não executada**.

## Paridade numérica (mesmo módulo de métricas, val e teste)
| Modelo | Conjunto | mAP50 PyTorch | mAP50 ONNX (CPU) | mAP50-95 PyTorch | mAP50-95 ONNX (CPU) |
|---|---|---|---|---|---|
| YOLO11n | val | 0,821 | 0,827 | 0,535 | 0,543 |
| YOLO11n | test | 0,725 | 0,723 | 0,488 | 0,487 |
| YOLO11s | val | 0,810 | 0,813 | 0,549 | 0,554 |
| YOLO11s | test | 0,727 | 0,728 | 0,512 | 0,512 |

A diferença é ≤ 0,006 de mAP50. Não é bit a bit: o `predict` do Ultralytics usa letterbox **retangular** (640×384 em imagens 16:9) e o ONNX fixo usa 640×640 quadrado; numa amostra de 28 imagens, 88 de 105 caixas pareiam com IoU > 0,95 e o |Δ confiança| máximo foi 0,10. Com o provedor **CoreML** (que pode usar FP16/ANE e rodar parte do grafo na CPU), o YOLO11n deu mAP50 0,827 (val) e 0,723 (teste), mAP50-95 0,535 e 0,485: igual no mAP50 e **−0,008 no mAP50-95 do val** (causa — p. ex. FP16 — **não verificada**).

## Benchmark (`scripts/benchmark.py`; `runs/benchmark.json`)
Batch 1, 100 imagens 1280×720 em memória, 10 de aquecimento; latência ponta a ponta (pré + inferência + pós). Cada configuração em **subprocesso próprio**, executado com a máquina sem outros processos pesados.
<!--T:BENCH-->
| Configuração | média (ms) | p50 | p95 | FPS | RSS pico (MB) |
|---|---|---|---|---|---|
| yolo11n · onnxruntime CPU | 13.1 | 13.0 | 14.1 | 76.5 | 654 |
| yolo11s · onnxruntime CPU | 25.7 | 23.9 | 34.1 | 38.9 | 837 |
| yolo11n · onnxruntime CoreML | 6.6 | 6.7 | 7.1 | 150.4 | 628 |
| yolo11s · onnxruntime CoreML | 10.6 | 10.6 | 11.2 | 94.5 | 812 |
| yolo11n · PyTorch CPU | 21.6 | 21.6 | 22.7 | 46.3 | 827 |
| yolo11s · PyTorch CPU | 35.0 | 36.0 | 37.1 | 28.5 | 1002 |
| yolo11n · PyTorch MPS | 8.0 | 7.5 | 10.6 | 125.0 | 886 |
| yolo11s · PyTorch MPS | 9.5 | 9.5 | 11.1 | 105.5 | 947 |
| frcnn_mnv3 · PyTorch CPU | 184.8 | 183.9 | 197.5 | 5.4 | 1257 |
| frcnn_mnv3 · PyTorch MPS | 16.7 | 16.4 | 21.1 | 59.8 | 1005 |

_Hardware: Apple M4 Max, 14 núcleos, macOS-15.8.2-arm64-arm-64bit._ Batch 1, 100 imagens 1280×720, 10 de aquecimento; latência ponta a ponta (pré+infer+pós) sem E/S de disco. RSS = pico do processo: inclui bibliotecas, o modelo E ≈ 300 MB de imagens de teste pré-carregadas (110 × 1280×720×3) — serve para comparar configurações entre si, NÃO é o consumo de borda. Uma execução por configuração; sem repetição entre dias/temperatura. O modelo 'frcnn_mnv3' usa os pesos v1; a latência independe dos pesos (mesma arquitetura).
<!--/T:BENCH-->

### Leitura (apenas nesta máquina)
- **ONNX Runtime CPU é ≈ 1,6× mais rápido que PyTorch CPU** para o mesmo modelo (13,1 vs 21,6 ms no 11n; 25,7 vs 35,0 ms no 11s), com mAP praticamente igual — mas os dois caminhos têm pré/pós-processamento **diferentes** (o do Ultralytics é mais completo), então a diferença **não é só do motor**.
- **CoreML (≈ 6,6 ms) e MPS (≈ 8,0 ms)** aceleram o 11n; no 11s os dois ficam próximos (≈ 10 ms). O log do ONNX Runtime informa que o CoreML **aceitou 314 de 320 nós** (5 partições; os 6 restantes ficam na CPU). "Aceitou" não prova em qual unidade (GPU/ANE) cada nó rodou — **isso não foi medido**.
- **Faster R-CNN MobileNetV3: 185 ms na CPU** (≈ 5 FPS) contra 16,7 ms na MPS: na CPU é ≈ 14× mais lento que o YOLO11n em ONNX **e** teve mAP menor. (A entrada do Faster R-CNN é maior — 1138×640 — o que também pesa.)
- O **RSS** inclui ≈ 300 MB de imagens de teste pré-carregadas pelo próprio benchmark: serve só para comparar configurações; o consumo real de borda é **menor e não foi medido**.
- Uma execução por configuração, sem repetição em outros momentos/temperaturas: **sem intervalo de confiança**. Energia, térmica e latência de captura da câmera **não foram medidas**.

## Para levar a hardware de borda real (não feito)
1. Medir **no alvo**; a latência do M4 Max não transfere.
2. Avaliar **FP16/INT8** (quantização) **com o mesmo teste de mAP**, porque a perda em objetos minúsculos (≈ 15 px) pode ser maior.
3. Considerar TensorRT (Jetson), NNAPI/CoreML (celular), OpenVINO (x86) — cada um muda operadores suportados.
4. Incluir captura, decodificação JPEG e I/O no orçamento de latência; a 640 px o 11n está a ≈ 15×15 px nos objetos medianos — resolução maior custa quadrático.
5. Se houver **qualquer** atuador: a detecção só pode **sugerir**; o acionamento exige regras agronômicas, intertravamento de segurança e validação de campo — **fora do escopo deste projeto**, que não controla atuadores.

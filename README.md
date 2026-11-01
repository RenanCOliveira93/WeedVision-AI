# Projeto 04 · WeedVision AI

🇧🇷 Português · [🇺🇸 English](README.en.md)

Plataforma de **Computer Vision e Edge AI** para **detecção de plantas daninhas** em imagens RGB de campo: ingestão e validação de anotações, treino/fine-tuning, avaliação com métricas próprias, exportação ONNX com runtime de borda, API FastAPI e painel React.

> ### ⚠️ O que este projeto NÃO é
> - **Detectar uma daninha não é recomendar pulverização.** Aplicação exige critério agronômico e validação em campo. **Não há controle de atuadores.**
> - **Não há dado de café.** Nenhum modelo foi validado para café brasileiro; o teste em outro dataset (milho, Nigéria) deu **mAP50 = 0**.
> - **Segmentação não foi treinada.** Classificação (DeepWeeds), detecção (Sudars) e segmentação são tarefas separadas; **nenhum rótulo de imagem virou caixa**.
> - **Georreferenciamento só com calibração + pose;** validado apenas com dados sintéticos.

## Resultados (medidos; ver `docs/pt-BR/06` e os notebooks)
Detecção de **uma classe (`weed`)**, treinada em Sudars et al. 2020 (CC BY 4.0). mAP medido pelo **módulo próprio**, conferido contra `pycocotools`; uma semente por modelo.

| Modelo | Teste (sessões não vistas) mAP50 / mAP50-95 | Origem retida `IMG_*` | Milho (Nigéria, zero-shot) |
|---|---|---|---|
| YOLO11n | 0,725 / 0,488 | 0,291 / 0,127 | **0,000** |
| YOLO11s | 0,727 / 0,512 | 0,217 / 0,092 | **0,000** |
| Faster R-CNN MobileNetV3 (v1) | 0,612 / 0,326 | 0,186 / 0,070 | **0,000** |
| Faster R-CNN MobileNetV3 (v2, 1 reajuste) | 0,605 / 0,317 | 0,179 / 0,063 | **0,000** |

- YOLO11n e 11s são **indistinguíveis** no mesmo domínio (bootstrap pareado); o Faster R-CNN ficou abaixo com a receita usada (loss crescente **não diagnosticada**).
- **A generalização é o problema:** de ≈ 0,7 (sessões não vistas) a 0,2–0,3 (outra origem) e 0 (outro dataset). No grupo `IMG_*`, **145 de 534** falsos positivos do YOLO11n caem sobre a **cultura**.
- **DeepWeeds (classificação, não detecção):** split aleatório 91,8–94,3% de acurácia vs **46,5–61,2%** com split por data (3 sorteios de dias) — o split aleatório **vaza** quadros do mesmo dia. Os testes não são idênticos; ver doc 02/06.
- **Borda (Apple M4 Max, não é hardware de borda):** YOLO11n em ONNX Runtime CPU 13,1 ms (≈ 77 FPS); CoreML 6,6 ms; PyTorch CPU 21,6 ms. ONNX a ≤ 0,006 de mAP50 do PyTorch.

## Dados (verificados na fonte)
| Dataset | Uso | Licença |
|---|---|---|
| Sudars 2020 (detecção, 2 classes) | treino/val/teste | CC BY 4.0 |
| Milho Nigéria (detecção) | só teste de domínio | CC BY 4.0 |
| DeepWeeds (classificação) | baseline de classificação | **inconsistente** (CC BY 4.0 vs Apache-2.0) |
| CropAndWeed / CWFID | não usados | não comercial / pouco clara |

Limites do Sudars: **sessões inferidas** de IDs de arquivo, `crop` quase só no grupo retido (modelo de 1 classe), objetos minúsculos (≈ 0,1% da imagem), sem imagens negativas, estufa + campo misturados.

## Arquitetura
`weedvision/` (dados, anotações, augment, métricas, ONNX, georef, calibração, drift) → `scripts/` (treino/avaliação) → `models/registry.json` + `runs/eval/*.json` → `backend/app` (FastAPI, só ONNX) → `frontend/` (React/TS). Diagrama em [docs/pt-BR/00_visao_geral.md](docs/pt-BR/00_visao_geral.md).

## Início rápido
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt && pip install -e .
python -m uvicorn backend.app.main:app --port 8000     # OpenAPI em /docs
cd frontend && npm install && npm run dev               # http://localhost:5173
pytest -q                                               # 40 testes (sem navegador)
```
Treino, dados e reprodução completa: [docs/pt-BR/11_reproducao.md](docs/pt-BR/11_reproducao.md). Os dados brutos **não** estão versionados.

## API (FastAPI, `/docs`)
`POST /infer/image`, `/infer/image/annotated`, `/infer/video` · `GET /models`, `/evaluations`, `/benchmark`, `/classification`, `/limits` · `POST /export/csv` (pixels) e `/export/geojson` (**422 sem calibração e pose**).

## Documentação
`docs/pt-BR/`: 00 visão geral · 01 pesquisa de datasets · 02 dados e split · 03 anotações e augmentation · 04 georreferenciamento e calibração · 05 edge/ONNX/benchmark · 06 treino e avaliação · 07 domain shift e drift · 08 coleta em campo · 09 limitações · 10 licenças · 11 reprodução. Além de [MODEL_CARD](docs/MODEL_CARD.md), [dataset cards](docs/dataset_cards/) e 3 notebooks executados em `notebooks/`.

## Status de verificação
| Item | Status |
|---|---|
| 40 testes pytest, 5 vitest, 5 E2E Playwright | ✅ executados |
| Notebooks (3) | ✅ executados com dados reais |
| Métricas × pycocotools | ✅ conferidas |
| **Docker / docker-compose** | ❌ **escritos, nunca executados** (sem Docker na máquina) |
| Webcam real | ❌ só dispositivo falso do Chromium |
| Hardware de borda real, quantização, energia | ❌ não medidos |
| Calibração/georreferenciamento em campo | ❌ só sintético |
| MLflow | ❌ não integrado |
| Licença do código | GPL-3.0 (`LICENSE`); pesos YOLO11 sob **AGPL-3.0** (doc 10): revise a compatibilidade antes de redistribuir |

## Repositório, versionamento e execução em outra máquina

Código publicado a partir da máquina de desenvolvimento original (macOS, Podman). Isto **não** foi executado em GitHub Codespaces nem em Docker Desktop; se rodar lá, trate como primeira verificação e compare com a seção de testes.

- **Versionado:** `weedvision/`, `backend/`, `frontend/`, `scripts/`, notebooks, testes (40), `docs/pt-BR/`, `models/onnx/` (pesos ONNX treinados, ≈ 48 MB, derivados do Ultralytics YOLO11, **AGPL-3.0**), `models/registry.json` e os JSONs de resultado em `runs/` (`runs/eval/`, `benchmark`, `drift`, `bootstrap`, `deepweeds/*.json`), de onde a API e a documentação leem os números.
- **Não versionado:** `data/` (datasets brutos; veja `docs/pt-BR/02_dados.md` para baixar e conferir licenças), pesos `.pt` e demais artefatos de treino em `runs/` (≈ 250 MB), `models/pretrained/` (o Ultralytics baixa de novo ao treinar).
- Treino completo (`make train`) exige os dados e é demorado; Docker/compose **nunca foram executados** (veja *Status de verificação*).
- **Licença do código:** GPL-3.0 (arquivo `LICENSE`). Licenças e atribuições de dados/modelos de terceiros: veja as seções e docs indicados acima.

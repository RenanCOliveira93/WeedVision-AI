# 10 · Licenças

> Esta é uma leitura técnica das licenças, **não é parecer jurídico**. Antes de uso comercial, valide com o jurídico da sua organização.

## Código e pesos
| Componente | Licença | Consequência prática |
|---|---|---|
| **Ultralytics (YOLO11)** — código e pesos | **AGPL-3.0** ou licença Enterprise paga | Treinar/servir com `ultralytics` num serviço acessível por rede aciona obrigações de código-fonte da AGPL para o serviço. Os pesos YOLO11 treinados aqui são derivados dos pesos Ultralytics: tratar como AGPL. **Uso comercial fechado → licença Enterprise ou trocar de modelo.** |
| Runtime de borda deste projeto (`weedvision/onnx_infer.py`) | código do projeto (licença a definir pela organização) | Usa só `onnxruntime` + OpenCV + numpy; **não importa `ultralytics`**. Isso evita acoplar o *código de inferência* à AGPL, mas **não muda a licença dos pesos**. |
| torchvision Faster R-CNN MobileNetV3 | BSD-3-Clause; pesos COCO iniciais | Alternativa permissiva comparada aqui (ver doc 06). |
| ONNX Runtime, OpenCV, FastAPI, Albumentations | MIT / Apache-2.0 / BSD | sem restrição relevante |

## Dados
| Dataset | Uso aqui | Licença verificada | Observação |
|---|---|---|---|
| **Sudars et al. 2020** (Mendeley, Ronin_OPEN_DB) | treino/val/teste de detecção | **CC BY 4.0** | exige atribuição; contagem de imagens difere (artigo: 1118; depósito: 1176) |
| **Milho/plantas daninhas (Nigéria)**, Olaniyi et al. 2023 | teste de generalização | **CC BY 4.0** (conferida na fonte Mendeley; a cópia HF é secundária) | protocolo de anotação diferente |
| **DeepWeeds** (Olsen et al. 2019) | baseline de **classificação** | **inconsistente**: README fala CC BY 4.0; Zenodo e GitHub marcam Apache-2.0 | documentar a discrepância; não redistribuímos as imagens |
| CropAndWeed | **não usado** | licença própria **não comercial** | descartado por restrição de uso |
| CWFID | **não usado** | licença própria / pouco clara | descartado |

As amostras em `sample_data/` são cópias CC BY 4.0 com atribuição em `sample_data/ATTRIBUTION.md`.

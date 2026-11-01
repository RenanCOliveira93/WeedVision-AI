# 03 · Validação de anotações e augmentation

## Validação (`weedvision/annotations.py`)
`validate_yolo` e `validate_coco` **relatam**, sem corrigir em silêncio: label órfão, imagem sem arquivo de label, linha malformada, classe fora do intervalo, coordenada fora da imagem, caixa minúscula, duplicata exata, `id` duplicado e referências inexistentes (COCO). Cada tipo de erro é coberto por teste. Rode a cada novo lote de anotação.

## Versionamento
O manifesto guarda sha256 por imagem; o hash do manifesto identifica a versão do dataset. Alterar qualquer imagem, caixa ou split muda o hash.

## Augmentation realista (`weedvision/augment.py`, Albumentations)
| Transformação | Fenômeno de campo que aproxima | Limite |
|---|---|---|
| brilho/contraste ±30%, gama 0,7–1,4 | sol forte, nuvem, exposição automática | não reproduz sombra projetada |
| desfoque de movimento / gaussiano | vibração, foco | |
| neblina leve | **poeira** em suspensão | aproximação grosseira, não é modelo de poeira real |
| matiz/saturação/valor | solo úmido vs seco, senescência | |
| `CoarseDropout` (1–6 retângulos) | oclusão por palha/folhas | retângulos pretos não parecem folhas |
| ruído gaussiano | sensor em pouca luz | |
| espelhamento horizontal | simetria da cena | |

Caixas são recortadas e **descartadas se restar < 20% da área** (`min_visibility`). O teste `test_augment_caixas_validas` roda 50 amostras e exige que toda caixa continue válida e dentro da imagem.

> O treino YOLO usou o pipeline de augmentation **padrão do Ultralytics** (mosaico, HSV, flip, escala etc.; e Albumentations se instalado). O módulo `augment.py` é o pipeline próprio, **disponível e testado**, mas **não foi o usado para produzir os pesos reportados**. Ele **não foi avaliado** quanto a ganho de desempenho.

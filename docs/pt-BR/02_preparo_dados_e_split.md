# 02 · Preparo de dados e split sem vazamento

`weedvision/data_prep.py` converte VOC → **YOLO** e **COCO**, recorta caixas ao limite da imagem, descarta caixas degeneradas (< 2 px) e grava `manifest.json` com **sha256 de cada imagem**, sessão e split (versão do dataset = hash do manifesto).

## Resultado
Descartes: 1 caixa degenerada; 0 imagens sem arquivo; 0 classes desconhecidas.

| Split | Imagens | Sessões* | `weed` | `crop` |
|---|---|---|---|---|
| train | 650 | 3 | 4.462 | 3 |
| val | 151 | 1 | 1.127 | 0 |
| test | 134 | 6 | 452 | 4 |
| test_origin (`IMG_*`) | 241 | 1 | 1.401 | 403 |

\* ver abaixo: **sessão é uma suposição**.

## Como "sessão" foi definida (e por que é só uma suposição)
O Sudars **não traz** fazenda, data ou sessão. Usamos:
1. **IDs numéricos contíguos:** um salto de ID > 100 abre nova sessão (`gap=100`). Capturas consecutivas tendem a ser quase idênticas; verificamos que vizinhos de ID são mais parecidos que imagens aleatórias.
2. **Grupo `IMG_####`** (nomes no padrão de câmera, tamanhos mistos 1000×750/750×1000) tratado como **origem distinta e retida** (`test_origin`). Que `IMG_` corresponda à Canon EOS 800D é **hipótese**, não verificada.

Sessões **inteiras** vão para um único split (atribuição gulosa por contagem de imagens). Isso **reduz, não elimina**, vazamento: sessões diferentes podem ser do mesmo local e dia. Consequência prática: as sessões são poucas e **desiguais** (a maior tem 616 imagens, a do val 151), então as métricas de val/teste dependem fortemente de qual sessão caiu onde.

## Características dos dados que afetam o treino
- Objetos **minúsculos**: área mediana ≈ 0,1% da imagem; 98% das caixas de treino < 0,5%. Em 1280×720 isso é ≈ 30×30 px, e **≈ 15×15 px depois de redimensionar para 640**.
- Mediana de 6 caixas `weed` por imagem no treino (máx. 23).
- 935 imagens 1280×720, 101 de 1000×750, 43 de 640×360, 34 de 750×1000 etc.

## Outros conjuntos
- **Milho (Nigéria):** `[x,y,w,h]` confirmado em 5.764 caixas (0 inválidas); `maize`→(guardado para análise), `weed`→classe 0; 500 imagens só como teste.
- **DeepWeeds:** o split **aleatório** vaza quadros do mesmo dia; o split **por data** separa dias inteiros. Como o *Parkinsonia* foi fotografado em **apenas 2 dias** (Parthenium e Siam weed em 3), é impossível ter as 9 classes em treino, validação e teste ao mesmo tempo com separação por dia → o split por data usa só **treino/teste** (70/30 de imagens aproximadas), exigindo cada classe nos dois.

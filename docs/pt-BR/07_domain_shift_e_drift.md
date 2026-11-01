# 07 · Domain shift e data drift

## Domain shift medido
| Teste | Modelo treinado em | Testado em | mAP50 |
|---|---|---|---|
| mesmo domínio, sessões não vistas | Sudars/train | Sudars/test | ≈ 0,72 (YOLO) |
| outra origem **do mesmo dataset** | Sudars/train | `IMG_*` (retido) | 0,18–0,30 |
| **outro dataset, país e cultura** | Sudars/train | Milho (Nigéria) | **0,000** |

Para o milho, **nenhum** modelo propõe caixa com confiança útil: o YOLO11n gera ≈ 15 caixas/imagem (10 imagens amostradas) a conf ≥ 0,001, mas com confiança **máxima ≈ 0,03** (0,028 na amostra) (a 0,05 não sobra nenhuma). O pipeline foi verificado (no val do Sudars o mesmo código dá ≈ 3,4 detecções/imagem a conf ≥ 0,25), então **não é bug de inferência**.

**Confundidores honestos** (não separamos): (1) **domínio visual** — cultura, país, câmera, solo, luz; (2) **protocolo de anotação** — o Sudars anota **planta individual** (≈ 30×30 px), o milho anota **região de mato** (caixas enormes). Um modelo que acerta plantas pequenas não deveria acertar regiões grandes de qualquer jeito. O zero é, portanto, **falha de generalização + incompatibilidade de protocolo**, e **não** permite concluir "o modelo não serve para milho". Permite concluir que **não se pode assumir transferência entre datasets** — nem para café.

## Data drift de aparência (`weedvision/drift.py`)
Estatísticas simples por imagem (brilho, contraste, nitidez por variância do Laplaciano, saturação, fração de verde, largura) comparadas ao **treino** com **PSI** (decis do treino) e **KS**.
<!--T:DRIFT-->
| Comparação (ref. = treino) | brilho | contraste | nitidez_laplaciano | saturacao | verde_relativo | largura_px |
|---|---|---|---|---|---|---|
| piso de ruído (metades do treino) | 0.20 | 0.48 | 0.43 | 0.13 | 0.37 | 0.00 |
| val (n=150) | 0.20 | 0.42 | 0.94 | 1.22 | 0.38 | 0.00 |
| test (n=134) | 0.64 | 0.52 | 0.87 | 0.65 | 0.21 | 0.00 |
| test_origin (n=150) | 2.72 | 7.37 | 4.80 | 8.29 | 3.60 | ∞ |
| maize_nigeria (n=150) | 8.29 | 5.06 | 8.29 | 8.29 | 8.29 | ∞ |
<!--/T:DRIFT-->

- **O piso de ruído importa.** Com ~75 imagens por grupo, o PSI entre duas *metades aleatórias do próprio treino* já chega a 0,1–0,5. A regra "PSI > 0,25 é grande" **não vale** com amostras tão pequenas.
- **val e test** (outras sessões, mesma câmera) já **passam do piso em algumas variáveis**: PSI de saturação 1,22 no val (≈ 10× o piso de 0,13) e nitidez 0,94/0,87 (piso 0,43). **Sessões do mesmo dataset já diferem de forma perceptível** (saturação, foco) — mais um motivo para o resultado depender de qual sessão cai no val/teste.
- **`IMG_*` e milho** ficam **muito acima do piso** (o brilho médio quase não muda — 118 no treino, 105 em `IMG_*`, 135 no milho —, então o deslocamento está em contraste, nitidez, saturação e cor, não em exposição) (PSI de ≈ 3 a 8,3; 8,29 é o teto numérico quando as distribuições **não se sobrepõem**). `largura_px = ∞` significa que a resolução de referência é constante (1280) e a do conjunto comparado é outra.
- Isto mede **aparência**, não *rótulos* nem *desempenho*: drift grande **prediz risco**, mas só a avaliação com rótulos confirma a queda.

## Monitoramento em produção (proposta)
Registrar estatísticas por lote de imagens e comparar ao treino com o mesmo cálculo; alertar quando ultrapassar o piso de ruído medido **para o tamanho do lote**; auditar amostras com rótulo periodicamente. Distribuição de confiança das detecções também é sinal: no milho ela colapsou para ≤ 0,03.

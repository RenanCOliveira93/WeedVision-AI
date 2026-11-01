# 09 · Limitações (o que não foi feito ou não foi validado)

## Dados e domínio
- **Nenhum dado de café ou de qualquer cultura brasileira.** Não se afirma generalização para café brasileiro a partir de datasets estrangeiros; o teste em outro dataset (milho, Nigéria) deu **mAP50 = 0**.
- Sudars mistura **vasos em estufa e campo aberto** e 3 câmeras; **não há metadado por imagem** para separar. Local de coleta **não confirmado**.
- **Sessão/fazenda é inferida** de IDs de arquivo (doc 02); o split reduz mas não elimina vazamento. Sessões poucas e desiguais (val = 1 sessão).
- Sem imagens negativas no Sudars (nenhuma imagem sem caixa): o comportamento em campo limpo **não foi medido**.
- Classe `crop`: 3 caixas no treino → **sem modelo e sem métrica**. O risco de pulverizar a cultura só foi sondado via falsos positivos sobre caixas `crop` do grupo retido.

## Modelos e avaliação
- **Uma semente** por modelo; sem variação entre execuções de treino. ICs de bootstrap por imagem são **otimistas** (correlação entre quadros de uma sessão).
- Faster R-CNN: **um** reajuste, sem melhora; **loss crescente não diagnosticada**; entrada em resolução diferente da do YOLO.
- `weedvision/augment.py` (pipeline próprio) **não foi usado nos pesos reportados** e não teve ganho medido.
- **Segmentação não foi treinada** (e nenhuma métrica de máscara existe).
- Limiar de confiança 0,25 para P/R/F1 é **convenção**, não calibrado para custo operacional (falso positivo = possível dano à cultura).

## Georreferenciamento
- Validado **só com dados sintéticos** (ida e volta, nadir, sensibilidade). **Sem calibração real e sem pontos de controle GNSS.** Solo plano assumido.

## Borda
- Benchmark em **Apple M4 Max** (CPU/CoreML/MPS), **não** em Jetson, Raspberry Pi, celular ou outro hardware de borda. Não extrapolar. Consumo de energia **não medido**. CoreML pode executar parte do grafo na CPU (conferir `providers` no JSON).
- Sem quantização (INT8/FP16) avaliada.

## Plataforma
- **Docker e docker-compose escritos mas NUNCA executados** (sem Docker nesta máquina).
- Webcam testada só com **dispositivo falso do Chromium** (E2E); nenhuma câmera real.
- Vídeo: contagem por quadro **sem rastreamento** (objetos repetidos entre quadros não são deduplicados). O vídeo de exemplo é **movimento sintético** sobre uma foto, não filmagem.
- MLflow não foi integrado (opcional na especificação).
- Licença do **código** do projeto indefinida; pesos YOLO11 sob **AGPL-3.0** (doc 10).

## O que o sistema NÃO faz
Não recomenda dose, produto nem momento de aplicação. Não controla atuadores. Identificar não é decidir.

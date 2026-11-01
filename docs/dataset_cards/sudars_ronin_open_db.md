# Dataset card · Sudars et al. 2020 (Ronin_OPEN_DB)

| Campo | Valor |
|---|---|
| Referência | Sudars, K., Jasko, J., Namatevs, I., Ozola, L., Badaukis, N. (2020). *Dataset of annotated food crops and weed images for robotic computer vision control.* Data in Brief 31:105833 · Mendeley Data DOI 10.17632/nj4vtk4tt6.1 |
| Licença | **CC BY 4.0** (exige atribuição) |
| Tarefa | detecção por caixas |
| Anotação | Pascal VOC XML; **2 classes** (`weed`, `crop`) — o artigo descreve 8 espécies daninhas e 6 culturas, mas o XML **não** as distingue |
| Imagens | **1176** no depósito (o artigo cita 1118 imagens e 7853 anotações: divergência não resolvida); 935 numéricas (1280×720) + 241 `IMG_####` de tamanhos mistos |
| Câmeras | RealSense D435, Canon EOS 800D, Sony W800 (conforme artigo); **não há metadado por imagem** |
| Condições | vasos em estufa **e** campo aberto (conforme artigo); **sem indicação por imagem**; local **não confirmado** |
| Uso aqui | treino/val/teste por sessão inferida + grupo `IMG_*` retido |
| Estatística | 7.442 `weed` e 411 `crop` (403 só em `IMG_*`); área mediana da caixa ≈ 0,1% da imagem; **0 imagens sem caixa** |
| Integridade | ZIP com SHA-256 verificado; sha256 por imagem no `manifest.json` |
| Riscos | sessões inferidas; objetos minúsculos; sem negativos; desequilíbrio extremo de classes; mistura estufa/campo |

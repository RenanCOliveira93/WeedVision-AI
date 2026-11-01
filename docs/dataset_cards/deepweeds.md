# Dataset card · DeepWeeds

| Campo | Valor |
|---|---|
| Referência | Olsen, A. et al. (2019). *DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning.* Scientific Reports · Zenodo 7939060 |
| Licença | **inconsistente entre fontes**: README do dataset diz CC BY 4.0; metadado do Zenodo e repositório GitHub dizem Apache-2.0. Documentado; imagens **não redistribuídas** |
| Tarefa | **classificação de imagem inteira** (9 classes: 8 espécies + `Negative`). **Não há caixas nem máscaras.** |
| Tamanho | 17.509 imagens 256×256; 9.106 negativas (52%) |
| Origem | Austrália; 60 dias de coleta (2016–2018) |
| Integridade | `images.zip` MD5 `b7b30f96d466fba86016aa5a26606e0f` (confere com o Zenodo) |
| Uso aqui | baseline de classificação, comparando split aleatório e split por data |
| **Restrição** | rótulos de imagem **nunca** convertidos em caixas |
| Achado | *Parkinsonia* em só 2 dias; *Parthenium* e *Siam weed* em 3 → split por dia impossível em 3 vias |

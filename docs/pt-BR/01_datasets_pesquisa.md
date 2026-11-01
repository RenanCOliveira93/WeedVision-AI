# 01 · Pesquisa de datasets públicos

Critérios: anotação de **detecção ou segmentação** (não só rótulo de imagem), licença verificável, origem geográfica, cultura, protocolo de uso. A verificação foi feita **na fonte** (Mendeley/Zenodo/GitHub); etiquetas de licença em espelhos (ex.: Hugging Face) foram tratadas como secundárias.

| Dataset | Tarefa | Anotação | Origem / cultura | Licença | Decisão |
|---|---|---|---|---|---|
| **Sudars et al. 2020** (Data in Brief 31:105833; Mendeley `nj4vtk4tt6`) | detecção | VOC XML, **2 classes** (`weed`, `crop`) | 3 câmeras RGB (RealSense D435, Canon EOS 800D, Sony W800); **vasos em estufa e campo aberto**; 6 culturas e 8 espécies daninhas no artigo, mas só `weed/crop` no XML. Local de coleta **não confirmado** | CC BY 4.0 | ✅ **base de treino/teste** |
| **Milho/daninhas** (Olaniyi et al. 2023) | detecção | caixas, `maize`/`weed` | milho, Nigéria; 500 imagens 1920×1080 na cópia HF; **`weed` = região de mato** | CC BY 4.0 (fonte) | ✅ **só teste de domínio** (zero-shot) |
| **DeepWeeds** (Olsen et al. 2019) | **classificação** | rótulo por imagem, 8 espécies + negativo | Austrália; 17.509 imagens 256×256; 60 dias | README: CC BY 4.0; Zenodo/GitHub: Apache-2.0 → **inconsistente** | ✅ baseline de classificação; **nunca** vira caixa |
| CropAndWeed | detecção/segmentação | rico (espécies, estádios) | Europa | **não comercial** (licença própria) | ❌ restrição de uso |
| CWFID | segmentação | máscaras | cenoura, Alemanha | custom / pouco clara | ❌ licença pouco clara; escala pequena |

## Achados que moldaram o projeto
- **Sudars tem 411 caixas `crop`, e 403 estão só no grupo `IMG_*`** (o treino tem 3). O detector principal é por isso de **1 classe (`weed`)**; a classe `crop` não é aprendível neste split e **não tem métrica reportada**. As caixas `crop` do grupo retido servem para medir **falsos positivos sobre a cultura**.
- **Sudars não tem imagens negativas** (0 imagens sem caixa). Um detector treinado só nisso não aprende "campo limpo".
- **Contagem diverge:** o artigo cita 1118 imagens e 7853 anotações; o depósito do Mendeley tem **1176** imagens. Usamos o depósito.
- **Protocolos de anotação divergem:** Sudars anota **planta individual**; o milho anota **região de mato**. Não são o mesmo alvo.
- **Nenhum conjunto é café.** Nada aqui sustenta afirmar desempenho em café brasileiro.
- **Segmentação:** nenhum dataset obtido com máscara utilizável e licença compatível; por isso não foi treinada.

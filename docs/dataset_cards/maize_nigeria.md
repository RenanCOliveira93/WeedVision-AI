# Dataset card · Milho e plantas daninhas (Nigéria)

| Campo | Valor |
|---|---|
| Referência | Olaniyi et al. (2023), conjunto de milho e daninhas; usada a cópia em Parquet no Hugging Face (2 arquivos) |
| Licença | **CC BY 4.0**, conferida na fonte original (Mendeley Data); a etiqueta do espelho HF é só secundária |
| Tarefa | detecção por caixas |
| Anotação | `bbox = [x, y, w, h]` em pixels (**verificado**: 0 inválidas em 5.764 caixas; seria inválido como xyxy em 4.352); `0 = maize`, `1 = weed` (mapeamento conferido visualmente) |
| Tamanho | 500 imagens 1920×1080; 2.887 `weed` e 2.877 `maize` |
| **Protocolo** | `weed` anota **regiões de mato** (caixas grandes), **não plantas individuais** como no Sudars |
| Uso aqui | **somente teste zero-shot** (nunca treino) |
| Riscos | imagens aparentemente de sequência (quase duplicadas); sem metadado de sessão; protocolo incompatível com o do treino |

# 08 · Estratégia de coleta de dados em campo

Nenhum dos conjuntos públicos usados representa café brasileiro. Para chegar a um modelo útil no seu sistema, os dados precisam vir do seu campo. Proposta (não executada nesta entrega):

## 1. Antes de fotografar
- **Defina a decisão** que a detecção apoia (mapa de infestação? contagem para amostragem? alerta ao agrônomo?). Isso define classes e métricas.
- **Defina o protocolo de anotação por escrito** e *não o mude no meio*: planta individual **ou** região de mato (os dois conjuntos usados aqui divergem exatamente nisso, e o modelo treinado num protocolo falhou no outro). Defina o que é "oclusão aceitável", tamanho mínimo e o que fazer com mato sob a copa.
- **Taxonomia agronômica** validada por um agrônomo: espécie-alvo vs. "mato tolerável" vs. cultura. Detectar ≠ decidir aplicar.

## 2. Cobertura (o que variar)
| Eixo | Mínimo recomendado |
|---|---|
| Talhões / fazendas | vários; **o teste deve ser uma fazenda inteira não vista** |
| Datas / estádios da cultura e do mato | ao longo do ciclo (chuva/seca, pós-capina) |
| Hora e luz | manhã, meio-dia, nublado, contraluz |
| Altura/ângulo da câmera | os que serão usados em operação (e só esses) |
| Câmera | a do sistema final; trocar de câmera = novo domínio |
| Negativos | imagens **sem** mato (o DeepWeeds tem 52% de negativos; o Sudars quase não) |

## 3. Metadados obrigatórios por imagem
fazenda/talhão, **sessão** (voo/passada), data/hora, câmera e lente, resolução, altura e ângulo, GNSS (se houver) e responsável. Sem `sessão` não há split sem vazamento — foi exatamente o que faltou no Sudars e precisou ser *inferido* aqui.

## 4. Anotação com controle de qualidade
- Dupla anotação em **10–20%** das imagens e medição de concordância (IoU entre anotadores, κ de classe).
- Ferramentas que exportam YOLO/COCO; rodar `weedvision.annotations.validate_*` a cada lote.
- Versionar o dataset (manifesto com sha256 por imagem, como `data_prep.build`).

## 5. Split
Por **fazenda/talhão/sessão**, nunca por imagem. Reserve uma fazenda inteira como teste final, tocada uma única vez.

## 6. Calibração e pose (se quiser mapas)
Calibre a câmera (doc 04), registre altura/ângulo fixos e GNSS sincronizado com o disparo. Valide a exatidão com **pontos de controle** medidos no solo antes de publicar qualquer mapa.

## 7. Ciclo de melhoria
Treine → meça no teste por fazenda → analise falsos negativos/positivos (mesmas tabelas do doc 06) → colete **onde falha** → monitore drift (doc 07) → retreine. Registre cada modelo no `models/registry.json`.

## 8. Quantidade
Não há número mágico. Como referência *de ordem de grandeza* do que foi usado aqui: ~650 imagens / ~4,5 mil caixas de treino deram mAP50 ≈ 0,7 em sessões não vistas **do mesmo domínio** — e ≈ 0 em outro domínio. A cobertura (fazendas, luz, estádios) pesa mais que o volume.

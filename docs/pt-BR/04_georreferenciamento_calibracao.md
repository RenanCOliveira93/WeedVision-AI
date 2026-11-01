# 04 · Georreferenciamento e calibração de câmera

**Regra do projeto:** uma coordenada geográfica só é produzida quando existem *calibração da câmera* **e** *pose* válidas. Sem isso a API responde `422` e o usuário recebe CSV em **pixels**. Pixel não é coordenada.

## O que o código faz (`weedvision/georef.py`)
1. Modelo **pinhole** com intrínsecos `fx, fy, cx, cy` (e distorção radial/tangencial opcional via OpenCV).
2. Pose da câmera: `lat, lon`, **altura sobre o solo**, `yaw` (rumo a partir do Norte, horário), `pitch` (inclinação para baixo; 90° = nadir) e `roll`.
3. Para cada pixel, traça-se o raio e intersecta-se com o **plano do solo** (z = 0 no referencial ENU local).
4. O deslocamento ENU (m) vira latitude/longitude por problema geodésico direto no elipsoide WGS84 (`pyproj.Geod`).
5. O ponto exportado de cada caixa é o **centro da base** (contato presumido com o solo), não o centro da caixa.

## Quando o código recusa (`GeorefError`)
- intrínsecos inválidos (fx ≤ 0, ponto principal fora da imagem) ou pose inválida (altura ≤ 0, pitch fora de [0°, 90°]);
- raio **acima do horizonte** (não intercepta o solo);
- ponto a mais de `max_range_m` (padrão 100 m) da câmera — perto do horizonte o erro explode;
- na exportação, **uma** falha aborta o arquivo inteiro: nunca saem coordenadas parciais.

## O que está validado e o que NÃO está
| Verificação | Status |
|---|---|
| projeção direta → inversa recupera o ponto (erro < 1e-6 m) | ✅ teste sintético |
| nadir: centro da imagem cai embaixo da câmera; deslocamento lateral = `altura·Δpx/f` | ✅ teste analítico |
| calibração por tabuleiro recupera `fx, fy` sintéticos (±2%, RMS < 0,5 px) | ✅ teste sintético (pontos projetados, não fotos reais) |
| exatidão em campo contra pontos de controle GNSS | ❌ **não medida** |
| calibração real de uma câmera real | ❌ **não feita** (nenhuma câmera/tabuleiro disponível nesta execução) |

## Fontes de erro que o modelo **não** corrige
- **Solo não plano.** O modelo assume plano; em declive ou micro-relevo o erro cresce.
- **Erro de pose.** `ground_error_sensitivity` mostra numericamente: o mesmo erro angular gera erro muito maior no solo perto do horizonte (testado: `far > near`).
- **Altura da câmera** medida errada desloca todos os pontos proporcionalmente.
- **Distorção residual**, **rolling shutter** e **movimento** durante a captura.
- **Plantas altas:** a base da caixa só toca o solo se a planta for baixa e a caixa a delimitar bem.

## Como calibrar (procedimento)
1. Imprima um tabuleiro 9×6 (cantos internos) e meça o lado do quadrado com paquímetro.
2. Fotografe ≥ 15 poses variadas (inclinado, perto/longe, cantos do quadro) **com a mesma resolução e foco** usados em campo.
3. `weedvision.calibration.calibrate_from_images(paths, pattern=(9,6), square_m=0.025)` devolve `Intrinsics` e o **RMS de reprojeção** (px). RMS alto ou poucas imagens detectadas → recalibre.
4. Registre a altura e o ângulo de montagem com a câmera **fixa**; qualquer mudança invalida a pose.

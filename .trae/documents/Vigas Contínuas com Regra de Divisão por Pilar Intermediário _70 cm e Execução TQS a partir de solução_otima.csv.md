## Objetivo

Implementar a criação de vigas ao longo das paredes como elementos contínuos entre faces de pilares alinhados, com divisão automática quando houver pilar intermediário com dimensão > 70 cm. Adicionar uma forma direta de executar o TQS a partir do CSV gerado pela otimização.

## Comportamento de Vigas

1. Alinhamento horizontal:

* Detectar pilares ao longo da parede (eixo X constante na parede; Y variável)

* Criar uma viga contínua da face direita do pilar mais à esquerda (min\_face do primeiro) até a face esquerda do pilar mais à direita (max\_face do último)

* Se existir qualquer pilar intermediário com largura (em X) > 70 cm, dividir em vigas entre pilares adjacentes: cada viga liga da face direita do pilar atual à face esquerda do próximo

1. Alinhamento vertical:

* Detectar pilares ao longo da parede (eixo Y constante na parede; X variável)

* Criar uma viga contínua da face superior do pilar mais abaixo (min\_face do primeiro) até a face inferior do pilar mais acima (max\_face do último)

* Se existir qualquer pilar intermediário com altura (em Y) > 70 cm, dividir em vigas entre pilares adjacentes: cada viga liga da face superior do pilar atual à face inferior do próximo

1. Detalhes de precisão:

* Faces dos pilares obtidas via bounds dos polígonos: para paredes horizontais usar `minx/maxx`, para paredes verticais usar `miny/maxy`

* Threshold de divisão configurável: `70.0 cm` (novo parâmetro em configuração)

* Apenas criar vigas quando houver ao menos dois pilares alinhados numa parede

## Implementação

### A. Geração de vigas por paredes

* Arquivo: `geometry/length_input_processor.py`

* Função: `_find_beam_locations`

* Alterações:

  * Ordenar interseções `column_intersections` pela coordenada ao longo da parede (horizontal: X; vertical: Y)

  * Calcular dimensão do pilar alinhada à parede: horizontal → `width_x = maxx - minx`; vertical → `height_y = maxy - miny`

  * Regra:

    * Se não houver intermediários com dimensão > threshold, criar uma única viga contínua entre o primeiro e o último

    * Caso contrário, criar vigas entre pares consecutivos de pilares (adjacentes), conectando faces exatas

* Referência atual da função: `geometry/length_input_processor.py:147–196`

### B. Parametrização do threshold

* Arquivo: `config/constants.py`

* Adição: `SPLIT_BEAM_COLUMN_THRESHOLD_CM = 70.0`

* Uso pela lógica de `_find_beam_locations`

### C. Cálculo geométrico do volume de vigas

* Arquivo: `utils/geometric_calculator.py`

* Continuar usando cálculo de comprimento efetivo subtraindo trechos dentro dos pilares: `calculate_beams_geometric_volume_with_subtractions`

* Referências: `utils/geometric_calculator.py:101–144` e uso em `utils/geometric_calculator.py:165–169`

## Execução TQS da solução ótima

### D. Método de conveniência na inferência

* Arquivo: `inference.py`

* Adição: `run_tqs_on_csv(csv_path: str) -> tuple[float, float]`

* Fluxo: ler CSV → processar segmentos → criar modelo no TQS → executar processamento global → ler `RESDES.HTM` → retornar `(aco_real, concreto_real)`

* Referência de execução TQS existente: `inference.py:377–423`

### E. Como usar

* Exemplo:

```
from inference import BuildingInference
inf = BuildingInference()  # opcionalmente com EXPERIMENT_ID
aco_pred, conc_geom, prob = inf.predict_from_csv("outputs/results/solucao_otima.csv")
aco_real, conc_real = inf.run_tqs_on_csv("outputs/results/solucao_otima.csv")
print(aco_pred, conc_geom, prob)
print(aco_real, conc_real)
```

## Validação

* Rodar uma otimização curta e verificar nos logs de geometria que `Vol. Vigas` varia entre gerações quando há mudanças nas posições/dimensões dos pilares

* Inspecionar endpoints das vigas criadas visualmente (utilitário de plot se existente) para confirmar conexões face a face

* Confirmar que o classificador de validade é treinado (presença de `validity_classifier.pkl`) e que a penalização é aplicada na função objetivo quando `prob_invalid` excede o threshold

## Impacto e Segurança

* Sem alterações de desempenho significativas: lógica é linear no número de pilares por parede

* Mantém parâmetros estruturais (`DEFAULT_BEAM_WIDTH_CM`, `DEFAULT_BEAM_HEIGHT_CM` e cargas) sem mudanças

* Toda a criação de vigas é determinística e respeita faces dos pilares

## Confirmação

* Ao aprovar, aplico as mudanças nos arquivos indicados, adiciono o parâmetro de threshold à configuração.


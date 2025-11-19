## Visão Geral

Vou analisar os artefatos do experimento `20251117-180524_Treino_com_600_amostras` e validar o cálculo geométrico das vigas face a face. Em seguida proponho melhorias objetivas para o modelo e para a verificação/trace do volume.

## Avaliação de Treinamento

* Métricas finais (regressão de aço):

  * Fonte: `metrics/summary.json`

  * Resultado atual: `r2_score≈0.214`, `MAE_steel≈54.9 kgf` ⇒ qualidade moderada; espaço para melhorar.

* Classificador de validade:

  * Fonte: `metrics/classifier.json`

  * `accuracy≈0.574`, `precision_0≈0.389`, `recall_0≈0.549`, `f1_0≈0.455`; `precision_1≈0.730`, `recall_1≈0.586`, `f1_1≈0.650`; `roc_auc≈0.612` ⇒ desempenho razoável, com possível desbalanceamento e limiar subótimo.

* Curva de aprendizado:

  * Fonte: `metrics/epochs.ndjson`

  * Tendência: `train_loss` desce de \~0.92 para \~0.45; `val_loss` estabiliza \~0.53–0.59 com patamares; há overfitting leve em blocos e melhora incremental após reduzir LR (scheduler ativo).

* Amostras efetivas:

  * `num_samples_trained=406`, `num_test_samples=61` (não chegou às 600 desejadas). Sinal de gargalo na coleta TQS ou baixa taxa de configurações válidas.

## Validação do Volume Geométrico das Vigas

* Código de cálculo:

  * Fonte: `utils/geometric_calculator.py`

  * Uso atual: `calculate_beams_geometric_volume_with_subtractions` + `get_geometric_concrete_volume` imprime pilares/vigas.

* Geração dos pontos das vigas:

  * Fonte: `geometry/length_input_processor.py` (`_find_beam_locations`)

  * Implementado: viga contínua entre faces do primeiro/último pilar por parede; divisão entre adjacentes se pilar intermediário > 70 cm.

* Checagem “face a face”:

  * Endpoints das vigas já são exatamente as faces (`min_face`/`max_face`). O método de volume ainda subtrai interseções com polígonos dos pilares, reforçando o comprimento efetivo (redundância segura).

* Sanidade numérica (exemplo manual):

  * Se `Vol. Vigas=2.2720 m³` e seção 0.20×0.40 ⇒ área=0.08 m² ⇒ comprimento total efetivo ≈ 28.4 m, compatível para 6–10 vigas face a face nas paredes do modelo.

## Problemas Identificados

* Regressão com `r2≈0.21` e `MAE≈55 kgf`: indica necessidade de mais dados, melhor regularização ou features.

* Classificador com `AUC≈0.612` e `precision_0` baixo: possível classe minoritária/threshold não ótimo.

* Confusão de matriz do classificador sugere amostra ampla (601 previsões), potencialmente fora do split padrão — vale confirmar escopo (treino/val/test) para não superestimar

## Melhorias Propostas

* Dados/Coleta:

  * Aumentar amostras válidas (tratar timeouts do TQS e aumentar `MAX_ITERATION_FACTOR`/`NUM_SAMPLES`).

  * Balancear classes inválido/válido na coleta para melhorar recall/precision da classe 0.

* Modelo/treino:

  * Testar `HuberLoss` para robustez a outliers; registrar `MAE`/`MAPE` por quantis.

  * Afinar `hidden_layers` e `dropout`; usar `tuning/tune_model.py` para grid restrito e fixar melhor combinação.

  * Early stopping já existe; manter scheduler e reduzir LR mais cedo (patience menor).

* Classificador de validade:

  * Ajustar limiar com base na `roc_curve.json` e gravar `validity_threshold.json` ótimo (Youden ou custo ponderado).

  * Calibrar probabilidades (Platt isotônico) se necessário.

* Features:

  * Revisar `FeatureEngineer.feature_names()` e adicionar métricas específicas de vigas pós-divisão (ex.: soma dos comprimentos adjacentes, max gap entre pilares).

* Traço/Verificação de volume:

  * Exportar por parede: comprimento efetivo total e número de vigas; CSV `outputs/results/beam_breakdown.csv`.

  * Adicionar validações unitárias simples (comprimento ‘face a face’ = diferença de faces quando coluna intermediária não supera limiar).

## Entregáveis

* Ajustes de treino/hyperparams e coleta para aumentar qualidade (novo experimento).

* Refinamento do limiar do classificador com base em ROC.

* Relatórios adicionais de geometria de vigas (por parede) e testes de sanidade.

## Confirmação

Posso implementar: (1) ajustes na coleta e hiperparâmetros, (2) calibração do classificador/limiar, (3) export detalhado de vigas por parede + testes de sanidade do volume face a face, e (4) reexecutar treino e gerar

Tome muito cuidado para não estragar nada que já funcione.

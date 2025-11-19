## Objetivo
- Adicionar debug/logs úteis e artefatos de métricas.
- Melhorar treino do surrogate com HuberLoss, weight decay e scheduler.
- Fortalecer o classificador com métricas salvas e class_weight.

## Mudanças
- `config/settings.py` (NeuralNetConfig): adicionar `LOSS_TYPE` ("mse"|"huber"), `WEIGHT_DECAY`, `LR_SCHEDULER` (bool), `LR_SCHEDULER_PATIENCE`, `LR_SCHEDULER_FACTOR`.
- `models/nn_manager.py`:
  - Selecionar loss por `LOSS_TYPE` (MSELoss/HuberLoss).
  - Passar `weight_decay` ao Adam.
  - Adicionar `ReduceLROnPlateau` com paciência e fator; `scheduler.step(val_loss)`.
- `main.py`:
  - Configurar `logging.basicConfig` no `main()`.
  - Após treinar o classificador, calcular métricas (accuracy, ROC-AUC) e salvar JSON `validity_metrics.json`.

## Validação
- Rodar `pytest -q` (tests existentes).
- Verificar que treino executa e salva artefatos (classificador + métricas) sem erros.

## Entregáveis
- Código atualizado em `config/settings.py`, `models/nn_manager.py`, `main.py`.
- Artefatos `validity_classifier.pkl` e `validity_metrics.json` gerados durante o treino.
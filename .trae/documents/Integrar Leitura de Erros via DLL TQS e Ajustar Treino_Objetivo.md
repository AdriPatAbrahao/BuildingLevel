## Objetivo
- Implementar verificação de erros graves usando as DLLs TQS (`NGERERRO` e `NMSGERRO`) em Python.
- Integrar a validação no fluxo TQS, rotular amostras válidas/inválidas.
- Treinar um classificador de validade e excluir amostras inválidas do surrogate de aço.
- Penalizar a função objetivo com a probabilidade de invalidez.

## O que já existe
- Penalização na função objetivo com limiar/penalidade: `optimization/objective_function.py:118–123`, parâmetros em `config/settings.py:44–46`.
- Inferência suporta `validity_classifier.pkl` e calcula probabilidade de invalidez: `inference.py:68–76, 94–112, 257–259`.
- Treino atual não cria nem salva `validity_classifier.pkl` e não rotula validade.

## Implementação
1) Leitura de erros via DLL
- Criar `tqs_interface/tqs_errors.py` com `TQSErrorReader` usando `ctypes.WinDLL` para carregar `NGERERRO.dll` e `NMSGERRO.dll` (diretório configurável por env var `BUILDOPT_TQS_DLL_DIR` e fallback padrão).
- Mapear funções usadas no exemplo C# (open/close, enumerar programas/erros, obter classificação e descrição) e iterar sobre três diretórios:
  - `C:\TQS\OptimizedBuilding\Tipo\VIGAS`
  - `C:\TQS\OptimizedBuilding\PILAR`
  - `C:\TQS\OptimizedBuilding\ESPACIAL`
- Estratégia: `os.chdir(path)` antes de `ERR_OPEN/ERRO_OPEN` para que as DLLs leiam o contexto correto.
- Retornar lista de erros críticos (CLASSIFICATION == 2) com número do elemento e cabeçalho.

2) Integração no fluxo TQS
- Em `main._execute_tqs_analysis_and_get_results`, após `RunModel` e extração de `RESDES.HTM`, chamar o `TQSErrorReader` para obter erros críticos.
- Retornar também um flag `is_valid` baseado na presença de erros críticos.

3) Coleta e rotulagem
- Em `_collect_training_data`:
  - Construir `feature_vectors` e `output_values` como hoje para amostras válidas.
  - Paralelamente construir `clf_features` e `clf_labels` (1 válido, 0 inválido) para todas as amostras processadas.
  - Excluir amostras inválidas de `output_values` (não treinar aço com inválidas).

4) Treinar classificador
- Em `_train_and_evaluate`:
  - Treinar um classificador (e.g., `sklearn.linear_model.LogisticRegression` ou `RandomForestClassifier`) com `clf_features`/`clf_labels`.
  - Persistir em `<experiment_dir>/validity_classifier.pkl` (compatível com `inference.py`).

5) Documentação
- Docstrings (inglês) em `tqs_errors.py` e nos métodos ajustados de `main.py`.
- Comentários mínimos em código conforme padrão do projeto.

## Validação
- Executar `pytest` para garantir ausência de regressão.
- Smoke test local: simular cenário onde DLL retorna erro para um dos diretórios e confirmar que a amostra é rotulada inválida e excluída do treino de aço.

## Entregáveis
- Novo módulo `tqs_errors.py` e alterações em `main.py`.
- Classificador salvo como `validity_classifier.pkl` durante o treino.
- Relatório breve das mudanças com referências por arquivo/linha.

Posso aplicar as mudanças e validar agora?
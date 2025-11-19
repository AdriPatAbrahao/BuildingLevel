## Visão Geral
- Auditar e aprimorar o projeto sem alterar o comportamento externo.
- Corrigir pequenos defeitos, melhorar robustez e performance onde seguro.
- Adicionar documentação (docstrings em inglês, estilo NumPy) com exemplos.

## Achados-Chave (com referências)
- Erro de fluxo em `_check_structural_errors` faz retorno prematuro no loop e não espera arquivo: `tqs_interface/tqs_exec.py:53–60` (definição em `tqs_interface/tqs_exec.py:42`).
- `EXPERIMENT_ID` hardcoded em inferência, acopla execução ao ambiente: `inference.py:28,31–35`.
- Parâmetros econômicos e penalidades fixos na função objetivo: `optimization/objective_function.py:15–20,68–77`.
- CSV de vetores finais usa formatação com separador de milhar (`,`) indevida: `utils/file_handler.py:52–57`.
- GA avalia população de forma estritamente sequencial; oportunidade de paralelizar sem mudar resultados: `optimization/optimizer.py:60–64,119–121`.
- Validações e compatibilidades de features na inferência estão boas, mas faltam testes cobrindo pipelines e otimizador: `inference.py:235–241,296–305`.

## Correções Obrigatórias (não quebram funcionalidade)
- Ajustar controle de fluxo e timeout em `_check_structural_errors` para realmente aguardar o arquivo e só então decidir; remover `return` prematuro e manter `time.sleep` dentro do loop: `tqs_interface/tqs_exec.py:53–60`.
- Remover separador de milhar e padronizar ponto decimal no CSV final; usar `"{valor:.6f}"` em vez de `":,.6f"`: `utils/file_handler.py:52–57`.
- Externalizar `EXPERIMENT_ID` via variável de ambiente (`BUILDOPT_EXPERIMENT_ID`) ou arquivo de config; manter fallback seguro e mensagens claras: `inference.py:28,31–35,47–55`.
- Mover preços e penalidades para `config/settings.py`/snapshot e ler na `ObjectiveFunction`; logar os valores carregados: `optimization/objective_function.py:15–20,26–29`.

## Melhorias Opcionais (comportamento inalterado)
- Paralelizar avaliação de custos do GA com `concurrent.futures` ou `multiprocessing` mantendo determinismo por semente; limitar workers por CPU: `optimization/optimizer.py:60–64`.
- Padronizar logging com `logging` (níveis INFO/WARN/ERROR) em TQS e inferência; substituir `print` onde apropriado: `tqs_interface/tqs_exec.py:52,63,67,70–73` e `inference.py:270–318`.
- Tornar leitura CSV mais robusta (validação de colunas, mensagens, suporte a `decimal='.'` quando aplicável): `geometry/length_input_processor.py:31–47`.
- Parametrizar caminhos (TQS, outputs) via `config`/`.env` para portabilidade: `config/paths.py` e usos em `tqs_interface/*`.

## Plano de Documentação (estilo NumPy, inglês)
- Adicionar docstrings consistentes em funções/métodos/classes principais:
  - `tqs_interface/tqs_exec.py`: `TQSCriticalError`, `_cleanup_report_files`, `_read_html_file`, `_check_structural_errors`, `RunModel`.
  - `inference.py`: `BuildingInference` (métodos `_validate_experiment_snapshot`, `predict_from_csv`, `run_comparison`, `_execute_full_tqs_analysis`, `_generate_report`).
  - `optimization/objective_function.py`: `ObjectiveFunction` e `calculate_cost`, `_discretize_vector`.
  - `optimization/optimizer.py`: `OptimizeResult`, `GeneticOptimizer.run` e helpers.
  - `geometry/length_input_processor.py`: `LengthProcessor` e métodos principais.
  - `utils/feature_pipeline.py`: `FeaturePipeline` e métodos.
- Para cada docstring: propósito, parâmetros, retornos, exceções levantadas e exemplo de uso breve.
- Exemplos de uso (in-line):
  - `ObjectiveFunction.calculate_cost(vector)` com vetor contínuo e discretização.
  - `BuildingInference.predict_from_csv(path_or_buffer)` com `StringIO`.
  - `GeneticOptimizer.run()` retornando `OptimizeResult`.
- Gerar documentação HTML com Sphinx + napoleon (NumPy docstrings) sem mudar APIs.

## Validação
- Rodar suíte existente (`tests/geometry/*`, `tests/utils/*`) e adicionar smoke tests para:
  - `_check_structural_errors` (simular ausência/presença do `PGLOERR.HTM`).
  - `ObjectiveFunction` (discretização, penalização, custo com parâmetros da config).
  - `GeneticOptimizer` (respeito a limites, elitismo, critério de parada).
  - `FeaturePipeline` (load/transform/inverse_transform). 
- Backward compatibility: manter assinaturas, caminhos e formatos de saída; só mover constantes para config com mesmos nomes e valores default.
- Verificar regressões com amostra conhecida (CSV semente) e comparar outputs antes/depois.

## Entregáveis
- Relatório detalhando melhorias (este plano + difs com referências por arquivo/linha).
- Lista de mudanças: "Obrigatórias" vs "Opcionais" claramente separadas.
- Docstrings adicionadas e pacote Sphinx (config + instruções de build) para gerar docs.

## Ordem de Execução
1) Corrigir `_check_structural_errors` e formatação de CSV final.
2) Externalizar `EXPERIMENT_ID` e parâmetros da função objetivo para `config`.
3) Adicionar docstrings e exemplos conforme plano.
4) Padronizar logging nos pontos críticos.
5) (Opcional) Paralelizar GA, robustez CSV e parametrização de caminhos.
6) Rodar e ajustar testes; comparar outputs e validar compatibilidade.

Confirma prosseguir com as correções obrigatórias, documentação e validações descritas?
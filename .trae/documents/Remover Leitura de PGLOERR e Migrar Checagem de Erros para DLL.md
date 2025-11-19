## Objetivo
- Eliminar toda a lógica de leitura do arquivo `PGLOERR.HTM` e migrar a verificação de erros críticos para a API/DLL do TQS, garantindo robustez e documentação consistente.

## Localizações Afetadas
- `tqs_interface/tqs_exec.py`: `_read_html_file`, `_check_structural_errors`, uso de `TQS_ERROR_REPORT_FILE/TQS_FATAL_ERROR_MARKER`, limpeza de `PGLOERR.HTM`, chamada e lançamento de `TQSCriticalError`.
- `config/settings.py`: `BuildingConfig.TQS_ERROR_REPORT_FILE`, `BuildingConfig.TQS_FATAL_ERROR_MARKER`.

## Mudanças Obrigatórias
1) Remover leitura de HTML e referências ao PGLOERR
- Excluir `_read_html_file` e `_check_structural_errors` em `tqs_interface/tqs_exec.py` (definições iniciando em `tqs_interface/tqs_exec.py:33` e `tqs_interface/tqs_exec.py:42`).
- Remover mensagens como `skipping PGLOERR check` (`tqs_interface/tqs_exec.py:81`).
- Remover docstrings que citam `PGLOERR.HTM` em `TQSCriticalError` e `RunModel`; atualizar para mencionar verificação via DLL.

2) Ajustar limpeza de arquivos temporários
- Atualizar `_cleanup_report_files()` para não referenciar/apagar `PGLOERR.HTM`, mantendo apenas a remoção de `RESDES.HTM` (`tqs_interface/tqs_exec.py:17–31`).

3) Atualizar configuração
- Remover `BuildingConfig.TQS_ERROR_REPORT_FILE` e `BuildingConfig.TQS_FATAL_ERROR_MARKER` de `config/settings.py` (`config/settings.py:10–11`).
- Garantir que nenhum outro módulo dependa desses campos (grep mostra apenas `tqs_exec.py`).

4) Verificação de erros via DLL (tratamento de exceções)
- Introduzir verificação baseada em DLL após `job.Execute()` em `RunModel`:
  - Envolver `job.Execute()` em `try/except` para capturar exceções da DLL e lançar `TQSCriticalError` com mensagem detalhada.
  - Se a API expuser indicador de erros (e.g., retorno não-zero ou método do relatório estrutural), consultar e lançar `TQSCriticalError` quando crítico.
  - Logar via `TQSUtil.writef` com níveis informativos e mensagens claras.

5) Limpeza de referências cruzadas
- Confirmar que `inference.py` e `tqs_interface/tqs_manager.py` não dependem de `PGLOERR.HTM` (nenhuma referência encontrada). Nenhuma alteração necessária.

6) Documentação no código-fonte
- Adicionar/atualizar docstrings (inglês, estilo NumPy) em:
  - `TQSCriticalError` (agora descrevendo a origem via DLL).
  - `_cleanup_report_files` (detalhando limpeza apenas de `RESDES.HTM`).
  - `RunModel` (parâmetros, exceções, comportamento e nota sobre verificação via DLL).

## Validação
- Executar a suíte de testes existente (`pytest`) para garantir que não há regressões (não há testes de TQS, mas garante que as mudanças não afetam utilitários/geométrico).
- Rodar um fluxo de inferência de exemplo que chama `RunModel` (fumace) para validar tratamento de exceções da DLL sem dependência de `PGLOERR`. 
- Revisar logs gerados por `TQSUtil.writef` em caso de erro para confirmar mensagens adequadas.

## Entregáveis
- Código atualizado sem referências ao `PGLOERR.HTM`.
- Tratamento de exceções no `RunModel` usando a DLL do TQS.
- Docstrings atualizadas e consistentes em `tqs_interface/tqs_exec.py`.
- Relatório breve das mudanças (no commit/descrição) e referências por arquivo/linha.

Posso aplicar agora as alterações e validar com a suíte de testes?
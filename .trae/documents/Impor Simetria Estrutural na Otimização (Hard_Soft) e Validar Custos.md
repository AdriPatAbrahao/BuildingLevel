## Objetivo
Impor simetria estrutural rígida (hard-constraint) usando `group_id` já presente nos CSVs (`Building1b.csv`, `Building1.csv`) para que segmentos espelhados compartilhem o mesmo valor de comprimento em todo o fluxo: leitura, geração de variações, criação de geometria, e otimização.

## Leituras e Mapeamento de Grupos
- `geometry/length_input_processor.py`
  - Ler opcionalmente a coluna `group_id` (string/int). Quando existir, construir um dicionário: `group_id -> [segment_indices]`.
  - Expor `self.group_map` para uso em variações.

## Geração de Variações (Treinamento)
- `LengthProcessor.generate_variation(...)`
  - Tornar a variação “group-aware”: ao ajustar `length` de um segmento pertencente a um `group_id`, aplicar o mesmo incremento para todos os segmentos do grupo.
  - Respeitar `maxlength` e discretização por passo (`step`) no nível do grupo.

## Espaço de Projeto com Simetria (Otimização)
- `optimization/design_space.py`
  - Ao carregar o seed, ler `group_id` e criar `symmetry_groups = {group_id: [row_indices]}`.
  - Reduzir dimensionalidade: definir `lower_bounds`, `upper_bounds` e `initial_guess` por grupo (um “driver” por `group_id`).
  - Em `create_geometry_from_vector(...)`, replicar o valor do driver para todos os segmentos do mesmo `group_id` ao montar o DataFrame (`x;y;dx;dy;length;maxlength`).

## Função Objetivo
- `optimization/objective_function.py`
  - Sem alterações funcionais: a geometria recebida já estará simétrica. O custo segue calculado sobre aço (surrogate) e concreto geométrico.

## Inferência e Validação
- `inference.py`
  - Compatível: o método `predict_from_csv` usará o CSV já simétrico.
  - (Opcional) Adicionar utilitário rápido que leia um CSV, compute e exiba “max_diff_por_grupo = max |len_i − len_j|” como sanidade (deve ser 0 com hard-constraint).

## Logs e Auditoria
- `outputs/results/optimization_log.json`
  - (Opcional) Registrar por iteração o `max_diff_por_grupo` (esperado 0) para confirmar que o hard-constraint permaneceu.

## Riscos e Cuidados
- Segmentar apenas os pares/grupos que realmente devem ser simétricos; elementos sem simetria desejada devem ter `group_id` distinto ou ausente.
- Garantir que `maxlength` e `bounds` por grupo sejam consistentes entre membros (usar o mínimo dos máximos, e o máximo dos mínimos para segurança).

## Próximo Passo
- Implementar a leitura de `group_id` e o mapeamento de grupos no `LengthProcessor`.
- Tornar as variações do treinamento “group-aware”.
- Reduzir o espaço no `DesignSpace` para drivers por `group_id` e replicar nos espelhos ao construir a geometria.
- Rodar uma otimização curta (10–20 gerações) e validar que a solução ótima é simétrica e que custos/volumes permanecem coerentes.
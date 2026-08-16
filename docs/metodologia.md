# Metodologia: Sistema de Otimização de Edificações via Rede Neural Substituta e Algoritmo Genético

> **Finalidade deste documento:** fonte técnica completa para redação do capítulo de Metodologia da tese de doutorado. Cobre a arquitetura computacional, as formulações matemáticas, as decisões de engenharia estrutural e as escolhas de ciência de dados de cada componente do sistema.

---

## Sumário

1. [Visão Geral do Sistema](#1-visão-geral-do-sistema)
2. [Edificação Estudada e Modelo Estrutural](#2-edificação-estudada-e-modelo-estrutural)
3. [Interface com o TQS — Análise Estrutural Computacional](#3-interface-com-o-tqs--análise-estrutural-computacional)
4. [Espaço de Projeto e Variáveis de Decisão](#4-espaço-de-projeto-e-variáveis-de-decisão)
5. [Coleta de Dados de Treinamento](#5-coleta-de-dados-de-treinamento)
6. [Engenharia de Atributos (Feature Engineering)](#6-engenharia-de-atributos-feature-engineering)
7. [Pipeline de Normalização](#7-pipeline-de-normalização)
8. [Arquitetura da Rede Neural](#8-arquitetura-da-rede-neural)
9. [Treinamento e Regularização](#9-treinamento-e-regularização)
10. [Avaliação do Modelo Substituto](#10-avaliação-do-modelo-substituto)
11. [Classificador de Validade Estrutural](#11-classificador-de-validade-estrutural)
12. [Otimização por Algoritmo Genético](#12-otimização-por-algoritmo-genético)
13. [Função Objetivo](#13-função-objetivo)
14. [Inferência e Predição em Produção](#14-inferência-e-predição-em-produção)
15. [Gerenciamento de Experimentos e Reprodutibilidade](#15-gerenciamento-de-experimentos-e-reprodutibilidade)
16. [Fluxo Completo de Execução](#16-fluxo-completo-de-execução)
17. [Parâmetros e Hiperparâmetros Consolidados](#17-parâmetros-e-hiperparâmetros-consolidados)

---

## 1. Visão Geral do Sistema

O sistema desenvolvido implementa uma metodologia de **otimização estrutural baseada em modelo substituto** (*surrogate-based optimization* — SBO). O objetivo é minimizar o custo combinado de aço e concreto de uma edificação de concreto armado, sujeita à restrição de que o projeto seja estruturalmente válido segundo as normas brasileiras.

O processo tem quatro fases principais:

```
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 1 — COLETA DE DADOS                                           │
│  Gerar N variações geométricas → Analisar no TQS → Registrar saídas │
├─────────────────────────────────────────────────────────────────────┤
│  FASE 2 — TREINAMENTO DO MODELO SUBSTITUTO                          │
│  Extrair 23 features → Normalizar → Treinar DNN → Avaliar           │
├─────────────────────────────────────────────────────────────────────┤
│  FASE 3 — OTIMIZAÇÃO                                                │
│  Algoritmo Genético → Consultar DNN (ms) → Minimizar custo          │
├─────────────────────────────────────────────────────────────────────┤
│  FASE 4 — VALIDAÇÃO                                                 │
│  Solução ótima → Executar TQS completo → Confirmar resultado        │
└─────────────────────────────────────────────────────────────────────┘
```

**Justificativa da abordagem:** A análise estrutural completa pelo TQS (dimensionamento de vigas e pilares conforme NBR 6118) tem custo computacional da ordem de 30–180 segundos por avaliação. Uma otimização direta com Algoritmo Genético exigiria dezenas de milhares de avaliações, tornando inviável a exploração ampla do espaço de projeto. O modelo substituto reduz o custo de cada avaliação para menos de 1 ms (aceleração superior a 30.000×), viabilizando populações genéticas amplas e múltiplas gerações.

---

## 2. Edificação Estudada e Modelo Estrutural

### 2.1 Planta Estrutural

A edificação é modelada por um sistema de pilares e vigas em concreto armado. Os pilares estão localizados nos seguintes nós em coordenadas cartesianas (em centímetros):

| Pilar | x (cm) | y (cm) |
|-------|--------|--------|
| P1    | 10     | 10     |
| P2    | 360    | 10     |
| P3    | 10     | 410    |
| P4    | 360    | 410    |
| P5    | 710    | 410    |
| P6    | 360    | 810    |
| P7    | 710    | 810    |

A geometria da planta resulta em um edifício com formato irregular (aproximação de "L" ou "T"), com dois vãos horizontais de ~350 cm cada e dois vãos verticais de ~400 cm cada. As lajes adicionais têm centros em (180, 105), (540, 105), (180, 615) e (540, 615) cm.

### 2.2 Representação das Seções Transversais dos Pilares

Cada pilar é representado por um ou mais **segmentos** que definem a geometria de sua seção transversal. Um segmento é descrito por:

| Campo     | Significado                                        |
|-----------|----------------------------------------------------|
| `x, y`    | Ponto de ancoragem do segmento (cm)                |
| `dx, dy`  | Vetor de direção normalizado (eixo de crescimento) |
| `length`  | Comprimento atual do segmento (cm) — variável      |
| `maxlength` | Comprimento máximo permitido (cm) — limite superior |
| `group_id` | Identificador de grupo de simetria (opcional)      |

O ponto final de cada segmento é calculado como:

```
end_x = x + dx × length
end_y = y + dy × length
```

Segmentos com o mesmo `group_id` pertencem ao mesmo grupo de simetria e recebem sempre o mesmo comprimento durante o processo de otimização, garantindo que pilares simétricos em planta mantenham dimensões iguais.

### 2.3 Geração dos Polígonos de Pilar

A classe `LengthProcessor` converte os segmentos em polígonos Shapely que representam a seção transversal dos pilares. O processo é:

1. **Agrupamento de segmentos conectados:** Segmentos que compartilham um nó são agrupados para compor a seção de um único pilar.
2. **Geração de retângulos:** Cada segmento gera um retângulo com espessura total de 20 cm, estendendo-se 10 cm para cada lado de seu eixo (`DEFAULT_BEAM_WIDTH_CM / 2 = 10 cm`).
3. **União de polígonos:** Os retângulos de cada grupo são unidos via operação booleana (Shapely `union`), produzindo a seção transversal resultante — que pode ser retangular, em L, em T etc.
4. **Restrição de seção retangular:** O `DesignSpace` detecta automaticamente pares de grupos perpendiculares cujos retângulos da geometria semente possuem área de sobreposição positiva. A detecção é geométrica e também reconhece cantos em que os eixos têm pontos iniciais deslocados. Para cada par, somente o grupo com maior desvio em relação ao comprimento inicial é ativado, garantindo que cada pilar cresça em apenas uma direção. O simples contato pela borda entre pilares distintos não cria uma restrição.

### 2.4 Vigas

As vigas são definidas ao longo dos segmentos de parede (`VectorConfig.WALL_SEGMENTS`) que não estejam ocupados por pilares. As paredes seguem:

```
Horizontal: y = 10 cm  (x: 0→720)
Horizontal: y = 410 cm (x: 0→720)
Horizontal: y = 810 cm (x: 0→720)
Vertical:   x = 10 cm  (y: 0→820)
Vertical:   x = 360 cm (y: 0→820)
Vertical:   x = 710 cm (y: 0→820)
```

As vigas têm dimensões fixas: **b = 20 cm × h = 40 cm** (`DEFAULT_BEAM_WIDTH_CM × DEFAULT_BEAM_HEIGHT_CM`). O comprimento efetivo de cada viga é calculado descontando os trechos que coincidem com seções de pilares:

```
L_efetivo = L_total - Σ comprimento_de_interseção_com_pilares
```

O volume geométrico total de concreto do pavimento segue o contrato:

```
V_total = V_pilares + V_vigas_líquidas + V_lajes
```

O desconto nas vigas evita dupla contagem do concreto nas interseções com os pilares. O volume fixo das quatro lajes é `6,0192 m³`, obtido por `4 × 3,30 m × 3,80 m × 0,12 m`. A área de fôrma calculada pelo sistema corresponde somente às faces laterais dos pilares (`perímetro × pé-direito`), não à fôrma total de vigas e lajes.

### 2.5 Cargas

| Tipo de Carga              | Valor                     |
|---------------------------|---------------------------|
| Carga permanente de viga  | 2,0 tf/m                  |
| Carga acidental de viga   | 1,0 tf/m                  |
| Carga permanente de laje  | 1,0 tf/m²                 |
| Carga acidental de laje   | 1,0 tf/m²                 |
| Espessura da laje         | 12 cm                     |

---

## 3. Interface com o TQS — Análise Estrutural Computacional

### 3.1 Papel do TQS no Sistema

O TQS é um *software* comercial brasileiro para projeto estrutural de edifícios em concreto armado, amplamente adotado no mercado nacional. No presente sistema, ele desempenha o papel de **oráculo estrutural**: recebe um modelo geométrico e retorna o consumo de materiais após o dimensionamento automático conforme a NBR 6118.

O TQS é acionado exclusivamente via sua API de DLL (`TQSExec`), sem interface gráfica, o que permite automação e execução programática.

### 3.2 Fluxo de Análise Estrutural

Para cada amostra de treinamento, o fluxo executado é:

```
1. Criar modelo TQS (TQSModelManager)
   ├── Abrir base de dados do edifício
   ├── Inserir pilares com seções definidas pelos polígonos
   ├── Inserir vigas e lajes com cargas padronizadas
   └── Fechar modelo

2. Executar processamento global (RunModel)
   ├── TQSExec.TaskFolder  → aponta para diretório do edifício
   ├── TQSExec.TaskGlobalProc → esforços em vigas (modo 1) e dimensionamento/desenho de pilares (modo 2)
   └── TQSExec.TaskStructuralReport → gera relatório RESDES.HTM

3. Ler resultados (extract_material_summary)
   └── Parsear RESDES.HTM → aço (kgf) e concreto (m³)
```

### 3.3 Parâmetros do Processamento Global TQS

| Parâmetro         | Valor | Significado                        |
|-------------------|-------|------------------------------------|
| `gridSlabsTrnsf`  | 0     | Sem transferência de grelha de laje |
| `slabs`           | 0     | Lajes não processadas individualmente |
| `beams`           | 1     | Somente esforços; sem dimensionamento/detalhamento de armadura das vigas |
| `columns`         | 2     | Dimensionamento completo de pilares |

### 3.4 Extração do Consumo de Materiais

O arquivo `RESDES.HTM` gerado pelo TQS contém o resumo do dimensionamento estrutural. A função `extract_material_summary` realiza *parsing* HTML/texto com múltiplas tentativas de codificação (UTF-8, Latin-1, CP1252) e extrai os valores totais de:
- **Aço (kgf):** consumo de armadura dos pilares — único elemento processado com armadura no modelo (vigas e lajes não são individualmente dimensionadas com detalhamento de aço neste pipeline)
- **Concreto (m³):** volume total de concreto em pilares, vigas e lajes (todo o pavimento)

> **Nota de escopo:** o aço ($M_{s,\mathrm{col}}$) tem escopo "somente pilares", enquanto o concreto ($V_{c,b}$) e a área de forma ($A_{f,\mathrm{col}}$, calculada em `utils/geometric_calculator.py::calculate_column_formwork_area`, também somente pilares) têm escopos diferentes entre si — concreto é do pavimento inteiro, forma é só dos pilares. Essa distinção de escopo por elemento estrutural é independente da distinção de escopo por número de pavimentos (§3.1–3.3), que é a mesma (um pavimento tipo) para as três grandezas.

### 3.5 Gerenciamento de Hangs e Timeouts

O TQS é executado por chamada de DLL bloqueante (`job.Execute()`). Para prevenir que um travamento do processo TQS paralise indefinidamente a coleta de dados, o sistema implementa:

1. **Timeout com kill automático:** `RunModel` é executado em uma *thread* daemon com timeout configurável — padrão de 120 s no modo sequencial (`RunConfig.TQS_TIMEOUT_SEC`), 180 s no modo paralelo (`ParallelConfig.TIMEOUT_SEC`). Caso exceda o limite, o processo `NTQSHTM.EXE` é finalizado via `taskkill /F /IM NTQSHTM.EXE` e uma `TimeoutError` é lançada.
2. **Mapeamento de unidade virtual:** O TQS requer a unidade `T:` mapeada para `C:\TQSWV26A`. Em subprocessos, este mapeamento é refeito via `subst T: C:\TQSWV26A` no início de cada worker.
3. **Limpeza de arquivos antigos:** Antes de cada execução, o arquivo `RESDES.HTM` e demais arquivos temporários do slot são removidos, evitando leituras de resultados de runs anteriores.

---

## 4. Espaço de Projeto e Variáveis de Decisão

### 4.1 Definição das Variáveis

O espaço de projeto é definido pelo arquivo CSV semente (`BuildingInput.csv`). Cada linha do CSV representa um segmento de pilar; linhas com o mesmo `group_id` são vinculadas (variam juntas). O vetor de decisão `x ∈ ℝᵈ` tem dimensão `d = número_de_grupos_únicos`.

Para cada grupo `i`:
- **Limite inferior:** `lb_i = max(length_inicial)` para os segmentos do grupo
- **Limite superior:** `ub_i = min(maxlength)` para os segmentos do grupo
- **Valor inicial:** `x₀ = lb` (geometria da semente)

### 4.2 Restrição de Seção Retangular

O sistema detecta automaticamente pares de grupos que atuariam no mesmo nó físico em eixos perpendiculares. Para cada par `(gᵢ, gⱼ)`:

```
desvio_i = vector[i] - lb_i
desvio_j = vector[j] - lb_j

se desvio_i ≥ desvio_j:
    lengths[grupo_j] = length_semente_j   (grupo j resetado)
senão:
    lengths[grupo_i] = length_semente_i   (grupo i resetado)
```

Esta restrição garante que em cada nó apenas uma dimensão da seção transversal cresça por vez, mantendo a seção retangular e evitando perfis em L, T ou U que seriam incomuns em projeto de edifícios correntes.

### 4.3 Discretização

Durante a otimização, o vetor contínuo proposto pelo Algoritmo Genético é discretizado para o múltiplo mais próximo de `LENGTH_STEP_CM = 5 cm`. O mesmo parâmetro controla a geração das amostras, mantendo alinhados os domínios de treinamento e otimização:

```
x_discreto = round(x_contínuo / 20) × 20
x_discreto = clip(x_discreto, lb, ub)
```

---

## 5. Coleta de Dados de Treinamento

### 5.1 Estratégia de Amostragem

O conjunto de treinamento é construído por **amostragem aleatória com variação progressiva** da geometria semente. A cada iteração:

1. A geometria corrente (segmentos) é modificada por `generate_variation`, que altera aleatoriamente comprimentos dentro dos limites `[length, maxlength]`.
2. A nova geometria é submetida ao TQS para análise estrutural.
3. O resultado (aço kgf, concreto m³) é registrado junto ao vetor de features extraído.

O número alvo de amostras válidas é `NUM_SAMPLES = 2500` (valor atual de `RunConfig.NUM_SAMPLES`), com limite máximo de tentativas `max_iterações = NUM_SAMPLES × MAX_ITERATION_FACTOR = 12.500`.

### 5.2 Critério de Validade de Amostra

Uma amostra é considerada **válida** se:
- O TQS concluiu sem erro (`success = True`)
- O arquivo `RESDES.HTM` foi gerado e parseado com sucesso
- Os valores de aço e concreto são positivos e finitos
- A DLL de erros críticos está disponível e retorna zero erros de classe crítica
- (Quando ativados) os limites `STEEL_MIN_KGF`, `STEEL_MAX_KGF`, `CONCRETE_MIN_M3` são respeitados

A validação pela DLL é obrigatória (`VALIDITY_CHECK_DLL = True`) e opera em
modo *fail-closed*: indisponibilidade da DLL, API incompleta ou falha de leitura
rejeita a amostra, em vez de classificá-la silenciosamente como válida.

### 5.3 Coleta Paralela vs. Sequencial

O sistema suporta dois modos:

**Modo sequencial** (`ParallelConfig.ENABLED = False`):
- Executa no processo principal, usando o diretório `OptimizedBuilding`
- Sem conflito de recursos compartilhados do TQS
- Taxa: ~1 amostra/50 s → ~35 horas para 2500 amostras

**Modo paralelo** (`ParallelConfig.ENABLED = True`):
- `NUM_WORKERS` subprocessos, cada um com diretório isolado (`TrainBuild815_01`, `_02`, ...)
- Janela deslizante: novos jobs são submetidos à medida que resultados chegam
- Configuração validada atual: um worker no slot `TrainBuild815_01`
- O número de workers deve permanecer em 1 enquanto a limpeza/recuperação
  encerrar `NTQSHTM.EXE` pelo nome global do processo; aumentar esse valor pode
  fazer um worker interromper a análise de outro
- O checkpoint v3 preserva regressão, classificador, configurações válidas,
  hashes de deduplicação, estado do gerador aleatório e hash do CSV semente,
  com escrita atômica a cada 10 minutos
- A coleta pode ser executada sem treinamento com `main.py --collect-only`; uma
  execução interrompida é retomada com `--resume-run`, e o treinamento posterior
  usa `--train-from-checkpoint`

#### Validação escalonada antes da coleta completa

O teste 12 executou uma coleta limpa em duas etapas, sem treinamento: primeiro 10
amostras válidas e depois retomada do mesmo checkpoint até 30. O resultado final
foi 30 amostras válidas e 10 inválidas em 40 tentativas, sem configurações ou
hashes duplicados, com atributos finitos em todas as entradas. Os 10 registros
da primeira etapa foram preservados exatamente após a retomada. A coleta completa
fica liberada a partir desse checkpoint; o treinamento deve continuar separado e
somente depois da auditoria do conjunto completo.

#### Piloto de concorrência TQS

Antes de elevar `ParallelConfig.NUM_WORKERS`, o script
`scripts/validate_tqs_concurrency.py` compara as mesmas seis geometrias em uma
cópia sequencial e em duas cópias processadas simultaneamente. O teste usa o
checkpoint das 230 amostras apenas como fonte de geometrias, inclui o CSV
semente como sonda de validade, não modifica o checkpoint de produção e exige:

- coincidência de validade entre os modos;
- diferença máxima de 0,5 kgf no aço e 0,001 m³ no concreto;
- uso efetivo dos dois slots;
- pelo menos um caso inviável para testar a DLL de erros;
- ganho de vazão mínimo de 1,25 vez;
- nenhuma falha ou timeout.

Execução, com confirmação explícita do modo simultâneo:

```powershell
.\.venv\Scripts\python.exe -m scripts.validate_tqs_concurrency `
  --confirm-simultaneous-tqs
```

O resultado é gravado em
`outputs/validation/tqs_concurrency_pilot/summary.json`. Um timeout ainda pode
encerrar globalmente todas as instâncias `NTQSHTM.EXE`; portanto qualquer timeout
reprova o uso de dois workers na coleta definitiva.

### 5.4 Coleta da Configuração Semente

A primeira amostra coletada é sempre a configuração semente (geometria base), garantindo que o ponto de partida conhecido esteja representado no conjunto de treinamento.

### 5.5 Dupla Rotulagem para Classificador de Validade

Paralelamente ao rótulo regressivo (aço kgf), cada amostra recebe um rótulo binário de validade estrutural (`1` = válida, `0` = inválida), usado para treinar um classificador auxiliar. Amostras inválidas são retidas no conjunto do classificador mas excluídas do conjunto do regressor.

---

## 6. Engenharia de Atributos (Feature Engineering)

### 6.1 Motivação

A extração de atributos (*features*) transforma a geometria bruta (polígonos e definições de vigas) em um vetor numérico que capture as propriedades estruturalmente relevantes para predição do consumo de aço. Features mal escolhidas podem tornar o modelo incapaz de distinguir edifícios estruturalmente diferentes; features redundantes aumentam a dimensionalidade sem ganho preditivo.

O vetor de features tem **23 dimensões** no esquema v11, organizadas nos blocos funcionais descritos a seguir.

### 6.2 Bloco 1 — Estatísticas de Área de Pilares (4 features)

Calculadas sobre `A_i = área(polígono_pilar_i)` para todos os `n` pilares:

| Feature | Fórmula | Unidade |
|---------|---------|---------|
| `columns_total_area_cm2` | `Σ A_i` | cm² |
| `columns_std_area_cm2` | `std(A_i)` | cm² |
| `columns_min_area_cm2` | `min(A_i)` | cm² |
| `columns_max_area_cm2` | `max(A_i)` | cm² |

**Justificativa estrutural:** A área total da seção transversal correlaciona-se diretamente com a capacidade resistente à compressão dos pilares e com o volume de concreto. A dispersão (std) indica heterogeneidade entre pilares, que influencia a distribuição de esforços. A quantidade de pilares, constante em `n = 9`, e a área média, exatamente igual a `área_total/9`, são mantidas apenas em `FeatureEngineer.get_diagnostics()`.

### 6.3 Bloco 2 — Vãos Livres Direcionais de Vigas (6 features)

Cada linha de viga é recortada pela união geométrica dos pilares. Os componentes
restantes são os vãos físicos entre apoios, classificados nas direções X e Y:
```
vãos_livres = linha_da_viga − união_dos_pilares
```

| Feature | Fórmula | Unidade |
|---------|---------|---------|
| `beams_std_clear_span_x_cm` | desvio-padrão dos vãos em X | cm |
| `beams_std_clear_span_y_cm` | desvio-padrão dos vãos em Y | cm |
| `beams_max_clear_span_x_cm` | maior vão físico em X | cm |
| `beams_max_clear_span_y_cm` | maior vão físico em Y | cm |
| `beams_span_entropy_x` | entropia da distribuição dos vãos em X | — |
| `beams_span_entropy_y` | entropia da distribuição dos vãos em Y | — |

**Justificativa estrutural:** Embora o alvo seja somente o aço dos pilares, os
vãos, cargas e rigidezes das vigas modificam os esforços transferidos aos pilares.
A separação X/Y preserva a interação com a orientação das seções dos pilares.

### 6.4 Bloco 3 — Momentos de Inércia (3 features)

Para cada pilar `i`, os momentos de inércia centroidais `Ixx_i` e `Iyy_i` são calculados via **Teorema de Green** (integração de linha sobre o contorno do polígono), seguido do **Teorema dos Eixos Paralelos** para translação ao centroide. Antes da integração, a orientação dos anéis é normalizada: contorno externo anti-horário e vazios internos horários. Assim, o sinal não depende da ordem original dos vértices e os vazios são subtraídos corretamente:

```
Ixx_origem = (1/12) × Σ [(y_k² + y_k·y_{k+1} + y_{k+1}²) × (x_k·y_{k+1} − x_{k+1}·y_k)]
Iyy_origem = (1/12) × Σ [(x_k² + x_k·x_{k+1} + x_{k+1}²) × (x_k·y_{k+1} − x_{k+1}·y_k)]

Ixx_centroid = Ixx_origem − A × ȳ²
Iyy_centroid = Iyy_origem − A × x̄²
```

onde `(x̄, ȳ)` é o centroide do polígono (calculado pelo Shapely).

| Feature | Fórmula | Unidade |
|---------|---------|---------|
| `inertia_sum_Ix` | `Σ Ixx_i` | cm⁴ |
| `inertia_sum_Iy` | `Σ Iyy_i` | cm⁴ |
| `inertia_ratio_Iy_over_Ix` | `Σ Iyy / (Σ Ixx + ε)` | — |

As médias `mean(Ixx_i)` e `mean(Iyy_i)` são mantidas somente em
`FeatureEngineer.get_diagnostics()`, pois, com nove pilares fixos, são exatamente
iguais às somas divididas por nove e não acrescentam informação ao modelo.

**Justificativa estrutural:** A inércia da seção transversal determina a rigidez à flexão e se relaciona com o consumo de armadura em regime de flexo-compressão. A razão `Iyy/Ixx` captura a assimetria direcional da rigidez estrutural.

### 6.5 Grandezas Derivadas de Vigas (somente diagnóstico)

Quantidade de objetos de viga, quantidade de vãos por direção, comprimentos
totais, médias, percentis 95 e volume geométrico são mantidos em
`FeatureEngineer.get_diagnostics()`. Neste edifício existem sempre seis vãos
físicos por direção; as médias são derivadas dos totais, os percentis 95
coincidem com os máximos e o volume é proporcional à soma dos comprimentos
porque a seção das vigas é fixa. Os comprimentos totais foram retirados do modelo
no schema v9 por serem transformações lineares exatas das dimensões médias dos
pilares sob a malha e as restrições geométricas atuais.

### 6.6 Compacidade (somente diagnóstico)

| Métrica de diagnóstico | Fórmula |
|------------------------|---------|
| `columns_mean_compactness` | `mean(4π × A_i / P_i²)` |

**Justificativa:** A compacidade (`4πA/P²`) atinge máximo 1,0 para círculo e valores menores para seções alongadas. Para cada seção, ela satisfaz exatamente `compacidade = 4π/(P/√A)²`; por isso foi retirada do modelo no schema v8 e mantida em `FeatureEngineer.get_diagnostics()`. Os três resumos de perímetro também permanecem apenas nos diagnósticos.

### 6.7 Bloco 6a — Distribuição Espacial com Referência Fixa (4 features)

O centro de cargas e as dimensões da planta são entradas explícitas do edifício
em `BuildingConfig`. Para o edifício atual:

```text
LOAD_CENTER_CM = (360, 410)
PLAN_WIDTH_CM = 720
PLAN_LENGTH_CM = 820
```

Os pontos de inserção das lajes não são usados como centroides. Para cada
pilar, definem-se coordenadas adimensionais em relação ao centro de cargas:

```text
dx_i = (x_i - x_carga) / largura_planta
dy_i = (y_i - y_carga) / comprimento_planta
```

Quatro grandezas que variam no espaço de projeto entram no modelo:

| Feature de treinamento | Fórmula / significado |
|------------------------|-----------------------|
| `column_area_spread_x_norm` | `Σ(A_i dx_i²) / ΣA_i` |
| `column_area_spread_y_norm` | `Σ(A_i dy_i²) / ΣA_i` |
| `columns_stiffness_spread_x_response_norm` | `Σ(Iyy_i dy_i²) / ΣIyy_i` |
| `columns_stiffness_spread_y_response_norm` | `Σ(Ixx_i dx_i²) / ΣIxx_i` |

Para uma translação em X, a rigidez à flexão do pilar é proporcional a `E Iyy`
e o braço associado à resposta torcional está em Y; para uma translação em Y,
a rigidez é proporcional a `E Ixx` e o braço está em X. Como o módulo de
elasticidade e o pé-direito são comuns aos pilares deste edifício, eles se
cancelam nas razões. Essas métricas são descritores geométricos da distribuição
de rigidez e não substituem a análise global de vínculos realizada pelo TQS.

As seis grandezas constantes pelas restrições de simetria são calculadas por
`FeatureEngineer.get_spatial_diagnostics()`, mas não entram na rede:

| Métrica de diagnóstico | Fórmula / significado |
|------------------------|-----------------------|
| `column_area_offset_x_norm` | `Σ(A_i dx_i) / ΣA_i` |
| `column_area_offset_y_norm` | `Σ(A_i dy_i) / ΣA_i` |
| `column_area_coupling_xy_norm` | `Σ(A_i dx_i dy_i) / ΣA_i` |
| `max_quadrant_area_ratio_fixed` | maior fração de área entre os quatro quadrantes fixos |
| `stiffness_ecc_x_norm` | `Σ(Iyy_i dx_i) / ΣIyy_i` |
| `stiffness_ecc_y_norm` | `Σ(Ixx_i dy_i) / ΣIxx_i` |

As áreas dos pilares que cruzam um eixo são divididas geometricamente entre os
quadrantes, evitando atribuição artificial ao lado positivo. Os deslocamentos
com sinal detectam desequilíbrio; as dispersões distinguem pilares centrais,
laterais e de canto; o termo cruzado identifica concentração diagonal. As duas
últimas métricas auditam o efeito da rotação das seções sobre a rigidez. Caso as
restrições de simetria sejam alteradas, a seleção de features deve ser revista.

### 6.8 Bloco 6b — Fatores de Forma da Seção (2 features)

As duas métricas `P/√A` descrevem o alongamento geométrico da seção, mas não são
a esbeltez do elemento `L_e/r`:

```text
shape_factor_i = Perímetro_i / √(A_i)
```

| Feature | Fórmula / significado |
|---------|-----------------------|
| `columns_mean_shape_factor` | média de `shape_factor_i` |
| `columns_p95_shape_factor` | percentil 95 de `shape_factor_i` |

A média representa a forma típica das seções e o percentil 95 destaca as seções
mais alongadas. Os nomes anteriores com `slenderness` foram removidos no schema
v8 para não confundir fator de forma com esbeltez estrutural.

### 6.9 Bloco 6c — Balanço Direcional dos Raios de Giração (1 feature)

O raio de giração expressa a distribuição da área em relação ao eixo de flexão:

```
r_x_i = √(Ixx_i / A_i)
r_y_i = √(Iyy_i / A_i)
r_min_i = min(r_x_i, r_y_i)
```

As médias direcionais são combinadas em uma única feature adimensional:

```text
radius_balance = [mean(r_y) - mean(r_x)] / [mean(r_y) + mean(r_x)]
```

| Feature | Fórmula / significado |
|---------|-----------------------|
| `columns_mean_radius_gyration_directional_balance` | balanço normalizado entre os raios médios; positivo indica predominância da dimensão em X e negativo, em Y |

**Justificativa:** O balanço preserva o efeito da rotação sem duplicar a área total e os comprimentos livres totais. `mean(r_x)`, `mean(r_y)`, `mean(r_min)` e `min(r_min)` continuam disponíveis em `FeatureEngineer.get_diagnostics()`. Como todas as seções mantêm uma dimensão mínima de 20 cm, as duas últimas são constantes em aproximadamente `5,7735 cm`.

### 6.10 Bloco 6d — Descritores Logarítmicos de Orientação (3 features)

A orientação e o alongamento são obtidos dos momentos de inércia centroidais:

```
log_aspect_i = 0,5 × log(|Iyy_i| / |Ixx_i|) = log(b_i/h_i)
```

Para seção retangular com dimensão `b` em X e `h` em Y, uma rotação de 90°
transforma `log(b/h)` em `−log(b/h)`. Uma seção quadrada produz zero, enquanto
seções horizontais e verticais de mesmo alongamento têm módulos iguais e sinais
opostos.

| Feature | Fórmula |
|---------|---------|
| `columns_mean_log_aspect_ratio` | `mean(log_aspect_i)`; balanço direcional com sinal |
| `columns_std_log_aspect_ratio` | `std(log_aspect_i)`; dispersão das orientações |
| `columns_max_abs_log_aspect_ratio` | `max(|log_aspect_i|)`; maior alongamento sem favorecer um eixo |

**Justificativa:** A transformação logarítmica preserva a direcionalidade da
rigidez sem a assimetria numérica da razão bruta `b/h`: valores `q` e `1/q`
passam a ter a mesma magnitude. Os três resumos anteriores da razão bruta foram
substituídos no schema v10.

### 6.11 Resumo do Vetor de Features

```
Índices  [0-3]    Área de pilares (4)
         [4-9]    Vãos livres direcionais de vigas (6)
         [10-12]  Momentos de inércia (3)
         [13-16]  Dispersão espacial de área e rigidez (4)
         [17-18]  Fatores de forma das seções (2)
         [19]     Balanço direcional dos raios de giração (1)
         [20-22]  Descritores logarítmicos de orientação (3)
         TOTAL: 23 features (schema v11)
```

---

## 7. Pipeline de Normalização

### 7.1 Estratégia de Normalização

Toda normalização usa `StandardScaler` do scikit-learn (média zero, desvio padrão unitário):

```
x̃ = (x − μ) / σ
```

Dois escaladores independentes são mantidos:
- **`scaler_X`:** ajustado sobre as features de treinamento, aplicado a treino, validação e teste
- **`scaler_y`:** ajustado sobre os alvos de treinamento, aplicado a treino, validação e teste; usado para reverter predições à escala original (kgf)

### 7.2 Protocolo de Divisão para Evitar Vazamento de Dados

O ajuste (*fit*) dos escaladores ocorre **exclusivamente** sobre o conjunto de treino, **nunca** sobre validação ou teste. As 230 amostras do piloto já foram usadas na análise exploratória de features e de modelos; por isso, elas são obrigatoriamente mantidas no conjunto de desenvolvimento e não são elegíveis para o teste final.

O protocolo rigoroso é:

```python
# 1. Índices 0:230 são protegidos como desenvolvimento já utilizado
indices_piloto = indices[:230]
indices_novos  = indices[230:]

# 2. Teste final: 15% do total, sorteado somente entre amostras novas
indices_desenvolvimento_novos, indices_teste = split_estratificado_por_faixas(
    indices_novos, n_test=375, random_state=42
)
indices_desenvolvimento = indices_piloto + indices_desenvolvimento_novos

# 3. Treino/validação também são estratificados por faixas do alvo
indices_treino, indices_validacao = split_estratificado_por_faixas(
    indices_desenvolvimento, validation_ratio=0.20, random_state=42
)

# 4. Ajuste apenas no treino
scaler_X.fit(X_train)
scaler_y.fit(y_train)

# 5. Transformação de todos os subconjuntos com o mesmo escalador
X_train_sc = scaler_X.transform(X_train)   # dados vistos no fit
X_val_sc   = scaler_X.transform(X_val)     # dados não vistos no fit
X_test_sc  = scaler_X.transform(X_test)    # dados não vistos no fit
```

A estratificação usa até 10 faixas balanceadas pelo posto (*rank*) do consumo de aço. Isso reduz a possibilidade de concentrar valores muito baixos ou muito altos em apenas um subconjunto. Os índices, estatísticas do alvo, proporções efetivas e hash do dataset são persistidos em `metrics/split_manifest.json`; os mesmos índices também são incluídos em `arrays.npz`.

As proporções resultantes com 2500 amostras:
- **Treino:** ~1700 amostras (68%)
- **Validação:** ~425 amostras (17%)
- **Teste:** ~375 amostras (15%)

### 7.3 Persistência da Pipeline

Os escaladores são serializados com `joblib` para o arquivo `feature_pipeline.pkl` dentro do diretório do experimento. Durante a inferência, os mesmos escaladores são carregados, garantindo transformação idêntica à usada durante o treinamento.

O modelo e a pipeline carregam o mesmo contrato semântico versionado: versão do formato do artefato, versão do schema, nomes e ordem das 23 features, nome e unidade do alvo, `INPUT_SIZE` e `OUTPUT_SIZE`. Um artefato legado, incompleto ou divergente é recusado; o código não renomeia, reordena nem trunca features automaticamente.

---

## 8. Arquitetura da Rede Neural

### 8.1 Tipo de Modelo

**Rede Neural Profunda Densa (DNN — Deep Neural Network)**, implementada em PyTorch. O modelo é um regressor que mapeia o vetor de 23 features do esquema v11 (normalizado) para o consumo de aço normalizado (escalar).

### 8.2 Arquitetura (classe `SimpleNN`)

```
Entrada: x ∈ ℝ²³ (normalizado)

Camada 1:
  Linear(23 → 128)
  BatchNorm1d(128)
  ReLU
  Dropout(p=0.2)

Camada 2:
  Linear(128 → 128)
  BatchNorm1d(128)
  ReLU
  Dropout(p=0.2)

Camada 3:
  Linear(128 → 64)
  BatchNorm1d(64)
  ReLU
  Dropout(p=0.2)

Saída:
  Linear(64 → 1)

Saída: ŷ ∈ ℝ¹ (consumo de aço normalizado)
```

**Total de parâmetros:** 28.545 parâmetros treináveis.

### 8.3 Componentes de Regularização

| Componente | Papel |
|---|---|
| **BatchNorm1d** | Normaliza ativações de cada mini-batch, estabiliza gradientes e acelera convergência |
| **Dropout(p=0.2)** | Zera aleatoriamente 20% dos neurônios durante o treino, prevenindo co-adaptação e overfitting |
| **Weight Decay (L2)** | `λ = 1×10⁻⁴` aplicado ao otimizador Adam; penaliza pesos grandes |

### 8.4 Função de Ativação

`ReLU(x) = max(0, x)` é usada em todas as camadas ocultas. Vantagens para este problema:
- Gradientes estáveis para redes de profundidade moderada
- Não sofre de saturação para valores positivos
- Computacionalmente eficiente

A camada de saída **não tem ativação** (regressão em escala livre).

---

## 9. Treinamento e Regularização

### 9.1 Função de Perda

A função de perda padrão é o **Erro Quadrático Médio (MSE)**, configurável para **Huber Loss** via `LOSS_TYPE`:

```
L_MSE(ŷ, y) = (1/N) × Σ (ŷᵢ − yᵢ)²

L_Huber(ŷ, y) = { ½(ŷ-y)²           se |ŷ-y| ≤ δ
                 { δ(|ŷ-y| − ½δ)    caso contrário
```

A Huber Loss é menos sensível a outliers que a MSE, sendo recomendada quando há amostras anômalas no conjunto de treinamento.

### 9.2 Otimizador

**Adam** (*Adaptive Moment Estimation*) com os seguintes parâmetros:
- Taxa de aprendizado: `η = 0,001`
- Decaimento de peso: `λ = 1×10⁻⁴`
- Parâmetros padrão: `β₁ = 0,9`, `β₂ = 0,999`, `ε = 10⁻⁸`

Adam combina as vantagens do AdaGrad (adaptação individual por parâmetro) e do RMSProp (média móvel de gradientes quadrados), sendo robusto a diferentes escalas de features.

### 9.3 Scheduler de Taxa de Aprendizado

**ReduceLROnPlateau:** reduz a taxa de aprendizado por um fator quando a loss de validação para de melhorar:

```
η_novo = η_atual × fator           se val_loss não melhora por patience épocas
η_novo = η_atual                   caso contrário
```

Parâmetros: `fator = 0,5`, `patience = 10 épocas`.

### 9.4 Early Stopping

O treinamento é interrompido antecipadamente se a loss de validação não melhorar por `patience = 50 épocas` consecutivas. O estado do modelo com **melhor loss de validação** é restaurado ao final do treinamento:

```
se val_loss < best_val_loss:
    best_model_state = deepcopy(model.state_dict())
    best_val_loss    = val_loss
    patience_counter = 0
senão:
    patience_counter += 1
    se patience_counter ≥ 50:
        carregar best_model_state
        encerrar treinamento
```

O número máximo de épocas é 500. Na prática, o early stopping tipicamente encerra antes.

### 9.5 Data Loader

- `batch_size = 32`
- Embaralhamento (`shuffle=True`) apenas no treino; sem embaralhamento em validação
- O gerador do embaralhamento usa a semente global do experimento
- Se o tamanho configurado produzir um último lote com apenas uma amostra, o
  tamanho efetivo do lote é reduzido para preservar todas as amostras e manter
  o `BatchNorm1d` válido
- `float32` para compatibilidade com GPU via CUDA

A perda de cada época é a média ponderada pelo número de amostras de cada
mini-batch. Portanto, um último lote menor não recebe o mesmo peso de um lote
completo. Antes da criação do modelo, treino e validação também são verificados
quanto a dimensionalidade, número de linhas, valores não finitos e aderência a
`INPUT_SIZE` e `OUTPUT_SIZE`; divergências interrompem o treinamento em vez de
alterar silenciosamente a arquitetura.

A época e a loss correspondentes ao melhor estado restaurado são armazenadas
no arquivo do modelo como `best_epoch` e `best_val_loss`.

### 9.6 Logging de Gradientes

A cada época, as normas L2 e médias absolutas dos gradientes de cada camada são registradas em `metrics/epochs.ndjson`. Isso permite diagnóstico de **vanishing/exploding gradients** durante o treinamento.

---

## 10. Avaliação do Modelo Substituto

### 10.1 Métricas no Conjunto de Teste

Após o treinamento, o modelo é avaliado exclusivamente sobre o conjunto de teste (não visto durante treino ou early stopping):

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **R²** | `1 − Σ(y−ŷ)² / Σ(y−ȳ)²` | Proporção de variância explicada (1 = perfeito) |
| **MAE** | `(1/N) Σ |y − ŷ|` | Erro médio absoluto em kgf |
| **RMSE** | `√[(1/N) Σ (y−ŷ)²]` | Erro quadrático médio em kgf |

### 10.2 Diagnósticos Visuais

O sistema gera automaticamente os seguintes gráficos diagnósticos no diretório `plots/` do experimento:

| Gráfico | Finalidade |
|---------|-----------|
| **Curvas de aprendizado** | Train/val loss por época + scheduler LR — detecta overfitting, subajuste ou instabilidade |
| **Scatter predito vs. real** | Inspeção visual da qualidade da regressão; desvio da diagonal indica viés |
| **Resíduos vs. predito** | Heterocedasticidade, bias sistemático por faixa de valor |
| **Q-Q plot** | Verifica normalidade dos resíduos (premissa para inferência estatística) |
| **Histograma de erros + ECDF** | Distribuição dos erros com ajuste de curva normal, % dentro de ±5% |
| **PFI (Permutation Feature Importance)** | Ranking de importância de features por permutação aleatória e medida de degradação do MSE |
| **SHAP** | Atribuição de valores de Shapley para interpretabilidade local e global |
| **PDP (Partial Dependence Plots)** | Efeito marginal das 3 features mais importantes sobre a predição |
| **PCA de cobertura** | Projeção 2D do espaço de features (treino vs. teste) colorida por aço kgf |
| **Distância de Mahalanobis** | % de pontos de teste fora da distribuição de treino (região de extrapolação) |
| **Cobertura KNN** | Distância média aos k=5 vizinhos mais próximos no treino — regiões mal cobertas |
| **Heatmap de correlação** | Matriz de correlação de Pearson entre features + target |
| **Normas de gradiente** | Evolução das normas L2 por camada ao longo do treinamento |

#### 10.2.1 Padrão editorial das figuras

Todas as figuras destinadas à tese usam o estilo centralizado em
`visualization/thesis_style.py`. Os textos internos são escritos em inglês
técnico e as figuras compostas recebem identificadores `(a)`, `(b)`, etc. A
paleta atribui papéis semânticos fixos: azul-marinho (`#1B3A5C`) para dados
principais, vermelho-tijolo (`#8C2F1B`) para referências e limiares, e azul
acinzentado (`#5C7A99`) apenas para uma terceira categoria real. Grandezas
sequenciais usam o mapa uniforme `thesis_navy`, construído de `#1B3A5C` a
`#A9BCCB`; a matriz de correlação constitui exceção por exigir uma escala
divergente centrada em zero.
Cada figura é exportada automaticamente em dois formatos com o mesmo nome-base:

- PNG a 300 dpi, para inspeção e inserção em editores que exigem imagem raster;
- PDF vetorial com fontes incorporáveis, preferencial para a versão final da tese.

Títulos são mantidos curtos, em 14 pt e negrito. Métricas e informações
auxiliares usam 11 pt e peso normal; rótulos de eixos usam 12 pt, ticks e
legendas usam 10 pt, e anotações usam 9–10 pt. Essa hierarquia preserva a
legibilidade após o redimensionamento para a largura da página. O grid é restrito
a linhas horizontais leves, exceto quando a própria representação exige outra
estrutura visual.

#### 10.2.2 Regeneração das figuras sem novo treinamento

Cada experimento preserva os dados numéricos usados nas figuras, além dos PNGs
e PDFs. A série de treinamento é gravada incrementalmente em
`metrics/epochs.ndjson`, enquanto `metrics/training_summary.json` registra a
melhor época, a menor perda de validação, o número de épocas executadas e o
tempo total. Os arrays bruto e normalizado dos três subconjuntos permanecem em
`arrays.npz`.

As avaliações finais são armazenadas em um contrato versionado:

- `metrics/figure_data.npz`: valores numéricos necessários para reconstrução;
- `metrics/figure_data_manifest.json`: versão, shapes, tipos, unidades e
  definições semânticas;
- `metrics/regression_test_predictions.csv`: valores TQS, predições, resíduos e
  erros do conjunto de teste;
- `metrics/classifier_test_predictions.csv`: classes reais e previstas,
  probabilidade de inviabilidade e limiar aplicado.

Depois de modificar o código de visualização, as figuras independentes do
modelo podem ser reconstruídas sem carregar ou treinar novamente a rede:

```powershell
python -m visualization.nn_diagnostics --exp "outputs/experiments/<experimento>"
```

Um diretório alternativo pode ser informado com `--out`. O comando reconstrói
curvas de treinamento, gráficos do regressor, matriz de confusão, ROC e
diagnósticos de cobertura diretamente dos artefatos salvos. PFI, PDP e SHAP são
análises dependentes do modelo; seus PNGs/PDFs originais permanecem no
experimento e, se precisarem ser recalculados, utilizam o modelo e a pipeline
salvos, sem repetir o treinamento.

### 10.3 Triagem Preliminar de Importância antes do Treinamento Final

Após a coleta intermediária de 230 amostras válidas no schema v11, foi executada
uma triagem independente da DNN final. Um `ExtraTreesRegressor` temporário foi
ajustado apenas dentro dos folds, com:

- validação cruzada `5 folds × 3 repetições`;
- importância por permutação medida pelo aumento do MAE fora da amostra;
- ablação pareada dos sete blocos de features;
- 50 reamostragens bootstrap sobre um conjunto retido;
- SHAP calculado de forma cruzada como confirmação secundária.

O modelo temporário apresentou `MAE = 20,23 ± 2,36 kgf` e `R² médio = 0,964`.
`columns_total_area_cm2` e `columns_mean_shape_factor` foram as únicas features
com importância individual forte e positiva em todos os critérios de
estabilidade. `columns_min_area_cm2` apresentou sinal secundário. As demais
features tiveram importância individual fraca ou inconclusiva.

A ablação por blocos mostrou, entretanto, que descritores geométricos
correlacionados se substituem quando o modelo é reajustado. O bloco de
distribuição espacial teve a maior contribuição conjunta entre os blocos
secundários, embora pequena. Por esse motivo, nenhuma feature foi removida no
schema v11: importância preditiva não equivale a causalidade estrutural, e a
orientação e a posição da rigidez continuam fisicamente relevantes. A decisão
deve ser repetida com a DNN e conjunto de teste finais. A importância do
classificador foi adiada porque a coleta intermediária contém somente 23 casos
inválidos.

### 10.4 Triagem Preliminar de Família e Hiperparâmetros

Antes da coleta final, o afinador legado — que ainda pressupunha duas saídas e
possuía erro de sintaxe — foi substituído por uma avaliação exclusiva do alvo
de aço. Foram comparados 10 candidatos:

- MLPs `[32,16]`, `[64,32]` e `[128,128,64]`;
- MSE e Huber Loss para cada MLP;
- quatro configurações de Extra Trees, com `min_samples_leaf ∈ {1,2}` e
  `max_features ∈ {0,8; 1,0}`.

O protocolo usou as 230 amostras piloto como desenvolvimento, com `5 folds × 3
repetições`, estratificação por faixas do consumo de aço e um split interno de
20% para early stopping das MLPs. O mesmo split interno foi retirado do ajuste
das árvores para manter igual o número de amostras efetivamente ajustadas. Não
foi salvo modelo de produção.

Resultados principais:

| Candidato | MAE médio | RMSE médio | R² médio | P90 de subestimação | MAE no quartil inferior |
|---|---:|---:|---:|---:|---:|
| Extra Trees, leaf=2, max_features=0,8 | 19,11 kgf | 24,96 kgf | 0,9682 | 28,36 kgf | 25,12 kgf |
| MLP `[64,32]`, dropout=0,1, Huber | 22,88 kgf | 28,08 kgf | 0,9594 | 35,11 kgf | 25,75 kgf |
| MLP atual `[128,128,64]`, dropout=0,2, MSE | 23,97 kgf | 29,38 kgf | 0,9551 | 36,59 kgf | 27,60 kgf |

Na comparação pareada dos 15 folds, a melhor Extra Trees reduziu o MAE da MLP
atual em média `4,86 kgf`, foi melhor em 80% dos folds e apresentou intervalo
bootstrap aproximado de 95% entre `2,83 e 6,95 kgf`. A MLP `[64,32] + Huber`
melhorou em média `1,09 kgf`, mas seu intervalo `[-0,40; 2,82] kgf` inclui zero;
portanto, ainda não há evidência suficiente para afirmar que ela supera a MLP
atual.

A conclusão é provisória: Extra Trees e a MLP `[64,32] + Huber` formam a lista
curta, mas a escolha de produção será repetida após as 2.500 amostras usando
somente o conjunto de desenvolvimento. O teste final permanecerá intocado. Os
resultados completos estão em `outputs/validation/teste17/model_tuning/`.

### 10.5 Repetibilidade dos Rótulos do TQS antes da Coleta Final

Antes de iniciar as 2.500 amostras, três geometrias conhecidas da coleta de 230
casos foram novamente processadas três vezes cada: a semente (índice 0), o caso
de menor consumo de aço (índice 101) e o caso de maior consumo (índice 218). Foi
usado um único worker no slot isolado `ValRep816B_01`, com ordem intercalada por
rodada (`semente → mínimo → máximo`) para também detectar eventual estado
residual deixado pela geometria anterior.

Antes das chamadas ao TQS, as features foram recalculadas a partir das geometrias
salvas e comparadas com os vetores do checkpoint. As nove execuções tiveram
validação estrutural independente pelas DLLs e seus relatórios foram arquivados.
O critério de aprovação admitia no máximo 1 kgf de variação do aço e 0,001 m³ do
concreto, mas os resultados foram exatamente repetíveis:

| Caso | Índice | Aço nas três execuções | Concreto nas três execuções | Diferença máxima para o checkpoint |
|---|---:|---:|---:|---:|
| Semente | 0 | 730,0 kgf | 16,02 m³ | 0,0 kgf |
| Mínimo de aço | 101 | 291,0 kgf | 12,01 m³ | 0,0 kgf |
| Máximo de aço | 218 | 1001,0 kgf | 18,36 m³ | 0,0 kgf |

Portanto, não foi observado ruído numérico, contaminação entre análises ou
divergência em relação aos rótulos da coleta intermediária. Nenhum treinamento
foi executado nesse teste. O checkpoint, o resumo e os nove relatórios estão em
`outputs/validation/teste18/tqs_repeatability/`.

### 10.6 Artefatos Persistidos por Experimento

Cada execução de treinamento cria um diretório autocontido em `outputs/experiments/<timestamp>/` com:

```
<timestamp>/
├── trained_model.pth          # Pesos da rede neural (PyTorch state_dict)
├── feature_pipeline.pkl       # Escaladores ajustados (joblib)
├── arrays.npz                 # Arrays brutos e normalizados (X_train, X_val, X_test, y_...)
├── config_snapshot.json       # Snapshot de todas as configurações no momento do treino
├── metadata.json              # Métricas finais e metadados do experimento
├── metrics/
│   ├── epochs.ndjson          # Métricas por época (loss, LR, gradientes)
│   ├── feature_names.json     # Nomes das 23 features
│   └── summary.json           # R², MAE, RMSE do teste
└── plots/
    ├── learning_curves.png
    ├── steel_scatter_residuals.png
    ├── error_histogram.png
    ├── coverage_pca.png
    ├── mahalanobis_outliers.png
    ├── knn_coverage.png
    ├── feature_correlation_heatmap.png
    ├── pfi_steel_regression.png
    ├── shap_summary_steel.png
    └── ...
```

O arquivo `arrays.npz` permite reexecutar qualquer análise de diagnóstico sobre um experimento passado **sem necessidade de retreinar o modelo**.

---

## 11. Classificador de Validade Estrutural

### 11.1 Motivação

Nem toda combinação de dimensões de pilares resulta em um projeto estruturalmente viável. O TQS pode reprovar um projeto por insuficiência de armadura, esbeltez excessiva, ou outros critérios normativos. Durante a otimização, o modelo substituto pode propor geometrias que, embora economicamente atrativas, seriam fisicamente inviáveis.

Para penalizar automaticamente estas configurações, um **classificador binário** é treinado separadamente, com os rótulos:
- `1` = amostra estruturalmente válida
- `0` = amostra estruturalmente inválida (reprovada pelo TQS)

### 11.2 Treinamento do Classificador

- **Modelo:** Regressão Logística (`sklearn.linear_model.LogisticRegression`, `class_weight='balanced'`, `max_iter=1000`), precedida por um `StandardScaler` no mesmo `Pipeline` (`sklearn.pipeline.make_pipeline`)
- **Features:** vetor de 23 features, normalizado pelo `StandardScaler` interno do pipeline antes de entrar no classificador
- **Split:** 60% treino, 20% validação e 20% teste, estratificado por classe. A validação é usada somente para calibrar o limiar; o teste permanece fora de qualquer decisão de ajuste.
- **Limiar:** escolhido na validação pelo índice de Youden sobre `P(inviável)` e depois congelado.
- **Métricas finais:** calculadas apenas no teste com `inviável` como classe positiva: Acurácia, Precisão, Recall, F1-score e ROC AUC.
- **Erro crítico de segurança:** quantidade e proporção de amostras realmente inviáveis classificadas como viáveis, correspondente à célula `[0, 1]` da matriz de confusão.

O arquivo `classifier_test.json` registra as métricas da regra final e
`roc_curve_test.json` contém a curva ROC do teste. `roc_curve.json` contém apenas
a curva da validação usada para selecionar o limiar e não deve ser apresentada
como desempenho final.

### 11.3 Uso na Otimização

Durante a otimização, o classificador retorna `prob_invalid = P(classe=0 | x)`. A função objetivo usa o mesmo limiar calibrado e persistido em `validity_threshold.json`. O valor `INVALID_PROB_THRESHOLD = 0,5` é apenas o fallback quando não existe um artefato calibrado:

```
se prob_invalid ≥ limiar_calibrado:
    custo += INVALID_COST_PENALTY (padrão: 1.000.000)
```

---

## 12. Otimização por Algoritmo Genético

### 12.1 Justificativa da Escolha

O Algoritmo Genético (AG) é adotado pela capacidade de:
- Explorar espaços de busca não convexos e multidimensionais
- Lidar com variáveis discretas (comprimentos em múltiplos de 5 cm)
- Não exigir gradientes da função objetivo (que inclui chamadas ao classificador e penalidades não diferenciáveis)
- Manter diversidade de soluções, evitando convergência prematura a ótimos locais

### 12.2 Representação da Solução

Cada indivíduo da população é um vetor contínuo `x ∈ ℝᵈ` onde `d` é o número de grupos de variáveis de projeto. Os valores representam os comprimentos das seções transversais dos pilares (em cm). Durante a avaliação da função objetivo, o vetor é discretizado para múltiplos de 5 cm antes de reconstruir a geometria.

### 12.3 Inicialização da População

```
pop[0] = x₀ = lower_bounds    (configuração semente — garantidamente viável)
pop[1..N-1] ~ Uniform(lower_bounds, upper_bounds)
```

O indivíduo inicial (configuração semente) é sempre incluído, garantindo que a população começa com pelo menos uma solução estruturalmente conhecida.

### 12.4 Operadores Genéticos

#### Seleção por Torneio (k=3)
```
candidatos = amostrar_aleatório(população, k=3)
vencedor = argmin(custo(candidatos))
```

#### Crossover Aritmético (*Blend Crossover*)
```
se random() < taxa_crossover (0,9):
    α ~ Uniform(0, 1)   (vetor de pesos, um por dimensão)
    filho_1 = α × pai_1 + (1 − α) × pai_2
    filho_2 = α × pai_2 + (1 − α) × pai_1
    clipar(filho, lower, upper)
```

#### Mutação Gaussiana
```
se random() < taxa_mutação (0,2):
    ruído ~ Normal(0, 0,2 × (upper − lower))
    mutante = individuo + ruído
    clipar(mutante, lower, upper)
```

#### Elitismo
O melhor indivíduo de cada geração é automaticamente copiado para a próxima, garantindo monotocidade do melhor custo ao longo das gerações.

### 12.5 Parâmetros do AG

| Parâmetro | Valor | Significado |
|-----------|-------|-------------|
| `pop_size` | 40 | Tamanho da população |
| `generations` | 80 | Número máximo de gerações |
| `crossover_rate` | 0,9 | Probabilidade de crossover |
| `mutation_rate` | 0,2 | Probabilidade de mutação |
| `tournament_k` | 3 | Tamanho do torneio |
| `patience` | 20 | Gerações sem melhora para parada por estagnação |
| `max_time_sec` | 600 | Tempo máximo de otimização (s) |

### 12.6 Critérios de Parada

O AG encerra quando qualquer um dos seguintes critérios for satisfeito:
1. Atingidas `generations = 80` gerações
2. `patience = 20` gerações consecutivas sem melhora no melhor custo
3. Tempo total excede `max_time_sec = 600 s`
4. `prob_invalid ≤ 0,1` **e** `steel ≤ min_steel_kg` (metas mínimas atingidas)

---

## 13. Função Objetivo

### 13.1 Formulação

A função objetivo combina custo de materiais com penalidades de validade estrutural:

```
f(x) = (aço × P_aço) + (concreto × P_concreto) + penalidades

onde:
  aço = consumo de aço predito pelo modelo substituto (kgf)
  concreto = volume de concreto calculado geometricamente (m³)
  P_aço = R$ 100,00/kg    (STEEL_PRICE_KG)
  P_concreto = R$ 10,00/m³  (CONCRETE_PRICE_M3)
```

### 13.2 Penalidades

```
penalidade_validade = INVALID_COST_PENALTY (= 1.000.000) se prob_invalid ≥ 0,5
penalidade_negativo = 1.000.000  se aço < 0 ou concreto < 0
```

### 13.3 Fluxo de Avaliação

```
1. Discretizar vetor x → x_disc = round(x/20)×20
2. Reconstruir geometria: create_geometry_from_vector(x_disc)
3. Extrair features: FeatureEngineer(col_polygons, beam_defs).extract_features()
4. Normalizar: scaler_X.transform(features)
5. Predizer aço: model(features_norm) → scaler_y.inverse_transform() → aço kgf
6. Predizer prob_invalid: classifier.predict_proba(features_raw) → P(inválido)
7. Calcular concreto: get_geometric_concrete_volume(col_polygons) + vol_vigas
8. Calcular custo total: f = aço×100 + concreto×10 + penalidades
```

**Tempo de avaliação:** ~0,3–1 ms (vs. 30–180 s do TQS) — aceleração de 10.000–600.000×.

---

## 14. Inferência e Predição em Produção

### 14.1 Componentes da Inferência

A classe `BuildingInference` encapsula toda a lógica de predição pós-treinamento:

1. **Carregamento de artefatos:** modelo `.pth` + pipeline `.pkl` do diretório do experimento
2. **Predição de aço:** `predict_from_segments(segments)` ou `predict_from_csv(csv_path)`
3. **Predição de concreto:** cálculo geométrico via `get_geometric_concrete_volume`
4. **Predição de validade:** `prob_invalid = classifier.predict_proba(features_raw)[0, idx_invalid]`

### 14.2 Seleção Explícita e Contrato dos Artefatos

O experimento deve ser indicado explicitamente pelo argumento de `BuildingInference` ou pela variável de ambiente `BUILDOPT_EXPERIMENT_ID`. Se o identificador não existir, a inicialização falha; não há fallback para o diretório mais recente.

Antes de qualquer previsão são verificados:

- igualdade entre os contratos do `trained_model.pth` e do `feature_pipeline.pkl`;
- `FEATURE_SCHEMA_VERSION = 11`;
- nomes e ordem exatos das features;
- alvo `column_steel_weight`, em `kgf`;
- dimensões do modelo, scalers e vetor recalculado pelo `FeatureEngineer`;
- presença de valores finitos no vetor de inferência.

Qualquer divergência cancela a inferência com erro explícito.

---

## 15. Gerenciamento de Experimentos e Reprodutibilidade

### 15.1 Identificação de Experimento

Cada execução de treinamento cria um diretório com timestamp no formato:
```
<YYYYMMDD-HHMMSS>_Treino_com_<N>_amostras/
```

### 15.2 Snapshot de Configurações

No início de cada experimento, as configurações (`BuildingConfig`, `RunConfig`, `DataSplitConfig`, `NeuralNetConfig`, `ParallelConfig`, `ObjectiveConfig`, `VectorConfig`) e o contrato de features/alvo são serializados em `config_snapshot.json`, garantindo reprodutibilidade do contexto de treinamento.

### 15.3 Semente Aleatória

A semente `SEED = 42` é usada para:
- Divisão treino/val/teste (`random_state=42`)
- Inicialização do AG (`random_state=42`)
- Amostragens SHAP e Mahalanobis

### 15.4 Hash do Dataset

Um hash SHA-256 do array de dados de treinamento é computado e registrado nos metadados, permitindo verificar se dois experimentos usaram exatamente o mesmo conjunto de dados.

---

## 16. Fluxo Completo de Execução

```
main.py
│
├── 1. Inicialização
│   ├── LengthProcessor — lê segmentos do CSV semente (BuildingInput.csv)
│   ├── FeaturePipeline — inicializa escaladores (não ajustados ainda)
│   ├── NeuralNetworkManager — instancia modelo (não treinado)
│   ├── TQSModelManager — prepara interface com TQS
│   └── ExperimentManager — cria diretório do experimento, salva config_snapshot.json
│
├── 2. Coleta de Dados de Treinamento
│   ├── Análise da configuração semente (sempre primeiro)
│   └── Loop (até 2500 amostras válidas / 12.500 tentativas):
│       ├── generate_variation() → novos segmentos
│       ├── LengthProcessor.process_segments() → polígonos + vigas
│       ├── TQSModelManager.create_building_model_and_elements()
│       ├── RunModel(building_name) → TQS analisa estrutura
│       ├── extract_material_summary(RESDES.HTM) → aço kgf, concreto m³
│       ├── FeatureEngineer.extract_features() → vetor de 23 features
│       └── Armazenar (features, aço, label_validade)
│
├── 3. Treinamento do Modelo Substituto
│   ├── train_test_split + scaler.fit(X_train, y_train)
│   ├── np.savez_compressed → arrays.npz
│   ├── NNManager.train(X_train_sc, y_train_sc, X_val_sc, y_val_sc)
│   │   ├── Loop de épocas (max 500):
│   │   │   ├── _run_train_epoch → forward, loss, backward, Adam step
│   │   │   ├── _run_eval_epoch → val loss
│   │   │   ├── ReduceLROnPlateau.step(val_loss)
│   │   │   ├── Early stopping check
│   │   │   └── Gravar epochs.ndjson
│   │   └── Carregar best_model_state
│   ├── NNManager.save_model → trained_model.pth
│   └── FeaturePipeline.save → feature_pipeline.pkl
│
├── 4. Treinamento do Classificador de Validade
│   └── make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced')).fit(X_clf, y_clf)
│
├── 5. Avaliação e Diagnósticos
│   ├── Predições no conjunto de teste
│   ├── Cálculo de R², MAE, RMSE
│   ├── Salvar summary.json, metadata.json
│   └── run_full_diagnostics() → ~15 gráficos em plots/
│
└── 6. Otimização
    ├── DesignSpace(BuildingInput.csv) → espaço de busca
    ├── BuildingInference() → carrega modelo treinado
    ├── ObjectiveFunction(design_space, inference)
    ├── GeneticOptimizer.run() → 80 gerações × 40 indivíduos
    └── Solução ótima → validar com TQS completo
```

---

## 17. Parâmetros e Hiperparâmetros Consolidados

### 17.1 Arquitetura da Rede Neural

| Parâmetro | Valor |
|---|---|
| `INPUT_SIZE` | 23 |
| `HIDDEN_LAYERS` | [128, 128, 64] |
| `DROPOUT_RATE` | 0,2 |
| `OUTPUT_SIZE` | 1 |
| `LEARNING_RATE` | 0,001 |
| `NUM_EPOCHS` | 500 |
| `BATCH_SIZE` | 32 |
| `EARLY_STOPPING_PATIENCE` | 50 |
| `TEST_SPLIT_RATIO` | 0,15 |
| `VALIDATION_SPLIT_RATIO` | 0,20 |
| `PREUSED_DEVELOPMENT_PREFIX_SAMPLES` | 230 (nunca entram no teste final) |
| `PREUSED_CLASSIFIER_PREFIX_SAMPLES` | 253 (casos válidos e inválidos do piloto; nunca entram no teste final do classificador) |
| `REGRESSION_STRATIFICATION_BINS` | 10 |
| `LOSS_TYPE` | MSE (configurável para Huber) |
| `WEIGHT_DECAY` | 1×10⁻⁴ |
| `LR_SCHEDULER_PATIENCE` | 10 épocas |
| `LR_SCHEDULER_FACTOR` | 0,5 |

### 17.2 Coleta de Dados

| Parâmetro | Valor |
|---|---|
| `NUM_SAMPLES` | 2500 amostras válidas |
| `MAX_ITERATION_FACTOR` | 5 (máx 12.500 tentativas) |
| `TQS_TIMEOUT_SEC` (modo sequencial, `RunConfig`/`BuildingConfig`) | 120 s |
| `ParallelConfig.TIMEOUT_SEC` (modo paralelo, ativo com 1 worker) | 180 s |
| `CHECKPOINT_INTERVAL_MIN` | 10 min |
| `VALIDITY_CHECK_DLL` | `True` (obrigatório, *fail-closed*) |
| `SEED` | 42 |

### 17.3 Algoritmo Genético

| Parâmetro | Valor |
|---|---|
| `pop_size` | 40 |
| `generations` | 80 |
| `crossover_rate` | 0,9 |
| `mutation_rate` | 0,2 |
| `tournament_k` | 3 |
| `patience` | 20 gerações |
| `max_time_sec` | 600 s |

### 17.4 Função Objetivo

| Parâmetro | Valor |
|---|---|
| `STEEL_PRICE_KG` | R$ 100,00/kg |
| `CONCRETE_PRICE_M3` | R$ 10,00/m³ |
| `LENGTH_STEP_CM` | 5 cm |
| `INVALID_PROB_THRESHOLD` | 0,5 |
| `INVALID_COST_PENALTY` | 1.000.000 |

### 17.5 Dimensões Fixas da Edificação

| Parâmetro | Valor |
|---|---|
| Largura de vigas | 20 cm |
| Altura de vigas | 40 cm |
| Espessura de lajes | 12 cm |
| Vão máximo de vigas (splitThreshold) | 70 cm |
| Carga permanente (vigas) | 2,0 tf/m |
| Carga acidental (vigas) | 1,0 tf/m |
| Carga permanente (lajes) | 1,0 tf/m² |
| Carga acidental (lajes) | 1,0 tf/m² |

---

## Referências Técnicas de Implementação

| Biblioteca | Versão | Uso |
|---|---|---|
| PyTorch | ≥2.0 | Rede neural, autograd, GPU |
| scikit-learn | ≥1.3 | StandardScaler, LogisticRegression, métricas, PFI, PCA, KNN |
| NumPy | ≥1.24 | Computação numérica, álgebra linear |
| Shapely | ≥2.0 | Polígonos de pilares, operações booleanas |
| pandas | ≥2.0 | Leitura de CSV, manipulação de dados tabulares |
| joblib | — | Serialização de escaladores |
| SHAP | ≥0.51 | Análise de interpretabilidade SHAP |
| matplotlib / seaborn | — | Visualizações diagnósticas |
| TQSExec (DLL) | v26A | API de análise estrutural |

---

*Documento gerado automaticamente a partir do código-fonte do projeto `BuildingOptimization`.*
*Para análises pós-treinamento sem re-execução, usar:*
```bash
python visualization/nn_diagnostics.py --exp outputs/experiments/<id_experimento>
```

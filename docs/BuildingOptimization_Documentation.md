# BuildingOptimization: Technical Documentation

## 1. Code Overview

- Architecture: Orchestrator `main.py` drives data collection, model training, and evaluation via `BuildingOptimizer` (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\main.py:75). Geometry preprocessing lives under `geometry/`, TQS integration under `tqs_interface/`, ML under `models/` and `utils/`, and optimization under `optimization/`.
- Paradigms: Modular design with separation of concerns; data orchestration, geometry processing, ML pipelines, and external system integration are isolated. Use of data classes where appropriate and factory methods for TQS elements (`tqs_interface/tqs_model.py`).
- Technical specs: Python 3.x on Windows; relies on TQS DLLs and HTML outputs. Neural network implemented in PyTorch; preprocessing and classification in scikit-learn; geometry in Shapely. Outputs organized under `config/paths.py`.

## 2. Implementation Details

- Core algorithms annotated by source:
  - Building workflow
    - `BuildingOptimizer.run_optimization` executes the end-to-end pipeline (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\main.py:142).
    - Data collection loop `_collect_training_data` generates variations, analyzes, and stores labels and features (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\main.py:241–360).
    - TQS execution `_execute_tqs_analysis_and_get_results` runs TQS, waits for `RESDES.HTM`, parses totals, and checks critical DLL errors (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\main.py:628–682).
  - Feature pipeline
    - Fit/transform with `StandardScaler` (`FeaturePipeline.fit_transform`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\utils\feature_pipeline.py:23–49).
    - Inverse transform outputs (`FeaturePipeline.inverse_transform_outputs`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\utils\feature_pipeline.py:71–89).
    - Persist scalers via `joblib` (`FeaturePipeline.save`/`load`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\utils\feature_pipeline.py:121–137).
  - Surrogate model manager
    - Training with early stopping and optional LR scheduler (`NeuralNetworkManager.train`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\models\nn_manager.py:42–108).
    - Model components initialization with `MSELoss` or `HuberLoss` and Adam weight decay (`_initialize_model_components`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\models\nn_manager.py:213–228).
    - Epoch loops for train/eval (`_run_train_epoch`, `_run_eval_epoch`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\models\nn_manager.py:230–254).
  - Validity classifier
    - Training logistic regression with class balancing (`BuildingOptimizer._train_and_evaluate`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\main.py:382–409).
    - Inference for invalid probability (`BuildingInference._predict_validity_probability`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\inference.py:92–117).
  - Objective function and GA
    - Cost computation combining steel, concrete, and invalidity penalty (`ObjectiveFunction.calculate_cost`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\optimization\objective_function.py:67–138).
    - Genetic algorithm loop with tournament selection, blend crossover, Gaussian mutation, and elitism (`GeneticOptimizer.run`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\optimization\optimizer.py:96–164).
  - Geometry and volumes
    - Column and beam geometric volumes (`utils/geometric_calculator.py`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\utils\geometric_calculator.py:27–66, 67–99, 146–171).
    - Beam location inference along walls (`LengthProcessor._find_beam_locations`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\geometry\length_input_processor.py:133–196).
  - TQS integration
    - Global processing execution via DLL with cleanup (`RunModel`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\tqs_interface\tqs_exec.py:46–82).
    - Critical error reading from NGERERRO/NMSGERRO (`TQSErrorReader.get_critical_errors`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\tqs_interface\tqs_errors.py:38–149).

- Mathematical formulations:
  - Standardization: X' = (X − μ_X) / σ_X; y' = (y − μ_y) / σ_y.
  - Losses: MSE L = \(\frac{1}{n}\sum_i \lVert \hat{y}_i - y_i \rVert^2\); Huber L_δ as in Huber (1964): quadratic near zero, linear tails.
  - Adam (Kingma & Ba, 2015): \(m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t\), \(v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2\), parameter update \(\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}\) with weight decay.
  - Logistic regression classifier: binary probability \(P(y=1\mid x)=\sigma(w^\top x+b)\), trained with class_weight='balanced'. Invalid probability reported as \(P(y=0\mid x)\).
  - Objective function: \(C(x) = p_s \cdot S(x) + p_c \cdot C_{conc}(x) + \mathbb{1}[P_{inv}(x) \ge \tau] \cdot \lambda\), where steel surrogate S comes from NN inference and concrete volume \(C_{conc}\) from geometry; \(\tau\) is threshold and \(\lambda\) penalty.
  - GA operators: blend crossover \(c_1 = \alpha p_1 + (1-\alpha) p_2\), mutation \(x' = \mathrm{clip}(x + \epsilon, [l,u])\), tournament selection.

- Computational complexity:
  - Surrogate training: O(E · N · B · L) where E epochs, N samples, B batch size, L layer cost; early stopping reduces E.
  - GA run: O(G · P · T_cost) where G generations, P population, T_cost is cost evaluation time (in geometric mode small; in TQS mode dominated by external processing latency).
  - Feature pipeline: O(N · d) for scaling d-dimensional features.

- Memory management strategies:
  - Persist scalers (`joblib`) and model checkpoints (`torch.save`) without embedding large datasets; artifacts tracked via `ExperimentManager` (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\utils\experiment_manager.py:12–85).
  - Avoid holding entire raw datasets after scaling; training uses `DataLoader` batches.
  - Use environment paths to avoid hardcoding and reduce duplicated artifacts.

## 3. Experimental Methodology

- Dataset and preprocessing:
  - Geometry vectors from CSV files configured in `config/paths.py` (SEED/TEST) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\config\paths.py:21–27).
  - Features engineered from columns and beams including areas, lengths, inertia, and approximate volumes (`FeatureEngineer.extract_features`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\utils\feature_engineer.py:20–134).
  - Standardization applied consistently via `FeaturePipeline`.

- Training protocol:
  - Surrogate network: `SimpleNN` architecture with hidden layers `[128,128,64]`, dropout `0.2`, optimizer Adam (`NeuralNetConfig`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\config\settings.py:22–46).
  - Loss: MSE or Huber; learning rate `1e-3`, early stopping with patience `50`, optional `ReduceLROnPlateau` scheduler.
  - Validity classifier: Logistic regression with `class_weight='balanced'` trained on collected features/labels (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\main.py:382–409).

- Evaluation metrics and statistical tests:
  - Primary metrics: R² and MAE for steel and concrete when available (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\main.py:432–451).
  - Suggested statistical validation: bootstrap confidence intervals for MAE/R²; paired t-tests for surrogate vs. TQS values.

- Baselines:
  - Linear regression on engineered features for steel prediction.
  - Constant predictor using training-set mean as a naive baseline.

## 4. Results Analysis

- Quantitative metrics:
  - Store final metrics in `ExperimentManager` metadata (samples trained, test samples, metric dict) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\main.py:456–472).
  - For classifiers, save `accuracy` and optionally `ROC AUC` to JSON (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\main.py:392–404).
  - Confidence intervals recommended via bootstrap over test set.

- Qualitative behavior:
  - Inspect predicted vs actual plots, analyze over/underestimation patterns across geometry regimes (long beams, dense columns).
  - Evaluate sensitivity to length discretization (`ObjectiveFunction._discretize_vector`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\optimization\objective_function.py:47–66).

- Error analysis and failure cases:
  - TQS timeouts waiting for `RESDES.HTM` (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\main.py:651–657).
  - Missing beam definitions or empty column polygons resulting in degraded geometric volume estimation (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\main.py:536–544).
  - Classifier absence yields `None` probability; objective penalty disabled in such cases (`inference.py`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\inference.py:92–117, 286–288).

- Computational resource utilization:
  - GPU acceleration when available (`torch.cuda.is_available`) (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\models\nn_manager.py:38–41).
  - GA runtime scales with generations and population; TQS mode significantly increases `T_cost` due to external processing.

## 5. Academic Context

- Literature review:
  - Surrogate modeling for structural optimization: see Bishop (2006), Hastie et al. (2009), Goodfellow et al. (2016) for ML fundamentals.
  - Optimization via genetic algorithms: Goldberg (1989) and modern GA techniques.
  - Optimization with penalty methods for constraint violations: standard constrained optimization texts.
  - Adam optimizer: Kingma & Ba (2015); Huber loss: Huber (1964).

- Theoretical implications:
  - The surrogate model approximates a complex mapping from geometry-derived features to steel quantities, enabling rapid evaluation within GA loops.
  - Classifier-driven penalization introduces a soft constraint that biases the search away from invalid structures while preserving exploration.

- Limitations and future work:
  - Geometric concrete volume approximates beam-column overlap; improved estimators could subtract intersections (see `calculate_beams_geometric_volume_with_subtractions`).
  - Validity classifier trained on collected labels may be biased; consider cross-validation and additional features.
  - Explore Bayesian optimization or CMA-ES for continuous design vectors; incorporate multi-objective optimization (steel vs. concrete vs. deflection).

## Reproducibility Instructions

- Environment:
  - OS: Windows; Python 3.10+ recommended.
  - TQS installed with accessible DLLs (`NGERERRO.dll`, `NMSGERRO.dll`). Set `BUILDOPT_TQS_DLL_DIR` if not `config.paths.TQS_OUTPUT_DIR`.

- Dependencies:
  - `torch`, `numpy`, `scikit-learn`, `joblib`, `pandas`, `shapely`, `bs4`, `lxml`, `matplotlib` (for plotting), `TQS` native Python bindings.

- Steps:
  - Configure `config/paths.py` and `config/settings.py` to match your environment.
  - Run training via `main.py` after preparing the seed CSVs under `data/`.
  - Artifacts are saved under `outputs/experiments/<timestamp>_.../` by `ExperimentManager`.
  - Inference uses `inference.py` with `EXPERIMENT_ID` or environment variable override.

## Ethical Considerations

- Structural safety: Surrogate predictions and geometric approximations must not be used as a safety-critical replacement for certified structural analysis. Use TQS results for final validation.
- Dataset bias: Ensure training data represents diverse building configurations to avoid biased recommendations.

## Complete Dependency Specifications

- Python: 3.10+
- Packages: torch>=2.0, numpy>=1.24, scikit-learn>=1.3, joblib>=1.3, pandas>=2.0, shapely>=2.0, beautifulsoup4>=4.12, lxml>=4.9, matplotlib>=3.7
- External: TQS installed with DLL access; Windows environment with permissions to read/write under `C:\TQS`.

## Appendix: Line-by-Line Annotations (Core Snippets)

- `NeuralNetworkManager.train` (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\models\nn_manager.py:42–108)
  - 53–54: Validates input/output sizes and aligns `INPUT_SIZE`/`OUTPUT_SIZE` to data.
  - 57–61: Splits normalized data into train/val/test; builds loaders.
  - 63–64: Initializes model, loss, optimizer per `NeuralNetConfig`.
  - 70–77: Optionally configures `ReduceLROnPlateau` scheduler.
  - 79–86: Per-epoch train/eval; scheduler step on validation loss.
  - 87–98: Early stopping based on best validation loss with patience.
  - 100–108: Restores best weights and returns normalized test sets.

- `ObjectiveFunction.calculate_cost` (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\optimization\objective_function.py:67–138)
  - 96–98: Discretizes continuous vector by `LENGTH_STEP_CM` and clips to bounds.
  - 100–107: Converts discretized geometry to CSV buffer.
  - 113–121: Predicts steel, computes concrete, and applies invalidity penalty.
  - 122–126: Guards against negative values with large penalty; returns total cost.

- `GeneticOptimizer.run` (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\optimization\optimizer.py:96–164)
  - 112–118: Initializes population and evaluates costs; sets initial best.
  - 122–138: Generates new population via elitism, tournament selection, blend crossover, and mutation.
  - 139–151: Evaluates new population and updates global best with stagnation tracking.
  - 156–164: Early exit by patience; returns `OptimizeResult`.

- `TQSErrorReader.get_critical_errors` (d:\Trabalho\01_Desenvolvimento\TESTES IA\Python program\BuildingOptimization\tqs_interface\tqs_errors.py:38–149)
  - 47–51: Enumerates `VIGAS`, `PILAR`, `ESPACIAL` folders under the building.
  - 73–91: Validates DLL function availability and sets ctypes signatures.
  - 96–132: Iterates programs/errors; filters `CLASSIFICATION==2`; collects unique `(element, message)` tuples.

## Citations

- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Hastie, T., Tibshirani, R., Friedman, J. (2009). The Elements of Statistical Learning. Springer.
- Goodfellow, I., Bengio, Y., Courville, A. (2016). Deep Learning. MIT Press.
- Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley.
- Kingma, D. P., Ba, J. (2015). Adam: A Method for Stochastic Optimization.
- Huber, P. J. (1964). Robust Estimation of a Location Parameter.

## Procedimentos de Emergência

- Intervenção manual:
  - Verifique `outputs/experiments/<id>/metrics/system_health.ndjson` para CPU/memória/disk e rede.
  - Consulte `alerts.ndjson` para alertas de espaço em disco e travamento.
  - Se `RESDES.HTM` não for gerado, rode novamente o TQS e confirme permissões em `C:\TQS`.
- Reinício seguro:
  - Habilite `RunConfig.RESUME_FROM_CHECKPOINT=True` e reinicie `main.py`; o sistema retoma de `checkpoint.json`.
  - Se necessário, apague `checkpoint.json` para iniciar do zero.
- Suporte técnico:
  - SMTP configurável via variáveis `BUILDOPT_ALERT_SMTP_*`; e-mail de alerta em `BUILDOPT_ALERT_EMAIL_TO`.
  - Logs detalhados de erros em `metrics/tqs_errors.ndjson` e `errors.ndjson` quando aplicável.
- Recuperação de desastres:
  - Copie `outputs/experiments/<id>/` para um local seguro.
  - Restaure `feature_pipeline.pkl`, `trained_model.pth`, `summary.json`, `epochs.ndjson` e `checkpoint.json` e reexecute `inference.py` para validação.
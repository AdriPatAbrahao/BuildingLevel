# BuildingOptimization/inference.py

import traceback
import os
from pathlib import Path
import json
import time

import numpy as np

# --- Importe as MESMAS classes e configurações do seu main.py ---
from config.settings import BuildingConfig
from config import paths
from geometry.length_input_processor import LengthProcessor # Importa a classe atualizada
from geometry.binary_input_processor import BinaryProcessor
from utils.feature_engineer import FeatureEngineer
from models.nn_manager import NeuralNetworkManager
from tqs_interface.tqs_manager import TQSModelManager
from tqs_interface.tqs_exec import RunModel
from results.resultsext import extract_material_summary
from utils.feature_pipeline import FeaturePipeline
from utils.geometric_calculator import get_geometric_concrete_volume, calculate_column_formwork_area
from joblib import load
            
# --- CONFIGURAÇÃO DA INFERÊNCIA ---
# Você só precisa definir o ID do experimento que quer usar.
# Ex: "20250830-180000_Treino_com_200_amostras_e_BN"
##EXPERIMENT_ID = "20250901-234401_Treino_com_200_amostras_e_BN"
# Default experiment ID used when no environment override is provided
_DEFAULT_EXPERIMENT_ID = "20260407-190546_Treino_com_1500_amostras"
# Allow overriding via environment variable for operational flexibility
EXPERIMENT_ID = os.getenv("BUILDOPT_EXPERIMENT_ID", _DEFAULT_EXPERIMENT_ID)


EXPERIMENT_DIR = Path("outputs/experiments") / EXPERIMENT_ID
PIPELINE_PATH = EXPERIMENT_DIR / "feature_pipeline.pkl"
MODEL_PATH = EXPERIMENT_DIR / "trained_model.pth"
CONFIG_SNAPSHOT_PATH = EXPERIMENT_DIR / "config_snapshot.json"

class BuildingInference:
    """
    Orchestrates inference using saved scalers and model artifacts.

    Loads experiment artifacts (feature pipeline, surrogate model, optional
    validity classifier), validates compatibility against the snapshot, and
    provides utilities for CSV-based prediction and TQS comparison.
    """
    def __init__(self, experiment_id: str | None = None):
        self.validity_classifier = None
        self._validity_classifier_classes = None
        self.invalid_threshold = None
        try:
            print("--- Inicializando o Orquestrador de Inferência ---")
            # Resolve experiment directory — instance-level, never relies on module globals
            # so that multiple BuildingInference instances in the same session don't clash.
            eid = experiment_id or os.getenv("BUILDOPT_EXPERIMENT_ID") or _DEFAULT_EXPERIMENT_ID
            exp_dir = paths.EXPERIMENTS_DIR / eid
            if not exp_dir.exists():
                raise FileNotFoundError(
                    f"Diretório do experimento não encontrado: '{exp_dir}'. "
                    "Informe explicitamente um experimento treinado compatível."
                )

            # Store paths as instance attributes — all methods use self.exp_dir etc.
            self.exp_dir = exp_dir
            self.experiment_id = exp_dir.name
            self._pipeline_path = exp_dir / "feature_pipeline.pkl"
            self._model_path = exp_dir / "trained_model.pth"
            self._config_snapshot_path = exp_dir / "config_snapshot.json"

            # Carrega a pipeline e o modelo
            print(f"Carregando artefatos do experimento: '{self.experiment_id}'")
            self.feature_pipeline = FeaturePipeline()
            self.feature_pipeline.load(self._pipeline_path)

            self.nn_manager = NeuralNetworkManager()

            self.input_processor = self.feature_pipeline.input_processor
            if not self.nn_manager.load_model(self._model_path):
                raise RuntimeError("Falha ao carregar o modelo treinado.")
            if self.feature_pipeline.artifact_contract != self.nn_manager.artifact_contract:
                raise RuntimeError(
                    "Os contratos semânticos do modelo e da pipeline são diferentes."
                )

            classifier_path = self.exp_dir / "validity_classifier.pkl"
            if classifier_path.exists():
                try:
                    self.validity_classifier = load(classifier_path)
                    self._validity_classifier_classes = self._extract_classifier_classes(self.validity_classifier)
                    print(f"Validity classifier loaded from '{classifier_path}'.")
                except Exception as clf_err:
                    print(f"Warning: failed to load validity classifier: {clf_err}")
                    self.validity_classifier = None
                    self._validity_classifier_classes = None
            else:
                print(f"Info: Validity classifier not found at '{classifier_path}'.")

            thr_path = self.exp_dir / "metrics" / "validity_threshold.json"
            if thr_path.exists():
                try:
                    with open(thr_path, 'r', encoding='utf-8') as f:
                        obj = json.load(f)
                        self.invalid_threshold = float(obj.get('threshold'))
                        print(f"Invalidity threshold loaded: {self.invalid_threshold}")
                except Exception as te:
                    print(f"Warning: failed to load validity threshold: {te}")

            # Valida consistência entre snapshot do experimento e o ambiente atual
            self._validate_experiment_snapshot()
            # Calibra limiar de invalidez a partir da curva ROC (se disponível)
            self._calibrate_invalid_threshold_from_roc()

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Erro: Artefato não encontrado! Verifique se o ID do experimento ('{eid}') está correto e se os arquivos existem. Detalhe: {e}")
        except Exception as e:
            print(f"ERRO CRÍTICO na inicialização: {e}")
            raise e
        
        print("--- Orquestrador pronto ---") 
        
    def _predict_validity_probability(self, feature_vector):
        """
        Return the probability of the sample being INVALID (class 0).

        Parameters
        ----------
        feature_vector : list[float]
            Feature vector used by the classifier.

        Returns
        -------
        float | None
            Probability in [0, 1] if classifier is available; otherwise `None`.
        """
        if self.validity_classifier is None or not feature_vector:
            return None
        try:
            proba = self.validity_classifier.predict_proba([feature_vector])[0]
            classes = self._validity_classifier_classes or self._extract_classifier_classes(self.validity_classifier)
            if not classes or 0 not in classes:
                return None
            idx_invalid = classes.index(0)
            return float(proba[idx_invalid])
        except Exception as err:
            print(f"Warning: validity classifier failed to evaluate sample: {err}")
            return None

    def _extract_classifier_classes(self, clf):
        """
        Safely get class labels from a plain estimator or a Pipeline.
        """
        try:
            # Direct estimator (e.g., LogisticRegression)
            if hasattr(clf, "classes_"):
                return list(clf.classes_)
            # Pipeline case: assume final step is 'logisticregression' or similar
            if hasattr(clf, "named_steps"):
                for step in reversed(clf.named_steps.values()):
                    if hasattr(step, "classes_"):
                        return list(step.classes_)
        except Exception:
            pass
        return []

    def _calibrate_invalid_threshold_from_roc(self):
        try:
            roc_path = self.exp_dir / "metrics" / "roc_curve.json"
            thr_path = self.exp_dir / "metrics" / "validity_threshold.json"
            if thr_path.exists():
                return
            if not roc_path.exists():
                return
            with open(roc_path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            fpr = obj.get('fpr')
            tpr = obj.get('tpr')
            thresholds = obj.get('thresholds')
            if not (isinstance(fpr, list) and isinstance(tpr, list) and isinstance(thresholds, list)):
                return
            if not (len(fpr) == len(tpr) == len(thresholds)):
                return
            # Youden's J statistic
            best_idx = int(np.argmax(np.array(tpr) - np.array(fpr)))
            best_thr = float(thresholds[best_idx])
            with open(thr_path, 'w', encoding='utf-8') as f:
                json.dump({"threshold": best_thr, "method": "youden"}, f, ensure_ascii=False, indent=2)
            self.invalid_threshold = best_thr
            print(f"Calibrated invalidity threshold from ROC: {best_thr}")
        except Exception:
            pass

    def _validate_experiment_snapshot(self):
        """
        Validate experiment `config_snapshot.json` against current environment.

        Checks include:
        - BuildingConfig name and coordinates
        - VectorConfig wall segment count
        - Constants (beam thickness)
        - Feature size: scaler vs snapshot vs recomputed features

        Raises
        ------
        RuntimeError
            If hard incompatibilities are detected between model/scaler/features.
        """
        try:
            if not self._config_snapshot_path.exists():
                raise RuntimeError(
                    f"Snapshot de configuração não encontrado em '{self._config_snapshot_path}'."
                )

            with open(self._config_snapshot_path, 'r', encoding='utf-8') as f:
                snap = json.load(f)

            # Imports locais para evitar dependências cíclicas na carga
            from config.settings import BuildingConfig, NeuralNetConfig
            from config.constants import DEFAULT_BEAM_WIDTH_CM, DEFAULT_BUILDING_COORDINATES, DEFAULT_SLAB_COORDINATES
            from config.vector_config import VectorConfig
            from config import paths as cfg_paths
            from utils.feature_engineer import FeatureEngineer

            hard_mismatches = []
            soft_warnings = []

            # BuildingConfig checks
            sb = snap.get('BuildingConfig', {})
            if sb:
                if sb.get('NAME') != BuildingConfig.NAME:
                    soft_warnings.append(f"Building NAME diverge: snapshot='{sb.get('NAME')}', atual='{BuildingConfig.NAME}'")

                def as_tuple_list(x):
                    return tuple(tuple(map(float, p)) for p in x) if x is not None else None

                snap_building = as_tuple_list(sb.get('BUILDING_COORDINATES'))
                snap_slab = as_tuple_list(sb.get('SLAB_COORDINATES'))
                cur_building = tuple(tuple(map(float, p)) for p in DEFAULT_BUILDING_COORDINATES)
                cur_slab = tuple(tuple(map(float, p)) for p in DEFAULT_SLAB_COORDINATES)
                if snap_building and snap_building != cur_building:
                    soft_warnings.append("BUILDING_COORDINATES divergem entre snapshot e constantes atuais")
                if snap_slab and snap_slab != cur_slab:
                    soft_warnings.append("SLAB_COORDINATES divergem entre snapshot e constantes atuais")

            # VectorConfig checks
            sv = snap.get('VectorConfig', {})
            if sv:
                expected_count = sv.get('WALL_SEGMENTS_COUNT')
                current_count = len(VectorConfig.WALL_SEGMENTS)
                if expected_count is not None and expected_count != current_count:
                    soft_warnings.append(f"WALL_SEGMENTS_COUNT diverge: snapshot={expected_count}, atual={current_count}")

            # Constants checks
            sc = snap.get('Constants', {})
            if sc:
                snap_beam_thick = sc.get('BEAM_THICKNESS_CM')
                if snap_beam_thick is not None and float(snap_beam_thick) != float(DEFAULT_BEAM_WIDTH_CM):
                    soft_warnings.append(f"BEAM_THICKNESS_CM diverge: snapshot={snap_beam_thick}, atual={DEFAULT_BEAM_WIDTH_CM}")

            # Feature size checks
            snn = snap.get('NeuralNetConfig', {})
            snap_input_size = snn.get('INPUT_SIZE') if snn else None
            snap_schema = snn.get('FEATURE_SCHEMA_VERSION') if snn else None
            scaler_in = getattr(self.feature_pipeline.scaler_X, 'n_features_in_', None)
            model_input = getattr(self.nn_manager, '_input_size', None)

            # Computa dimensão de features com FeatureEngineer atual a partir do CSV semente
            computed_len = None
            try:
                self.input_processor.csv_path = cfg_paths.SEED_VECTOR_CSV
                segs = self.input_processor.read_length_from_csv()
                if segs:
                    col_polys, beam_defs = self.input_processor.process_segments(segs)
                    fv = FeatureEngineer(col_polys, beam_defs).extract_features()
                    computed_len = len(fv)
            except Exception as e:
                hard_mismatches.append(
                    f"Falha ao recalcular as features atuais a partir do CSV semente: {e}"
                )

            # O contrato dos artefatos já valida nomes e ordem. O snapshot e a
            # recomputação independente abaixo validam o contexto do experimento.
            if snap_schema != NeuralNetConfig.FEATURE_SCHEMA_VERSION:
                hard_mismatches.append(
                    "FEATURE_SCHEMA_VERSION diverge entre snapshot e código atual "
                    f"({snap_schema!r} != {NeuralNetConfig.FEATURE_SCHEMA_VERSION!r})"
                )
            if snap_input_size is not None and scaler_in is not None and snap_input_size != scaler_in:
                hard_mismatches.append(f"INPUT_SIZE (snapshot vs scaler) diverge: {snap_input_size} vs {scaler_in}")
            if snap_input_size is not None and computed_len is not None and snap_input_size != computed_len:
                hard_mismatches.append(f"INPUT_SIZE (snapshot vs features) diverge: {snap_input_size} vs {computed_len}")

            # Regras duras: modelo, scaler e extrator devem coincidir exatamente.
            if model_input is not None and scaler_in is not None and model_input != scaler_in:
                hard_mismatches.append(f"INPUT_SIZE incompatível: modelo={model_input}, scaler={scaler_in}")
            if model_input is not None and computed_len is not None and computed_len != model_input:
                hard_mismatches.append(
                    f"INPUT_SIZE incompatível: modelo={model_input}, features_atual={computed_len}"
                )

            if soft_warnings:
                print("AVISOS de snapshot/configuração:\n - " + "\n - ".join(soft_warnings))

            if hard_mismatches:
                msg = ("\nInconsistências críticas entre artefatos e ambiente que impedem a inferência:\n- "
                       + "\n- ".join(hard_mismatches)
                       + "\nGaranta que modelo (.pth), pipeline (scalers) e FeatureEngineer estejam alinhados.")
                raise RuntimeError(msg)
            else:
                print("Artefatos compatíveis: schema, nomes, ordem e dimensões coincidem.")

        except Exception as e:
            # Propaga erro para impedir uso inconsistente
            raise

    def _validate_feature_vector(self, feature_vector: list) -> list:
        """Reject any runtime feature vector that violates the saved contract."""
        values = np.asarray(feature_vector, dtype=float)
        if values.ndim != 1:
            raise RuntimeError(
                f"Vetor de features deve ser unidimensional; shape recebido={values.shape}."
            )
        contract = self.feature_pipeline.artifact_contract
        if not contract:
            raise RuntimeError("Pipeline carregada sem contrato semântico.")
        expected_n = int(contract["input_size"])
        if len(values) != expected_n:
            raise RuntimeError(
                "Tamanho do vetor de features incompatível com o artefato "
                f"({len(values)} != {expected_n}); inferência cancelada."
            )
        if not np.isfinite(values).all():
            raise RuntimeError("Vetor de features contém NaN ou infinito.")
        return values.tolist()
    
    def predict_from_segments(self, segments: list) -> tuple[float, float, float, float | None]:
        """
        Predict steel/concrete/formwork directly from a pre-built segment list.

        Skips CSV serialisation and parsing entirely — use this in the
        optimisation hot-loop instead of ``predict_from_csv``.

        Parameters
        ----------
        segments : list[dict]
            Segment dicts as produced by ``DesignSpace.segments_from_vector`` or
            ``LengthProcessor.read_length_from_csv``.

        Returns
        -------
        tuple[float, float, float, float | None]
            ``(steel_pred, concrete_geom, formwork_area_m2, prob_invalid)``
        """
        if not segments:
            raise ValueError("Lista de segmentos vazia.")

        column_polygons, beam_definitions = self.input_processor.process_segments(segments)
        feature_engineer = FeatureEngineer(column_polygons, beam_definitions)
        feature_vector = self._validate_feature_vector(
            feature_engineer.extract_features()
        )

        concreto_geom = get_geometric_concrete_volume(column_polygons, beam_definitions)
        formwork_area = calculate_column_formwork_area(column_polygons)
        feature_vector_scaled = self.feature_pipeline.transform_features([feature_vector])
        prediction_scaled = self.nn_manager.predict(feature_vector_scaled)
        prediction_final = self.feature_pipeline.inverse_transform_outputs(prediction_scaled)

        if prediction_final.ndim == 2 and prediction_final.shape[1] >= 1:
            aco_predito = float(prediction_final[0][0])
        else:
            raise RuntimeError("Predição inválida: modelo não retornou pelo menos 1 saída para aço.")

        prob_invalid = self._predict_validity_probability(feature_vector)
        return aco_predito, float(concreto_geom), float(formwork_area), prob_invalid

    def predict_from_csv(self, csv_path_or_buffer) -> tuple[float, float, float, float | None]:
        """
        Predict steel from CSV/buffer and compute geometric concrete/formwork quantities.

        Parameters
        ----------
        csv_path_or_buffer : str | io.StringIO
            Path to the CSV or an in-memory buffer.

        Returns
        -------
        tuple[float, float, float, float | None]
            `(steel_pred, concrete_geom, formwork_area_m2, prob_invalid)`; probability may be `None`.
        """
        # 1. Lê os segmentos do CSV/buffer usando o método já existente
        self.input_processor.csv_path = csv_path_or_buffer
        segments = self.input_processor.read_length_from_csv()
        if not segments:
            raise ValueError("Não foi possível ler os segmentos do CSV/buffer.")

        # 2. Processa geometria e extrai features
        column_polygons, beam_definitions = self.input_processor.process_segments(segments)
        feature_engineer = FeatureEngineer(column_polygons, beam_definitions)
        feature_vector = self._validate_feature_vector(
            feature_engineer.extract_features()
        )

        # 2b. Concreto e área de forma geométricos exatos a partir da geometria
        concreto_geom = get_geometric_concrete_volume(column_polygons, beam_definitions)
        formwork_area = calculate_column_formwork_area(column_polygons)

        # 3. Predição do AÇO com o modelo surrogate (suporta OUTPUT_SIZE 1 ou 2)
        feature_vector_scaled = self.feature_pipeline.transform_features([feature_vector])
        prediction_scaled = self.nn_manager.predict(feature_vector_scaled)
        prediction_final = self.feature_pipeline.inverse_transform_outputs(prediction_scaled)

        # Se o modelo tiver 2 saídas [steel, concrete], usamos apenas o steel (índice 0)
        # Se tiver 1 saída, usa essa única previsão como steel
        if prediction_final.ndim == 2 and prediction_final.shape[1] >= 1:
            aco_predito = float(prediction_final[0][0])
        else:
            raise RuntimeError("Predição inválida: modelo não retornou pelo menos 1 saída para aço.")

        prob_invalid = self._predict_validity_probability(feature_vector)

        return aco_predito, float(concreto_geom), float(formwork_area), prob_invalid

    def run_comparison(self):
        """
        End-to-end comparison: surrogate prediction vs TQS real analysis.

        Steps:
        1) Read CSV → geometry → features
        2) Predict steel with surrogate model
        3) Run TQS global processing to obtain real values
        4) Print a comparison report
        """
        try:
            # --- ETAPA 1: Ler o CSV para obter os segmentos ---
            test_csv_path = paths.INFERENCE_TEST_CSV
            print(f"\n[PASSO 1/5] Lendo segmentos de '{test_csv_path.name}'...")

            self.input_processor.csv_path = test_csv_path
            segments = self.input_processor.read_length_from_csv()

            if not segments:
                raise ValueError(f"Não foi possível ler os segmentos do arquivo '{test_csv_path}'")
            print(f"   - {len(segments)} segmentos carregados com sucesso.")

            # --- ETAPA 2: Extrair o Vetor de Features (Lógica Manual) ---
            print("\n[PASSO 2/5] Extraindo o vetor de features da geometria...")
            # 2a. Processa segmentos para obter a geometria (pilares e vigas)
            column_polygons, beam_definitions = self.input_processor.process_segments(segments)
            print("   - Geometria de pilares e vigas processada.")


            # 2b. Usa o FeatureEngineer para criar o vetor de input final
            feature_engineer = FeatureEngineer(column_polygons, beam_definitions)
            feature_vector = feature_engineer.extract_features()
            print(f"   - Vetor de features BRUTO extraído: {np.array(feature_vector)}")
            
            # --- ETAPA 3: Predição com o Modelo Surrogate ---
            print("\n[PASSO 3/5] Executando predição com o modelo surrogate...")
            
            # 3a. Validar o contrato exato antes de normalizar
            feature_vector = self._validate_feature_vector(feature_vector)

            # 3b. Normalizar o vetor de features
            feature_vector_scaled = self.feature_pipeline.transform_features([feature_vector])
            
            # 3c. Fazer a predição (retorna valor normalizado)
            prediction_scaled = self.nn_manager.predict(feature_vector_scaled)
            print(f" - Vetor de features NORMALIZADO: {feature_vector_scaled}")
            # 3d. Desnormalizar o resultado para obter o valor final
            prediction_final = self.feature_pipeline.inverse_transform_outputs(prediction_scaled)
            
            aco_predito = prediction_final[0][0]
            concreto_predito = get_geometric_concrete_volume(column_polygons, beam_definitions)
            print(f"   -> PREDIÇÃO DO MODELO: Aço={aco_predito:.2f} kgf | Concreto={concreto_predito:.2f} m³")

            
            # --- ETAPA 4: Análise Real com TQS ---
            print("\n[PASSO 3/4] Executando análise no TQS para obter o resultado real...")
            aco_real, concreto_real = self._execute_full_tqs_analysis(segments)
            if aco_real is None:
                raise RuntimeError("Falha ao executar a análise completa do TQS.")
            print(f"   -> RESULTADO REAL TQS: Aço={aco_real:.2f} kgf | Concreto={concreto_real:.2f} m³")
            
            # --- ETAPA 4: Relatório Comparativo ---
            self._generate_report(aco_predito, concreto_predito, aco_real, concreto_real)

        except Exception as e:
            print(f"\n--- ERRO CRÍTICO DURANTE A EXECUÇÃO ---")
            print(f"Erro: {e}")
            print(traceback.format_exc())

    def run_tqs_on_csv(self, csv_path: str) -> tuple[float, float]:
        """
        Executa o TQS diretamente a partir de um CSV de segmentos e retorna (aço, concreto).

        Parameters
        ----------
        csv_path : str
            Caminho para o arquivo CSV no formato `x;y;dx;dy;length;maxlength`.

        Returns
        -------
        tuple[float, float]
            `(aco_real, concreto_real)` lidos do TQS após o processamento global.
        """
        self.input_processor.csv_path = csv_path
        segments = self.input_processor.read_length_from_csv()
        if not segments:
            raise ValueError(f"Não foi possível ler os segmentos do arquivo '{csv_path}'.")
        aco_real, concreto_real = self._execute_full_tqs_analysis(segments)
        if aco_real is None:
            raise RuntimeError("Falha ao executar a análise completa do TQS.")
        return aco_real, concreto_real

    def _execute_full_tqs_analysis(self, segments: list) -> tuple:
        """
        Create structural model in TQS and execute global processing.

        Parameters
        ----------
        segments : list
            Input segments to build the structural model.

        Returns
        -------
        tuple
            `(steel_real, concrete_real)` parsed from TQS outputs.
        """
        tqs_manager = TQSModelManager(BuildingConfig.NAME)
        
        column_polygons, beam_definitions = self.input_processor.process_segments(segments)
        
        print("   - Criando modelo estrutural no TQS...")
        model_created = tqs_manager.create_building_model_and_elements(column_polygons, beam_definitions)
        if not model_created:
            print("   ERRO: Falha ao criar o modelo no TQS.")
            return None, None
        
        print("   - Executando processamento global do TQS...")
        RunModel(BuildingConfig.NAME)
        
        tqs_output_file = BuildingConfig.TQS_RESULTS_FILE
        timeout = int(getattr(BuildingConfig, 'TQS_TIMEOUT_SEC', 120))
        start_wait_time  = time.time()
        while not tqs_output_file.exists():
            if time.time() - start_wait_time  > timeout:
                print(f"   ERRO: Timeout após {timeout}s esperando pelo arquivo de resultados do TQS.")
                return None, None
            time.sleep(0.5)

        print("   - Extraindo resultados do arquivo gerado pelo TQS...")
        steel_str, concrete_str = extract_material_summary(tqs_output_file)
        
        if steel_str is None or concrete_str is None:
            print("   ERRO: Não foi possível extrair os totais do arquivo de resultados do TQS.")
            return None, None
            
        aco_real = float(steel_str.replace(",", "."))
        concreto_real = float(concrete_str.replace(",", "."))
        
        return aco_real, concreto_real

    def _generate_report(self, aco_predito, concreto_predito, aco_real, concreto_real):
        """Print the final formatted comparison report."""
        print("\n[PASSO 4/4] Relatório Final de Comparação")
        print("-" * 75)

        erro_concreto = (abs(concreto_predito - concreto_real) / concreto_real * 100) if concreto_real != 0 else float('inf')
        erro_aco = (abs(aco_predito - aco_real) / aco_real * 100) if aco_real != 0 else float('inf')

        print(f"{'MÉTRICA':<20} | {'SURROGATE (PREDITO)':>20} | {'TQS (REAL)':>15} | {'ERRO (%)':>10}")
        print("-" * 75)
        print(f"{'Volume Concreto (m³)':<20} | {concreto_predito:>20.2f} | {concreto_real:>15.2f} | {erro_concreto:>9.2f}%")
        print(f"{'Peso Aço (kgf)':<20} | {aco_predito:>20.2f} | {aco_real:>15.2f} | {erro_aco:>9.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(paths.RESULTS_DIR / "solucao_otima.csv"))
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--exp", default=None)
    args = parser.parse_args()
    inf = BuildingInference(args.exp)
    if args.predict:
        steel_pred, conc_geom, form_area, prob = inf.predict_from_csv(args.csv)
        print(f"Surrogate: steel={steel_pred:.2f} kgf, concrete_geom={conc_geom:.3f} m³, form_area={form_area:.2f} m², prob_invalid={prob}")
    aco_real, concreto_real = inf.run_tqs_on_csv(args.csv)
    print(f"TQS: steel={aco_real:.2f} kgf, concrete={concreto_real:.3f} m³")
    print("-" * 75)



# (main) bloco único acima

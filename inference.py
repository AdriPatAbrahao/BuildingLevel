# BuildingOptimization/inference.py

import traceback
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
from utils.geometric_calculator import get_geometric_concrete_volume
            
# --- CONFIGURAÇÃO DA INFERÊNCIA ---
# Você só precisa definir o ID do experimento que quer usar.
# Ex: "20250830-180000_Treino_com_200_amostras_e_BN"
##EXPERIMENT_ID = "20250901-234401_Treino_com_200_amostras_e_BN"
EXPERIMENT_ID = "20250908-000837_Treino_com_1300_amostras_e_BN"


EXPERIMENT_DIR = Path("outputs/experiments") / EXPERIMENT_ID
PIPELINE_PATH = EXPERIMENT_DIR / "feature_pipeline.pkl"
MODEL_PATH = EXPERIMENT_DIR / "trained_model.pth"
CONFIG_SNAPSHOT_PATH = EXPERIMENT_DIR / "config_snapshot.json"

class BuildingInference:
    def __init__(self):
        try:
            print("--- Inicializando o Orquestrador de Inferência ---")

            if not EXPERIMENT_DIR.exists():
                raise FileNotFoundError(f"Diretório do experimento não encontrado: '{EXPERIMENT_DIR}'")
            
            # Carrega a pipeline e o modelo
            print(f"Carregando artefatos do experimento: '{EXPERIMENT_ID}'")
            self.feature_pipeline = FeaturePipeline()
            self.feature_pipeline.load(PIPELINE_PATH) # Carrega os scalers
        
            self.nn_manager = NeuralNetworkManager()

            self.input_processor = self.feature_pipeline.input_processor
            if not self.nn_manager.load_model(MODEL_PATH):
                raise RuntimeError("Falha ao carregar o modelo treinado.")

            # Valida consistência entre snapshot do experimento e o ambiente atual
            self._validate_experiment_snapshot()

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Erro: Artefato não encontrado! Verifique se o ID do experimento ('{EXPERIMENT_ID}') está correto e se os arquivos existem. Detalhe: {e}")
        except Exception as e:
            print(f"ERRO CRÍTICO na inicialização: {e}")
            raise e
        
        print("--- Orquestrador pronto ---") 
        
    def _validate_experiment_snapshot(self):

        """
        Compara config_snapshot.json do experimento com o estado atual de
        - BuildingConfig (nome e coordenadas)
        - VectorConfig (contagem de segmentos de parede)
        - Constants (largura de viga)
        - Tamanho de features (scaler e FeatureEngineer) vs INPUT_SIZE salvo
        Lança erro em caso de divergência relevante.
        """
        try:
            if not CONFIG_SNAPSHOT_PATH.exists():
                print(f"AVISO: Snapshot de configuração não encontrado em '{CONFIG_SNAPSHOT_PATH}'.")
                return

            with open(CONFIG_SNAPSHOT_PATH, 'r', encoding='utf-8') as f:
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
                print(f"AVISO: Falha ao estimar tamanho de features atual: {e}")

            # Snapshot divergente é apenas aviso; o que importa é compatibilidade entre modelo e pipeline/features
            if snap_input_size is not None and scaler_in is not None and snap_input_size != scaler_in:
                soft_warnings.append(f"INPUT_SIZE (snapshot vs scaler) diverge: {snap_input_size} vs {scaler_in}")
            if snap_input_size is not None and computed_len is not None and snap_input_size != computed_len:
                soft_warnings.append(f"INPUT_SIZE (snapshot vs features) diverge: {snap_input_size} vs {computed_len}")

            # Regras duras: modelo deve bater com scaler; features podem ter extras (faremos corte automático)
            if model_input is not None and scaler_in is not None and model_input != scaler_in:
                hard_mismatches.append(f"INPUT_SIZE incompatível: modelo={model_input}, scaler={scaler_in}")
            if model_input is not None and computed_len is not None:
                if computed_len < model_input:
                    # Faltando features necessárias para o modelo: erro crítico
                    hard_mismatches.append(
                        f"INPUT_SIZE insuficiente: modelo={model_input}, features_atual={computed_len}"
                    )
                elif computed_len > model_input:
                    # Features atuais têm colunas a mais: aceitável, faremos slice; apenas aviso
                    soft_warnings.append(
                        f"INPUT_SIZE maior nas features atuais ({computed_len}) que o modelo ({model_input}); será aplicado corte automático."
                    )

            if soft_warnings:
                print("AVISOS de snapshot/configuração:\n - " + "\n - ".join(soft_warnings))

            if hard_mismatches:
                msg = ("\nInconsistências críticas entre artefatos e ambiente que impedem a inferência:\n- "
                       + "\n- ".join(hard_mismatches)
                       + "\nGaranta que modelo (.pth), pipeline (scalers) e FeatureEngineer estejam alinhados.")
                raise RuntimeError(msg)
            else:
                print("Artefatos compatíveis: modelo, scaler e features têm o mesmo INPUT_SIZE.")

        except Exception as e:
            # Propaga erro para impedir uso inconsistente
            raise
    
    def predict_from_csv(self, csv_path_or_buffer) -> tuple[float, float]:
        """
        Executa uma predição a partir de um arquivo CSV ou buffer de memória.
        
        Esta é a função centralizada que será usada tanto pela otimização
        quanto pela inferência.

        Args:
            csv_path_or_buffer: O caminho para o arquivo CSV ou um buffer StringIO.

        Returns:
            Uma tupla (aco_predito, concreto_predito).
        """
        # 1. Lê os segmentos do CSV/buffer usando o método já existente
        self.input_processor.csv_path = csv_path_or_buffer
        segments = self.input_processor.read_length_from_csv()
        if not segments:
            raise ValueError("Não foi possível ler os segmentos do CSV/buffer.")

        # 2. Processa geometria e extrai features
        column_polygons, beam_definitions = self.input_processor.process_segments(segments)
        feature_engineer = FeatureEngineer(column_polygons, beam_definitions)
        feature_vector = feature_engineer.extract_features()
        # Ajuste compatível: se adicionamos novas features mas o scaler espera menos, faz slice
        expected_n = getattr(self.feature_pipeline.scaler_X, 'n_features_in_', None)
        if expected_n is not None and len(feature_vector) != expected_n:
            if len(feature_vector) > expected_n:
                feature_vector = feature_vector[:expected_n]
            else:
                raise RuntimeError(f"Tamanho de features ({len(feature_vector)}) menor que o esperado pelo scaler ({expected_n}).")

        # 2b. Concreto geométrico exato a partir da geometria
        concreto_geom = get_geometric_concrete_volume(column_polygons, beam_definitions)
        
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

        return aco_predito, float(concreto_geom)

    def run_comparison(self):
        """
        Executa o fluxo completo:
        1. Lê o CSV de teste e o transforma em um vetor de features.
        2. Usa o modelo surrogate (.pth) para fazer uma predição.
        3. Roda a análise completa no TQS para obter o resultado real.
        4. Apresenta um relatório comparativo.
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
            
            # 3a. Ajustar tamanho das features para bater com o scaler, se necessário
            expected_n = getattr(self.feature_pipeline.scaler_X, 'n_features_in_', None)
            if expected_n is not None and len(feature_vector) != expected_n:
                if len(feature_vector) > expected_n:
                    feature_vector = feature_vector[:expected_n]
                else:
                    raise RuntimeError(
                        f"Tamanho de features ({len(feature_vector)}) menor que o esperado pelo scaler ({expected_n})."
                    )

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

    def _execute_full_tqs_analysis(self, segments: list) -> tuple:
        """Orquestra a criação e execução do modelo no TQS."""
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
        timeout = 20
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
        """Imprime o relatório final formatado."""
        print("\n[PASSO 4/4] Relatório Final de Comparação")
        print("-" * 75)

        erro_concreto = (abs(concreto_predito - concreto_real) / concreto_real * 100) if concreto_real != 0 else float('inf')
        erro_aco = (abs(aco_predito - aco_real) / aco_real * 100) if aco_real != 0 else float('inf')

        print(f"{'MÉTRICA':<20} | {'SURROGATE (PREDITO)':>20} | {'TQS (REAL)':>15} | {'ERRO (%)':>10}")
        print("-" * 75)
        print(f"{'Volume Concreto (m³)':<20} | {concreto_predito:>20.2f} | {concreto_real:>15.2f} | {erro_concreto:>9.2f}%")
        print(f"{'Peso Aço (kgf)':<20} | {aco_predito:>20.2f} | {aco_real:>15.2f} | {erro_aco:>9.2f}%")
        print("-" * 75)



if __name__ == '__main__':
    inference_runner = BuildingInference()
    inference_runner.run_comparison()

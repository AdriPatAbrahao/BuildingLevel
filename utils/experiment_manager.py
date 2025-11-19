import json
from pathlib import Path
import time
from typing import Any, Dict
import shutil

# Importe suas classes de configuração para poder tirar um "snapshot" delas
from config.settings import BuildingConfig, RunConfig, NeuralNetConfig
from config.constants import DEFAULT_BEAM_WIDTH_CM # Exemplo de constante
from config.vector_config import VectorConfig

class ExperimentManager:
    """
    Gerencia uma única execução de treinamento, salvando todos os artefatos
    e configurações em um diretório único e autocontido.
    """
    def __init__(self, base_dir: Path, run_name: str = None):
        """
        Inicializa o gerenciador para uma nova execução.

        Args:
            base_dir (Path): O diretório base para todos os experimentos (ex: 'outputs/experiments').
            run_name (str, optional): Um nome descritivo para a execução.
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        dir_name = f"{timestamp}_{run_name}" if run_name else timestamp
        
        self.run_dir = base_dir / dir_name
        self.plots_dir = self.run_dir / "plots"
        self.images_dir = self.run_dir / "images"
        self.metrics_dir = self.run_dir / "metrics"
        
        # Cria a estrutura de diretórios
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        
        print(f"[ExperimentManager] Nova execução iniciada: {self.run_dir.name}")
        print(f"[ExperimentManager] Diretório: {self.run_dir.resolve()}")

        # Salva um snapshot das configurações no momento da criação
        self._save_config_snapshot()

    def _save_config_snapshot(self):
        """Salva uma cópia de todas as configurações relevantes em um arquivo JSON."""
        config_snapshot = {
            "BuildingConfig": {k: v for k, v in vars(BuildingConfig).items() if not k.startswith('__')},
            "RunConfig": {k: v for k, v in vars(RunConfig).items() if not k.startswith('__')},
            "NeuralNetConfig": {k: v for k, v in vars(NeuralNetConfig).items() if not k.startswith('__')},
            "VectorConfig": {
                "WALL_SEGMENTS_COUNT": len(VectorConfig.WALL_SEGMENTS),
                # Não salvamos os segmentos inteiros para manter o arquivo limpo
            },
            "Constants": {
                "BEAM_THICKNESS_CM": DEFAULT_BEAM_WIDTH_CM,
            }
        }
        
        # Converte Paths para strings para serem serializáveis em JSON
        def convert_paths_to_strings(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    convert_paths_to_strings(v)
                elif isinstance(v, Path):
                    d[k] = str(v)
        
        convert_paths_to_strings(config_snapshot)

        snapshot_path = self.run_dir / "config_snapshot.json"
        with open(snapshot_path, 'w') as f:
            json.dump(config_snapshot, f, indent=4)
        print(f"[ExperimentManager] Snapshot das configurações salvo em '{snapshot_path.name}'")

    def log_metadata(self, metadata: Dict[str, Any]):
        """Salva os resultados e métricas da execução."""
        metadata_path = self.run_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"[ExperimentManager] Metadados da execução salvos em '{metadata_path.name}'")

    # Métodos para obter os caminhos dos artefatos
    def get_model_path(self) -> Path:
        return self.run_dir / "trained_model.pth"

    def get_pipeline_path(self) -> Path:
        return self.run_dir / "feature_pipeline.pkl"

    def get_metrics_dir(self) -> Path:
        return self.metrics_dir

    def write_json(self, path: Path, obj: Dict[str, Any]):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False)

    def append_ndjson(self, path: Path, obj: Dict[str, Any]):
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

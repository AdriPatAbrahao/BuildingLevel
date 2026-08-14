# config/paths.py
from pathlib import Path

# --- Caminhos Base ---
# Define a raiz do projeto como a pasta pai da pasta 'config'
PROJECT_ROOT = Path(__file__).parent.parent

# --- Diretórios de Dados ---
DATA_DIR = PROJECT_ROOT / "data"
TQS_OUTPUT_DIR = Path(r"C:\TQS") # Caminho externo para o TQS

# --- Diretórios de Saída do Projeto ---
# Todos os resultados gerados pelo SEU código
OUTPUTS_DIR = PROJECT_ROOT / "outputs" # Um único diretório de saída
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
RESULTS_DIR = OUTPUTS_DIR / "results"
IMAGE_DIR = OUTPUTS_DIR / "images"
PREDICTIONS_CSV_PATH = RESULTS_DIR / "predictions_output.csv"
FINAL_VECTORS_CSV_PATH = RESULTS_DIR / "final_tqs_input_vectors.csv"

# --- Caminhos para Arquivos de Dados de Entrada ---
# Nomes de arquivos de semente
SEED_VECTOR_CSV = DATA_DIR / "Building1b.csv"
SEED_VECTOR_CSV_OPTIMIZATION = DATA_DIR / "Building1c.csv"
SEED_BINARY_CSV = DATA_DIR / "Building1.csv"
# Arquivo de teste para inferência
INFERENCE_TEST_CSV = DATA_DIR / "BuildingTest01.csv"

# --- Criação de Diretórios ---
# Garante que todos os diretórios de saída existam ao iniciar
DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
EXPERIMENTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

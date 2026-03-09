import csv
import locale
from pathlib import Path
from config.paths import FINAL_VECTORS_CSV_PATH
from typing import List, Dict
from contextlib import contextmanager


@contextmanager
def use_decimal_point():
    """Temporarily force '.' as decimal separator regardless of OS locale."""
    original_locale = locale.setlocale(locale.LC_NUMERIC)
    try:
        try:
            locale.setlocale(locale.LC_NUMERIC, 'C')
        except locale.Error as e:
            print(f"Warning: Failed to set LC_NUMERIC to 'C' ({e}). Continuing with current locale.")
        yield
    finally:
        try:
            locale.setlocale(locale.LC_NUMERIC, original_locale or '')
        except locale.Error as e:
            print(f"Warning: Failed to restore LC_NUMERIC locale ({e}).")


def _read_cleanup_patterns(dat_file_path: Path) -> List[str]:
    """
    Lê padrões de arquivo de um DAT de limpeza do TQS.

    Formato de cada linha: padrão,flag,col3,col4,descrição
    Flags: S = Sim (apagar), I = Incondicional (apagar), N = Não (preservar).
    Retorna apenas padrões com flag S ou I.
    O caractere '$' do TQS é convertido para '?' para uso com glob.
    """
    patterns: List[str] = []
    if not dat_file_path.exists():
        return patterns
    for encoding in ('latin-1', 'cp1252', 'utf-8'):
        try:
            with open(dat_file_path, encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if len(parts) < 2:
                        continue
                    pattern = parts[0].strip()
                    flag = parts[1].strip().upper()
                    if flag in ('S', 'I'):
                        patterns.append(pattern.replace('$', '?'))
            break
        except Exception:
            continue
    return patterns


def cleanup_building_files(building_name: str, tqs_base_dir: Path, dat_dir: Path) -> None:
    """
    Apaga arquivos temporários do TQS antes de gerar um novo edifício.

    - Padrões de LIMPA ESPACIAL.DAT → pasta <building>/ESPACIAL/
    - Padrões de LIMPA VIGAS.DAT    → cada subpasta de pavimento (exceto ESPACIAL e PILAR)
    - Padrões de LIMPAPILAR.DAT     → pasta <building>/PILAR/

    Args:
        building_name: Nome do edifício TQS (ex.: "OptimizedBuilding").
        tqs_base_dir:  Diretório base do TQS (ex.: Path(r"C:\\TQS")).
        dat_dir:       Diretório que contém os arquivos LIMPA*.DAT.
    """
    building_dir = tqs_base_dir / building_name
    if not building_dir.exists():
        return

    espacial_patterns = _read_cleanup_patterns(dat_dir / "LIMPA ESPACIAL.DAT")
    vigas_patterns    = _read_cleanup_patterns(dat_dir / "LIMPA VIGAS.DAT")
    pilar_patterns    = _read_cleanup_patterns(dat_dir / "LIMPAPILAR.DAT")

    def _delete_matching(folder: Path, patterns: List[str]) -> int:
        if not folder.exists():
            return 0
        deleted = 0
        for pattern in patterns:
            for file_path in folder.glob(pattern):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted += 1
                    except OSError as exc:
                        print(f"Cleanup warning: could not delete '{file_path}': {exc}")
        return deleted

    total = 0
    total += _delete_matching(building_dir / "ESPACIAL", espacial_patterns)
    total += _delete_matching(building_dir / "PILAR", pilar_patterns)

    skip = {"ESPACIAL", "PILAR"}
    for subdir in building_dir.iterdir():
        if subdir.is_dir() and subdir.name.upper() not in skip:
            total += _delete_matching(subdir, vigas_patterns)

    print(f"Cleanup '{building_name}': {total} arquivo(s) removido(s).")


def save_final_vectors_to_csv(configurations: List[List[Dict]]):
    """
    Salva os dados dos segmentos finais (vetores de comprimento) em um arquivo CSV.

    Args:
        configurations (List[List[Dict]]): Lista de configurações, onde cada configuração é
                                          uma lista de dicionários de segmentos.
    """

    output_path = FINAL_VECTORS_CSV_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not configurations:
        print("Nenhuma configuração válida para salvar.")
        return

    # Assume que todas as configurações têm a mesma estrutura de segmentos
    fieldnames = ['config_index', 'segment_index', 'start_x', 'start_y', 'end_x', 'end_y', 'length']

    try:
        with use_decimal_point():
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()

                for config_idx, config in enumerate(configurations):
                    for seg_idx, segment in enumerate(config):
                        # Formata números com ponto decimal e sem separador de milhar
                        writer.writerow({
                            'config_index': config_idx,
                            'segment_index': seg_idx,
                            'start_x': f"{float(segment.get('start', (None, None))[0]):.6f}",
                            'start_y': f"{float(segment.get('start', (None, None))[1]):.6f}",
                            'end_x': f"{float(segment.get('end', (None, None))[0]):.6f}",
                            'end_y': f"{float(segment.get('end', (None, None))[1]):.6f}",
                            'length': f"{float(segment.get('length', None)):.6f}"
                        })
        print(f"Vetores finais salvos com sucesso em: {output_path}")

    except Exception as e:
        # Propaga o erro para que o chamador saiba que o salvamento falhou
        print(f"Erro ao salvar CSV: {e}")
        raise

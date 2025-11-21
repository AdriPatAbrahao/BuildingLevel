import csv
import locale
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

import csv
from config.constants import  CSV_FINAL_PATH
from typing import List, Dict


def save_final_vectors_to_csv(configurations: List[List[Dict]]):
    """
    Salva os dados dos segmentos finais (vetores de comprimento) em um arquivo CSV.

    Args:
        configurations (List[List[Dict]]): Lista de configurações, onde cada configuração é
                                          uma lista de dicionários de segmentos.
        filename (str): Nome do arquivo CSV a ser salvo.
    """

    output_path = CSV_FINAL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not configurations:
        print("Nenhuma configuração válida para salvar.")
        return

    # Assume que todas as configurações têm a mesma estrutura de segmentos
    # Pega os nomes das colunas do primeiro segmento da primeira configuração
    # Adapte conforme a estrutura exata que você quer salvar
    fieldnames = ['config_index', 'segment_index', 'start_x', 'start_y', 'end_x', 'end_y', 'length']

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()

            for config_idx, config in enumerate(configurations):
                for seg_idx, segment in enumerate(config):
                    writer.writerow({
                        'config_index': config_idx,
                        'segment_index': seg_idx,
                        'start_x': segment.get('start', (None, None))[0],
                        'start_y': segment.get('start', (None, None))[1],
                        'end_x': segment.get('end', (None, None))[0],
                        'end_y': segment.get('end', (None, None))[1],
                        'length': segment.get('length', None)
                    })
        print(f"Vetores finais salvos com sucesso em: {output_path}")

    except Exception as e:
        print(f"Erro ao salvar CSV: {e}")
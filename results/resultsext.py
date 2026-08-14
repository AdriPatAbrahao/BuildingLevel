from bs4 import BeautifulSoup
import os

def extract_material_summary(file_path):
    """
    Extracts the total values for steel and concrete from the "Resumo de materiais" table in the HTML file.

    Args:
        file_path (str): The full path to the RESDES.HTM file.

    Returns:
        tuple[str, str]: A tuple (steel_value, concrete_value) with the extracted text values.
                         Returns None if the table or row is not found.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # Try reading the file with different encodings
    encodings = ["utf-8", "latin-1", "ISO-8859-1"]  # Common encodings for HTML files
    html_content = None

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                html_content = file.read()
            break  # Stop if the file is successfully read
        except UnicodeDecodeError:
            continue  # Try the next encoding

    # If no encoding worked, raise an error
    if html_content is None:
        print("Failed to read the file with the tested encodings.")
        return None

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "lxml")

    # Find the table with the title "Resumo de materiais"
    target_table = None
    for table in soup.find_all("table"):
        header = table.find("td", text="Resumo de materiais")
        if header:
            target_table = table
            break

    # Check if the table was found
    if not target_table:
        print("Table 'Resumo de materiais' not found.")
        return None

    # Find the "Totais" row
    totals_row = None
    for row in target_table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) > 0 and "Totais" in cells[0].get_text():
            totals_row = cells
            break

    # Check if the "Totais" row was found
    if not totals_row:
        print("Row 'Totais' not found in the table.")
        return None

    # Extract the values for steel and concrete
    try:
        steel_value = totals_row[13].get_text(strip=True)  # Steel value
        concrete_value = totals_row[14].get_text(strip=True)  # Concrete value
    except IndexError:
        print("Error: The table structure does not match the expected format.")
        return None

    return steel_value, concrete_value
    
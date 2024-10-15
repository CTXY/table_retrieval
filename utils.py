import pandas as pd
import json
from src.schema import Document
import jsonlines

def convert_table(line):
    # Extract column headers
    columns = [col["text"] for col in line["columns"]]
    
    # Prepare to collect rows of data
    data = []
    
    # We will create an empty list for each row based on the highest row index in cells
    max_row_idx = max(cell["row_idx"] for cell in line["cells"])
    for _ in range(max_row_idx):
        data.append([""] * len(columns))
    
    # Fill the data list with the actual data from cells
    for cell in line["cells"]:
        # Adjust for 0-based indexing in Python (assuming row_idx and col_idx are 1-based)
        row_idx = cell["row_idx"] - 1
        col_idx = cell["col_idx"] - 1
        data[row_idx][col_idx] = cell["text"]
    
    # Create a DataFrame using the data and columns
    df = pd.DataFrame(data, columns=columns)
    
    return df


def convert_file_to_table(file_path: str):

    # the process function is for OTT_QA dataset
    documents = []
    with jsonlines.open(file_path, 'r') as f:
        for line in f:
            # line = json.loads(line)
            table = convert_table(line)
            metadata = {'title': line['title'], 'columns': line['columns'], 'rows': line['rows']}
            documents.append(Document(id=line['id'], content=table, content_type='table', meta=metadata))

    return documents
import json

with open("d:/credit-risk-prediction-training-and-eda.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code"]

with open("d:/extracted_notebook.py", "w", encoding="utf-8") as out:
    for cell in code_cells:
        out.write("".join(cell))
        out.write("\n\n")

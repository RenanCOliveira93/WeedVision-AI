"""Executa os notebooks em ordem (nbconvert, kernel do venv) e grava as saídas nos .ipynb."""
import sys
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

root = Path(__file__).resolve().parents[1]
for p in sorted((root / "notebooks").glob("*.ipynb")):
    nb = nbformat.read(p, as_version=4)
    ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(root / "notebooks")}})
    nbformat.write(nb, p); print("executado", p.name, flush=True)

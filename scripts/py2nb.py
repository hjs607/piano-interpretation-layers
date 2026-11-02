"""percent 형식(.py, `# %%` / `# %% [markdown]`) → .ipynb 변환.

사용: python scripts/py2nb.py notebooks_src/01_percepiano.py notebooks/01_percepiano.ipynb
"""
import re
import sys
from pathlib import Path

import nbformat

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text()
cells = []
for block in re.split(r"^# %%", text, flags=re.M)[1:]:
    header, _, body = block.partition("\n")
    body = body.strip("\n")
    if header.strip().startswith("[markdown]"):
        lines = [re.sub(r"^# ?", "", ln) for ln in body.splitlines()]
        cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
    else:
        cells.append(nbformat.v4.new_code_cell(body))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
dst.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, dst)
print(f"{dst}: {len(cells)} cells")

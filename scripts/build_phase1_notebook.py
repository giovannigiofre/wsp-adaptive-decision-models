"""Build notebooks/phase1_llada.ipynb from committed results (no model loading).

Usage: python scripts/build_phase1_notebook.py && \
       jupyter nbconvert --to notebook --execute --inplace notebooks/phase1_llada.ipynb
Narrative text lives in NARRATIVE below so it can be edited after the runs.
"""
import json
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
NARR = json.loads((REPO / "notebooks/phase1_narrative.json").read_text())

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(NARR["intro"])
code("""import os, json, csv
from pathlib import Path
import pandas as pd
from IPython.display import Image, Markdown, display
REPO = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
R = REPO / 'results/phase1'
pd.set_option('display.precision', 3); pd.set_option('display.max_colwidth', 60)""")

md(NARR["1.0"])
code("""display(Markdown((R / 'env_and_speed.md').read_text()))""")

md(NARR["1.1"])
code("""runs = pd.read_csv(R / 'unmasking/runs.csv')
display(runs[['task','id','remasking','steps','gen_length','score','correct','mean_conf','wall_s']])
for pid in runs['id'].unique()[::3]:   # one prompt per task, to keep the notebook small
    display(Markdown(f'**{pid}**  (GIFs: `results/phase1/unmasking/{pid}_low_confidence.gif`, `..._random.gif`)'))
    display(Image(filename=str(R / f'unmasking/{pid}_heatmap.png'), width=760))""")
code("""# Quantify the left-to-right tendency inside blocks: Spearman correlation between position and commit step
import numpy as np
from scipy.stats import spearmanr
rows = []
for f in sorted((R / 'unmasking').glob('*_*.json')):
    d = json.load(open(f)); cs = np.array(d['commit_step'])
    name = f.stem
    strat = 'low_confidence' if name.endswith('low_confidence') else 'random'
    pid = name.replace('_low_confidence', '').replace('_random', '')
    rhos = [spearmanr(np.arange(32), cs[b:b+32]).correlation for b in range(0, len(cs), 32)]
    rows.append({'id': pid, 'remasking': strat, 'mean within-block rho(position, commit step)': np.nanmean(rhos)})
df = pd.DataFrame(rows); display(df.drop(columns='id').groupby('remasking').agg(['mean', 'min', 'max']))""")

md(NARR["1.2"])
code("""s = pd.read_csv(R / 'strategies.csv'); display(s)""")
code("""for t in ['sudoku', 'countdown', 'gsm8k']:
    p = R / f'strategies_{t}.png'
    if p.exists(): display(Image(filename=str(p), width=480))""")
code("""# paired over examples, averaged over the 3 step budgets: confidence minus each other strategy
display(pd.read_csv(R / 'strategies_paired.csv'))""")
md(NARR["1.2b"])

md(NARR["1.3"])
code("""display(pd.read_csv(R / 'calibration/metrics.csv'))
p = R / 'calibration/early_vs_late.csv'
if p.exists(): display(pd.read_csv(p))
display(pd.read_csv(R / 'calibration/sudoku_cells_by_digit.csv'))""")
code("""for f in ['reliability_answer_level.png', 'sudoku_token_level.png']:
    p = R / 'calibration' / f
    if p.exists(): display(Image(filename=str(p), width=720))""")
md(NARR["1.3b"])

md(NARR["1.4"])
code("""# image links in SUMMARY.md are relative to results/phase1/; rewrite them for the notebook folder
display(Markdown((R / 'SUMMARY.md').read_text().replace('](', '](../results/phase1/')))""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
(REPO / "notebooks").mkdir(exist_ok=True)
nbf.write(nb, REPO / "notebooks/phase1_llada.ipynb")
print("wrote notebooks/phase1_llada.ipynb")

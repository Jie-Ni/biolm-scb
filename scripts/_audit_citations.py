"""Audit which bib entries are uncited."""
import re, os
ROOT = r"C:/Users/Jie/Desktop/BioLM_Scaling_Physics/11_manuscript/arxiv_submission"
with open(os.path.join(ROOT, "references.bib"), encoding="utf-8") as f:
    bib = f.read()
keys = set(re.findall(r"^@\w+\{(\w+),", bib, re.M))
texts = []
for fn in ("main.tex", "supplementary.tex"):
    with open(os.path.join(ROOT, fn), encoding="utf-8") as f:
        texts.append(f.read())
text = "\n".join(texts)
cited = set()
for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]+)\}", text):
    for k in m.group(1).split(","):
        cited.add(k.strip())
print(f"Total bib entries: {len(keys)}")
print(f"Cited: {len(cited)}")
print(f"Uncited bib entries: {sorted(keys - cited)}")
print(f"Cited but missing from bib: {sorted(cited - keys)}")

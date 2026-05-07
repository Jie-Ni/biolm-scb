"""Rough body word counter for main.tex (NMI-style: maketitle → \section{Methods} excluded)."""
import re
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Jie/Desktop/Bio-LM_v45_CURRENT/paper/main.tex")
txt = p.read_text(encoding="utf-8")

m1 = txt.find(r"\maketitle")
m2 = txt.find(r"\section{Methods}")
body = txt[m1:m2] if m2 > m1 else txt
# strip line comments
body = re.sub(r"(?m)^\s*%.*$", "", body)
body = re.sub(r"(?<!\\)%.*", "", body)
# kill caption/figure environments contents (NMI counts captions separately, often excluded)
body_no_floats = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", " ", body, flags=re.DOTALL)
# Strip latex
def strip_latex(s):
    s = re.sub(r"\\begin\{[a-zA-Z*]+\}|\\end\{[a-zA-Z*]+\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    s = re.sub(r"\$[^$]*\$", " x ", s)
    s = re.sub(r"[\{\}]", " ", s)
    return s

body_no_floats_clean = strip_latex(body_no_floats)
body_clean = strip_latex(body)
words_no = re.findall(r"\b[a-zA-Z][a-zA-Z\-]*\b", body_no_floats_clean)
words_all = re.findall(r"\b[a-zA-Z][a-zA-Z\-]*\b", body_clean)

n_fig = len(re.findall(r"\\begin\{figure\}", txt))
n_tab = len(re.findall(r"\\begin\{table\}", txt))

print(f"Body words (excl. captions/tables): {len(words_no)}")
print(f"Body words (incl. captions/tables): {len(words_all)}")
print(f"main figures: {n_fig}")
print(f"main tables : {n_tab}")

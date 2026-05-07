"""Count words in the abstract block of main.tex.

Strips LaTeX commands, math, and BibTeX cite tokens before counting.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
text = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")

m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.DOTALL)
assert m, "no abstract block found"
body = m.group(1)

clean = re.sub(r"\$[^$]*\$", " X ", body)
clean = re.sub(r"\\cite[a-z]*\{[^}]*\}", " X ", clean)
clean = re.sub(r"\\(emph|textbf|texttt)\{([^}]*)\}", r" \2 ", clean)
clean = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", clean)
clean = re.sub(r"[~%]", " ", clean)

words = re.findall(r"[A-Za-z][\w\-]*", clean)
print(f"abstract word count (rough): {len(words)}")
print(f"raw char count of LaTeX body: {len(body)}")
print(f"first 200 cleaned chars: {clean.strip()[:200]}")

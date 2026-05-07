"""Count NMI-defined body words for main.tex.

NMI body = abstract->Methods, excludes: Methods, captions (figure/table envs),
references, equations. Includes: Introduction, main sections, Discussion, section
titles. Strips LaTeX commands, refs/cites, comments.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "main.tex"
text = TEX.read_text(encoding="utf-8")
lines = text.split("\n")

# Locate \end{abstract} and \section{Methods}
start = None
end = None
for i, ln in enumerate(lines):
    if start is None and ln.strip().startswith(r"\end{abstract}"):
        start = i + 1
    if end is None and re.match(r"\s*\\section\*?\{Methods\}", ln):
        end = i
        break
assert start and end, f"start={start} end={end}"
body = "\n".join(lines[start:end])
print(f"raw body chars: {len(body)} (lines {start+1}..{end})")

# Strip figure / table / equation environments (and contents)
body = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", " ", body, flags=re.DOTALL)
body = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", " ", body, flags=re.DOTALL)
body = re.sub(r"\\begin\{(equation|align|gather|multline|eqnarray)\*?\}.*?\\end\{\1\*?\}",
              " ", body, flags=re.DOTALL)
# Display math \[ ... \]
body = re.sub(r"\\\[.*?\\\]", " ", body, flags=re.DOTALL)
# Strip comments (not preceded by backslash)
body = re.sub(r"(?<!\\)%.*", " ", body)
# Inline math $...$ -> count as ONE word "MATHTOKEN"
body = re.sub(r"\$[^$]*\$", " MATHTOKEN ", body)

# Remove ref/cite/label commands (with their arg)
body = re.sub(r"\\(cite|citep|citet|ref|label|eqref|pageref|autoref|nameref)\*?\{[^{}]*\}",
              " ", body)
# Section/format commands -> keep argument text
body = re.sub(r"\\(section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^{}]*)\}",
              r" \2 ", body)
body = re.sub(
    r"\\(textbf|textit|emph|texttt|textsc|textsf|textrm|underline|mathrm|mathbf|mathit)\*?\{([^{}]*)\}",
    r" \2 ", body,
)
# Remove all remaining backslash commands (with optional [opt] {arg})
prev = None
while prev != body:
    prev = body
    body = re.sub(r"\\[A-Za-z@]+\*?\s*(\[[^\]]*\])?\s*(\{[^{}]*\})?", " ", body)
# Strip leftover braces
body = re.sub(r"[{}]", " ", body)
# Collapse whitespace
body = re.sub(r"\s+", " ", body).strip()

# Tokenize: word = sequence of letters/digits/hyphen/apostrophe (matching NMI word counters)
words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", body)
print(f"NMI body word count: {len(words)}")
print("first 30:", words[:30])
print("last 30:", words[-30:])
print(f"limit: 3500 — overage: {len(words)-3500} (negative=under)")

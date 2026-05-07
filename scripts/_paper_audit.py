"""Quick audit: word counts + citation count + section breakdown."""
import re
P = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\11_manuscript\arxiv_submission\main.tex"
text = open(P, encoding='utf-8').read()


def clean_latex(s):
    s = re.sub(r'%.*?\n', '\n', s)
    s = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', s)
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    s = re.sub(r'[\\\$\{\}]', ' ', s)
    return s


m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', text, re.DOTALL)
if m:
    a = clean_latex(m.group(1))
    print(f"Abstract: {len(a.split())} words (NMI target ~200, max 250)")

cites = re.findall(r'\\cite[a-z]*\{([^}]+)\}', text)
all_keys = []
for c in cites:
    all_keys.extend([k.strip() for k in c.split(',')])
print(f"Citations: {len(set(all_keys))} unique | {len(all_keys)} total calls")

# Bibliography file
bibp = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\11_manuscript\arxiv_submission\references.bib"
try:
    bib = open(bibp, encoding='utf-8').read()
    entries = re.findall(r'@\w+\s*\{\s*([^,\s]+)', bib)
    print(f"references.bib: {len(entries)} entries total")
except FileNotFoundError:
    pass

print()
print("Section word counts:")
sections = re.split(r'\\section\{', text)
for s in sections[1:]:
    title = s.split('}', 1)[0]
    body = s.split('}', 1)[1] if '}' in s else ''
    # cut at next section
    body = re.split(r'\\section\{', body)[0]
    bc = clean_latex(body)
    wc = len(bc.split())
    print(f"  [{wc:5d} words] {title[:65]}")

# Total word count of main body (excluding methods/abstract)
total_body = re.search(r'\\begin\{document\}(.*?)\\section\{Methods\}', text, re.DOTALL)
if total_body:
    bc = clean_latex(total_body.group(1))
    print(f"\nMain body (intro+results+discussion, excl. methods): {len(bc.split())} words")
methods = re.search(r'\\section\{Methods\}(.*?)\\(section\*?|bibliography)', text, re.DOTALL)
if methods:
    bc = clean_latex(methods.group(1))
    print(f"Methods: {len(bc.split())} words")

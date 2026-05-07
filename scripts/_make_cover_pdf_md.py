"""Replace Unicode glyphs in cover_letter.md with LaTeX math equivalents
so pandoc xelatex (Latin Modern) renders cleanly. Writes a sibling
cover_letter_pdf.md that pandoc compiles."""
import pathlib

SRC = pathlib.Path("C:/Users/Jie/Desktop/Bio-LM_v45_CURRENT/paper/cover_letter.md")
DST = pathlib.Path("C:/Users/Jie/Desktop/Bio-LM_v45_CURRENT/paper/cover_letter_pdf.md")

repls = {
    "≈": "$\\approx$",
    "≥": "$\\geq$",
    "≤": "$\\leq$",
    "∈": "$\\in$",
    "×": "$\\times$",
    "·": "$\\cdot$",
    "±": "$\\pm$",
    "α": "$\\alpha$",
    "β": "$\\beta$",
    "γ": "$\\gamma$",
    "λ": "$\\lambda$",
    "σ": "$\\sigma$",
    "Δ": "$\\Delta$",
    "²": "$^{2}$",
    "³": "$^{3}$",
    "⁴": "$^{4}$",
    "⁵": "$^{5}$",
    "⁶": "$^{6}$",
    "⁷": "$^{7}$",
    "⁸": "$^{8}$",
    "⁹": "$^{9}$",
    "⁰": "$^{0}$",
    "⁻": "$^{-}$",
    "₀": "$_{0}$",
    "₁": "$_{1}$",
    "₂": "$_{2}$",
    "₃": "$_{3}$",
    "₄": "$_{4}$",
    "₅": "$_{5}$",
    "₆": "$_{6}$",
    "₇": "$_{7}$",
    "₈": "$_{8}$",
    "₉": "$_{9}$",
}

text = SRC.read_text(encoding="utf-8")
for k, v in repls.items():
    text = text.replace(k, v)

DST.write_text(text, encoding="utf-8")
print(f"wrote {DST}")

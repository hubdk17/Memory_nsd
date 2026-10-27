"""Word counts against the target journal's limits.

LaTeX markup, tables, math and \\num{} values are stripped so the count reflects
what an editor counts, not what wc(1) sees.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

LIMITS = {"abstract": 200, "main text": 5000}   # Cerebral Cortex, original research


def wc(s: str) -> int:
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s)
    s = re.sub(r"\\begin\{(table|tabular)\}.*?\\end\{\1\}", " ", s, flags=re.S)
    s = re.sub(r"\\(num|SI)\{[^}]*\}(\{[^}]*\})?", " X ", s)
    s = re.sub(r"\$[^$]*\$", " X ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    s = re.sub(r"[{}&\\~]", " ", s)
    return len([w for w in s.split() if any(c.isalnum() for c in w)])


def main() -> int:
    t = (HERE / "main.tex").read_text(encoding="utf-8")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", t, re.S).group(1)

    start = t.index("\\section{Introduction}")
    end = min(i for i in (t.find("\\section*{Acknowledg"),
                          t.find("\\section*{Data"),
                          t.find("\\section*{Funding}")) if i > 0)
    body = t[start:end]
    figs = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", body, re.S)
    body_nofig = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", body, flags=re.S)

    counts = {"abstract": wc(abstract), "main text": wc(body_nofig)}
    over = []
    for k, n in counts.items():
        lim = LIMITS[k]
        flag = "OK" if n <= lim else f"OVER by {n - lim}"
        if n > lim:
            over.append(k)
        print(f"  {k:<12s} {n:>5}  / {lim:<5} {flag}")
    print(f"  {'captions':<12s} {wc(' '.join(figs)):>5}    (excluded from the limit)")

    kw = re.search(r"\\textbf\{Keywords:\}(.*?)\n\n", t, re.S)
    keys = [k.strip().rstrip(".") for k in kw.group(1).replace("\n", " ").split(";")]
    alpha = keys == sorted(keys, key=str.lower)
    print(f"  {'keywords':<12s} {len(keys):>5}  / 5     "
          f"{'OK' if len(keys) <= 5 else 'TOO MANY'}, "
          f"{'alphabetical' if alpha else 'NOT alphabetical'}")
    if len(keys) > 5 or not alpha:
        over.append("keywords")

    print("\n" + ("[OK] within journal limits" if not over
                  else f"[FAIL] over limit: {', '.join(over)}"))
    return 0 if not over else 1


if __name__ == "__main__":
    sys.exit(main())

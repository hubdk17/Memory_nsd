"""Structural and numerical lint for the manuscript.

No LaTeX toolchain is installed on this machine, so the document cannot be compiled
here. These checks catch what would otherwise only surface at compile time, plus the
thing a compiler would never catch: a number in a table that no longer matches the
data it came from.

Run before handing main.tex to Overleaf.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Cerebral Cortex (OUP), original research article. The list is ordered, and the
# check below enforces the order as well as the presence of each section.
REQUIRED_SECTIONS = [
    r"\section{Introduction}",
    r"\section{Materials and Methods}",
    r"\section{Results}",
    r"\section{Discussion}",
    r"\section*{Acknowledgments}",
    r"\section*{Funding}",
    r"\section*{Conflict of Interest}",
    r"\section*{Data Availability}",
    r"\section*{Author Contributions}",
]


def norm(s: str) -> str:
    """Collapse runs of whitespace so alignment padding does not matter."""
    return " ".join(s.split())


def main() -> int:
    tex = (HERE / "main.tex").read_text(encoding="utf-8")
    bib = (HERE / "references.bib").read_text(encoding="utf-8")
    problems = []

    # ------------------------------------------------------------ structure
    print("=== environment balance ===")
    for env in ("document", "abstract", "table", "tabular", "equation", "figure"):
        o = tex.count("\\begin{%s}" % env)
        c = tex.count("\\end{%s}" % env)
        ok = o == c
        problems += [] if ok else [f"{env}: {o} begin vs {c} end"]
        print(f"  {env:10s} {o} / {c}  {'OK' if ok else 'MISMATCH'}")
    ob, cb = tex.count("{"), tex.count("}")
    problems += [] if ob == cb else [f"braces {ob} vs {cb}"]
    print(f"  {'braces':10s} {ob} / {cb}  {'OK' if ob == cb else 'MISMATCH'}")

    print("\n=== required sections and order (Cerebral Cortex) ===")
    pos = -1
    for sec in REQUIRED_SECTIONS:
        at = tex.find(sec)
        if at < 0:
            problems.append(f"missing section: {sec}")
            print(f"  MISSING {sec}")
            continue
        if at < pos:
            problems.append(f"section out of journal order: {sec}")
        print(f"  {'OK ' if at > pos else 'ORDER'} {sec}")
        pos = max(pos, at)

    # The journal's structure has no top-level Conclusion; it belongs in Discussion.
    if "\\section{Conclusion}" in tex:
        problems.append("top-level \\section{Conclusion} is not in the journal's "
                        "section list -- demote it to a Discussion subsection")

    # ------------------------------------------------------------ citations
    print("\n=== citations ===")
    cites = set()
    for grp in re.findall(r"\\cite[tp]?\{([^}]*)\}", tex):
        cites |= {c.strip() for c in grp.split(",") if c.strip()}
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing = sorted(cites - keys)
    unused = sorted(keys - cites)
    problems += [f"cited but absent from .bib: {missing}"] if missing else []
    print(f"  cited {len(cites)}, defined {len(keys)}")
    print(f"  missing from .bib : {missing or 'none'}")
    print(f"  defined but unused: {unused or 'none'}")

    # ------------------------------------------------------------ labels
    print("\n=== labels and refs ===")
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
    dangling = sorted(refs - labels)
    unref = sorted(labels - refs)
    problems += [f"dangling \\ref: {dangling}"] if dangling else []
    problems += [f"label never referenced: {unref}"] if unref else []
    print(f"  labels {len(labels)}, refs {len(refs)}")
    print(f"  dangling  : {dangling or 'none'}")
    print(f"  unreferenced: {unref or 'none'}")

    # ------------------------------------------------------------ figures
    print("\n=== figures ===")
    # \graphicspath may list several directories: {{figures/}{./}}
    gp = re.search(r"\\graphicspath\{(.*?)\}\s*$", tex, re.M)
    roots = re.findall(r"\{([^{}]*)\}", gp.group(1) + "}") if gp else [""]
    bases = [HERE / r for r in roots] or [HERE]
    print(f"  search path: {[str(b.relative_to(HERE)) or '.' for b in bases]}")
    for inc in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex):
        hit = next((b / inc for b in bases if (b / inc).is_file()), None)
        problems += [] if hit else [
            f"figure file not found on the graphics path: {inc}"]
        size = f"{hit.stat().st_size / 1024:.0f} kB" if hit else "MISSING"
        where = f"  <- {hit.parent.name}/" if hit else ""
        print(f"  {'OK ' if hit else 'FAIL'} {inc:28s} {size}{where}")

    n_fig = tex.count("\\begin{figure}")
    caps = len(re.findall(r"\\begin\{figure\}.*?\\caption", tex, re.S))
    print(f"  {n_fig} figure environments, {caps} with captions")
    if caps != n_fig:
        problems.append(f"{n_fig - caps} figure(s) without a caption")

    # ------------------------------------------------------------ table shape
    print("\n=== table shape ===")
    for i, m in enumerate(re.finditer(
            r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", tex, re.S), 1):
        spec, body = m.group(1), m.group(2)
        ncol = sum(spec.count(c) for c in "lcr")
        # Rows spanned by \multicolumn or \cmidrule are structural, not data rows.
        rows = [r for r in body.split("\\\\")
                if "&" in r and "multicolumn" not in r and "cmidrule" not in r]
        counts = sorted({r.count("&") + 1 for r in rows})
        ok = counts == [ncol]
        problems += [] if ok else [f"table {i}: spec {ncol} cols, rows have {counts}"]
        print(f"  table {i}: {ncol} cols, {len(rows)} data rows, "
              f"cells/row {counts}  {'OK' if ok else 'RAGGED'}")

    # ------------------------------------------------------------ table numbers
    print("\n=== table numbers vs source data ===")
    gen = subprocess.run([sys.executable, str(HERE / "make_tables.py")],
                         capture_output=True, text=True)
    if gen.returncode != 0:
        problems.append(f"make_tables.py failed: {gen.stderr.strip()[:300]}")
        print("  make_tables.py FAILED -- cannot verify table contents")
    else:
        tex_rows = {norm(r) for r in tex.split("\n") if "\\\\" in r and "&" in r}
        checked = mismatch = 0
        for line in gen.stdout.splitlines():
            line = line.strip()
            if not line.endswith("\\\\") or "&" not in line:
                continue
            checked += 1
            if norm(line) not in tex_rows:
                mismatch += 1
                problems.append(f"table row not found in main.tex: {norm(line)[:90]}")
        print(f"  {checked} generated rows checked, {mismatch} not matching main.tex")

    # ------------------------------------------------------------ numbers
    print("\n=== inline numbers ===")
    bad = [n for n in re.findall(r"\\num\{([^}]*)\}", tex)
           if not re.fullmatch(r"-?\d*\.?\d+(e-?\d+)?", n.strip())]
    problems += [f"unparseable \\num: {bad}"] if bad else []
    n_num = len(re.findall(r"\\num\{", tex))
    problems += [] if n_num else ["no \\num values found -- did the tables move?"]
    print(f"  num values: {n_num}, unparseable: {bad or 'none'}")

    print("\n" + ("[OK] no problems found" if not problems
                  else f"[FAIL] {len(problems)} problem(s):"))
    for p in problems:
        print(f"   - {p}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

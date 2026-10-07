import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata

import pymupdf


ROOT = Path(__file__).resolve().parent
STEM = "poster_IC_2026"


def normalize_text(text):
    normalized = unicodedata.normalize("NFKD", text)
    return " ".join(normalized.encode("ascii", "ignore").decode().split())


def compile_poster():
    environment = os.environ.copy()
    tinytex = Path(environment.get("APPDATA", "")) / "TinyTeX" / "bin" / "windows"
    environment["PATH"] = str(tinytex) + os.pathsep + environment.get("PATH", "")
    compiler = shutil.which("latexmk", path=environment["PATH"])
    if compiler is None:
        raise SystemExit("latexmk was not found. Install TeX or set PATH.")
    result = subprocess.run(
        [compiler, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         "-file-line-error", f"{STEM}.tex"],
        cwd=ROOT, env=environment, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode:
        print("\n".join((result.stdout + result.stderr).splitlines()[-65:]))
        raise SystemExit(result.returncode)
    print("PASS: latexmk completed.")


def check_poster():
    failures = []
    log_text = (ROOT / f"{STEM}.log").read_text(encoding="utf-8", errors="replace")
    issues = re.findall(r"^.*(?:Overfull|Underfull|Missing character|LaTeX Warning|^!).*$",
                        log_text, flags=re.MULTILINE)
    failures.extend(issues)
    capital_heights = dict(re.findall(
        r"POSTER-CHECK (Title|Unit) capital-height=([0-9.]+)pt", log_text))
    for name in ("Title", "Unit"):
        height_mm = float(capital_heights.get(name, "0")) * 25.4 / 72.27
        if height_mm < 15:
            failures.append(f"{name} capital height: {height_mm:.2f} mm, below 15 mm.")
        else:
            print(f"PASS: {name} capital height = {height_mm:.2f} mm.")

    with pymupdf.open(ROOT / f"{STEM}.pdf") as document:
        if len(document) != 1:
            raise SystemExit(f"Expected one page, found {len(document)}.")
        page = document[0]
        width_mm = page.rect.width * 25.4 / 72
        height_mm = page.rect.height * 25.4 / 72
        if abs(width_mm - 841) > 0.1 or abs(height_mm - 1189) > 0.1:
            failures.append(f"Unexpected page size: {width_mm:.2f} x {height_mm:.2f} mm.")
        else:
            print(f"PASS: one A0 page, {width_mm:.2f} x {height_mm:.2f} mm.")

        text = normalize_text(page.get_text())
        required_text = (
            "IRIDOLOGIA SOB LENTES CIENTIFICAS:",
            "AVALIACAO CRITICA COM APRENDIZADO DE MAQUINA",
            "CENTRO DE MATEMATICA, COMPUTACAO E COGNICAO",
            "UNIVERSIDADE FEDERAL DO ABC", "Vinicius Sedrim",
            "Vladimir Emiliano Moreira Rocha", "INTRODUCAO", "OBJETIVOS",
            "MATERIAIS E METODOS", "RESULTADOS E DISCUSSAO", "CONCLUSOES",
            "AGRADECIMENTOS", "REFERENCIAS BIBLIOGRAFICAS", "CNPq",
            "Carlos Andres Reyes", "CC BY 2.0", "AUDITORIA DOS DADOS",
            "Resultados preliminares da avaliacao interna", "83,67%",
            "92,36%", "91,74%", "95%",
        )
        for phrase in required_text:
            if phrase not in text:
                failures.append(f"Missing required text: {phrase}")

        used_fonts = {trace["font"] for trace in page.get_texttrace() if trace["chars"]}
        embedded_fonts = set()
        for font in page.get_fonts(full=True):
            name = re.sub(r"^[A-Z]{6}\+", "", font[3])
            if document.extract_font(font[0])[3]:
                embedded_fonts.add(name)
            elif name not in used_fonts:
                print(f"INFO: unused font declaration, no printed glyphs: {name}.")
        for name in sorted(used_fonts - embedded_fonts):
            failures.append(f"Printed font is not embedded: {name}")
        print("CHECKED printed fonts: " + ", ".join(sorted(used_fonts)))

        words = page.get_text("words")
        for word in words:
            if not page.rect.contains(pymupdf.Rect(word[:4])):
                failures.append(f"Text outside page: {normalize_text(word[4])}")

        overlaps = []
        for index, word in enumerate(words):
            first = pymupdf.Rect(word[:4])
            first.y0 += first.height * 0.16
            first.y1 -= first.height * 0.16
            first.x0 += 0.5
            first.x1 -= 0.5
            for other in words[index + 1:]:
                if word[5:7] == other[5:7]:
                    continue
                second = pymupdf.Rect(other[:4])
                second.y0 += second.height * 0.16
                second.y1 -= second.height * 0.16
                second.x0 += 0.5
                second.x1 -= 0.5
                intersection = first & second
                if not intersection.is_empty and intersection.get_area() > 4:
                    overlaps.append(f"{normalize_text(word[4])} / {normalize_text(other[4])}")
        if overlaps:
            failures.append("Possible text overlaps: " + "; ".join(overlaps))

        preview = ROOT / f"{STEM}_previa.png"
        page.get_pixmap(dpi=60, alpha=False).save(preview)
        print(f"PREVIEW: {preview}")
        if not overlaps:
            print("PASS: no significant overlaps between separate text lines.")

    if failures:
        for failure in failures:
            print("FAIL: " + failure)
        raise SystemExit(1)
    print("PASS: required text, embedded fonts, page bounds and LaTeX log.")
    print("Scientific note: IC1 numbers remain preliminary and require author review.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and render the A0 IC1 poster.")
    parser.add_argument("--compile", action="store_true", help="Run latexmk before validation.")
    arguments = parser.parse_args()
    if arguments.compile:
        compile_poster()
    check_poster()

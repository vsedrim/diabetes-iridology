"""
Demonstração: realce de vascularização + validação de pupila
============================================================
Script de aplicação dos dois itens pedidos pelo orientador:

1. Algoritmos consolidados de realce de vascularização (além de brilho/contraste):
   - Frangi (vesselness Hessiano multiescala)
   - Banco de filtros de Gabor orientados
   - Top-hat / black-hat morfológico multiescala

2. Validação da pupila: classificação pela geometria (diâmetro da pupila e
   espessura da íris), via src.segmentation.PupilValidator.

Roda sobre as imagens reais de íris em docs/imagens e escreve artefatos
(painéis comparativos, imagem anotada e tabelas de métricas) em results/.

Uso:
    python experiments/demo_vascularization_pupil.py
"""

import os
import sys
import csv

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Permite rodar como script a partir da raiz do repositório
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.preprocessing import create_vascularization_enhancer
from src.segmentation import create_pupil_validator
from src.config import PhotometricTransform, PupilValidationConfig

IMAGES_DIR = os.path.join(REPO_ROOT, "docs", "imagens")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
VASC_DIR = os.path.join(RESULTS_DIR, "vascularization")
PUPIL_DIR = os.path.join(RESULTS_DIR, "pupil")

# Imagens de íris reais disponíveis no repositório
VASC_IMAGES = ["iris_closeup.jpg", "iris_petr_novak.jpg"]
PUPIL_IMAGE = "iris_petr_novak.jpg"  # olho real -> geometria fisiologicamente válida

MAX_DIM = 768  # redimensiona imagens grandes (desempenho do Frangi e visualização)


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def load_image(name: str):
    path = os.path.join(IMAGES_DIR, name)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {path}")
    return img


def resize_max(img, max_dim=MAX_DIM):
    h, w = img.shape[:2]
    scale = max_dim / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def detect_pupil_iris(bgr):
    """
    Detector pragmático de pupila e íris para a demonstração.

    Pupila: maior região escura razoavelmente circular próxima ao centro.
    Íris: círculo de Hough mais concêntrico com a pupila; se não houver (íris
    preenche o quadro), estima o raio pelo campo visível.

    Retorna dict com centros, raios e flags de estimativa (pupila e íris).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    g = cv2.medianBlur(gray, 7)

    # --- Pupila: limiariza a parte mais escura ---
    thr_val = float(np.percentile(g, 3)) + 20.0
    _, dark = cv2.threshold(g, thr_val, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1.0
    img_center = np.array([w / 2.0, h / 2.0])
    for c in cnts:
        area = cv2.contourArea(c)
        if area < (0.0015 * h * w):  # descarta blobs minúsculos
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r <= 0:
            continue
        circularity = area / (np.pi * r * r)          # 1.0 = círculo perfeito
        centrality = 1.0 - (np.linalg.norm(np.array([cx, cy]) - img_center)
                            / np.linalg.norm(img_center))
        score = circularity + 0.5 * centrality
        if circularity > 0.55 and score > best_score:
            best, best_score = (cx, cy, r), score

    pupil_estimated = best is None
    if best is None:
        # fallback: ponto mais escuro como centro da pupila
        min_loc = cv2.minMaxLoc(g)[2]
        best = (float(min_loc[0]), float(min_loc[1]), 0.08 * min(h, w))

    pcx, pcy, pr = best

    # --- Íris: Hough circles concêntrico com a pupila ---
    iris_estimated = False
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h,
        param1=100, param2=40,
        minRadius=int(pr * 1.6), maxRadius=int(pr * 6)
    )
    iris = None
    if circles is not None:
        circles = np.round(circles[0, :]).astype(float)
        best_d = 1e18
        for (x, y, r) in circles:
            d = np.hypot(x - pcx, y - pcy)
            if d < 0.9 * pr and d < best_d:  # concêntrico com a pupila
                iris, best_d = (x, y, r), d
    if iris is None:
        # íris preenche o campo (ex.: close-up sem esclera) -> estima o raio
        iris = (pcx, pcy, 0.48 * min(h, w))
        iris_estimated = True

    icx, icy, ir = iris
    return {
        "pupil_center": (float(pcx), float(pcy)),
        "pupil_radius": float(pr),
        "iris_center": (float(icx), float(icy)),
        "iris_radius": float(ir),
        "iris_estimated": iris_estimated,
        "pupil_estimated": pupil_estimated,
    }


def annulus_mask(shape, det):
    """Máscara do anel da íris (entre pupila e íris) para medir no estroma."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(det["iris_center"][0]), int(det["iris_center"][1])),
               int(det["iris_radius"]), 255, -1)
    cv2.circle(mask, (int(det["pupil_center"][0]), int(det["pupil_center"][1])),
               int(det["pupil_radius"]), 0, -1)
    return mask


def iris_disk_mask(shape, det):
    """Máscara do disco completo da íris (para focar a exibição dos realces)."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(det["iris_center"][0]), int(det["iris_center"][1])),
               int(det["iris_radius"]), 255, -1)
    return mask


# ---------------------------------------------------------------------------
# Parte 1 — Realce de vascularização
# ---------------------------------------------------------------------------
def run_vascularization():
    os.makedirs(VASC_DIR, exist_ok=True)
    enhancer = create_vascularization_enhancer()

    # Baselines atuais (brilho/contraste) vs algoritmos consolidados
    def brightness(g):
        return cv2.convertScaleAbs(g, alpha=1.0, beta=40)   # brilho +40

    def contrast(g):
        return cv2.convertScaleAbs(g, alpha=1.8, beta=0)    # contraste x1.8

    # (rótulo, função(green,img)->imagem, é_baseline?)
    methods = [
        ("Original (verde)", lambda g, im: g, False),
        ("Brilho +40", lambda g, im: brightness(g), True),
        ("Contraste x1.8", lambda g, im: contrast(g), True),
        ("CLAHE", lambda g, im: enhancer._clahe.apply(g), True),
        ("Frangi", lambda g, im: enhancer.enhance(im, PhotometricTransform.FRANGI), False),
        ("Gabor", lambda g, im: enhancer.enhance(im, PhotometricTransform.GABOR), False),
        ("Black-hat", lambda g, im: enhancer.enhance(im, PhotometricTransform.BLACKHAT), False),
    ]
    panel_labels = {"Original (verde)", "Brilho +40", "Contraste x1.8",
                    "Frangi", "Gabor", "Black-hat"}

    rows = []
    print("\n=== Realce de vascularização ===")
    for name in VASC_IMAGES:
        img = resize_max(load_image(name))
        det = detect_pupil_iris(img)
        mask = annulus_mask(img.shape, det)
        disk = iris_disk_mask(img.shape, det)

        green = img[:, :, 1]
        panels = []
        computed = {}  # label -> (metrics, is_baseline)
        for label, fn, is_base in methods:
            out = fn(green, img)
            metrics = enhancer.quantify(out, mask=mask)
            computed[label] = (metrics, is_base)
            if label in panel_labels:
                # realces focados no disco da íris; baseline/original inteiros
                is_algo = label in ("Frangi", "Gabor", "Black-hat")
                display = cv2.bitwise_and(out, out, mask=disk) if is_algo else out
                panels.append((label, display))

        # Melhor baseline (ferramentas atuais do professor) como referência
        base_cnr = max(m["structure_contrast"] for m, b in computed.values() if b)
        base_ridge = max(m["ridge_energy"] for m, b in computed.values() if b)

        for label, fn, is_base in methods:
            metrics = computed[label][0]
            cnr_gain = metrics["structure_contrast"] / base_cnr if base_cnr else float("nan")
            ridge_gain = metrics["ridge_energy"] / base_ridge if base_ridge else float("nan")
            rows.append({
                "image": name,
                "method": label,
                "is_baseline": "sim" if is_base else "não",
                "structure_contrast_CNR": round(metrics["structure_contrast"], 3),
                "ridge_energy": round(metrics["ridge_energy"], 4),
                "entropy": round(metrics["entropy"], 3),
                "CNR_gain_vs_baseline": round(cnr_gain, 2),
                "ridge_gain_vs_baseline": round(ridge_gain, 2),
            })
            tag = "[baseline]" if is_base else ""
            print(f"  {name:20s} {label:17s} {tag:11s} "
                  f"CNR={metrics['structure_contrast']:5.2f} "
                  f"ridge={metrics['ridge_energy']:.3f} "
                  f"CNRx={cnr_gain:.2f}  ridgex={ridge_gain:.2f}")

        # Painel comparativo
        fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 4.3))
        for ax, (label, out) in zip(axes, panels):
            ax.imshow(out, cmap="gray")
            ax.set_title(label, fontsize=11)
            ax.axis("off")
        caption = (f"{name}  —  íris {'estimada' if det['iris_estimated'] else 'detectada'} "
                   f"(métricas medidas no anel da íris)")
        fig.suptitle(caption, fontsize=12)
        fig.tight_layout()
        out_path = os.path.join(VASC_DIR, f"{os.path.splitext(name)[0]}_panel.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> painel salvo em {os.path.relpath(out_path, REPO_ROOT)}")

    _write_table(rows, os.path.join(VASC_DIR, "metrics.csv"),
                 os.path.join(VASC_DIR, "metrics.md"),
                 title="Visibilidade da vascularização: baselines (brilho/contraste) vs "
                       "algoritmos consolidados (medido no anel da íris)")
    return rows


# ---------------------------------------------------------------------------
# Legenda sobreposta legível (faixa translúcida + texto nítido)
# ---------------------------------------------------------------------------
def _caption_font(size):
    """Fonte TrueType com acentuação (DejaVu Sans, distribuída com o matplotlib)."""
    from matplotlib import font_manager
    from PIL import ImageFont
    path = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold"),
        fallback_to_default=True)
    return ImageFont.truetype(path, size)


def _draw_caption(img, lines, highlight_index=None, highlight_color=(0, 200, 0),
                  font_size=19, origin=(10, 10), pad=10, line_gap=7):
    """Escreve uma legenda sobre a imagem sem o artefato de texto "fantasma".

    Em vez de empilhar um contorno preto grosso sob um texto fino (que deixa
    bordas pretas desalinhadas em volta de cada letra), desenha-se uma faixa
    escura semitransparente dimensionada ao texto e uma única passada de texto
    nítido por cima. O texto é rasterizado com uma fonte TrueType, o que
    permite acentos (as fontes Hershey do cv2.putText só cobrem ASCII).
    Cores em BGR, como no restante do OpenCV.
    """
    from PIL import Image, ImageDraw

    font = _caption_font(font_size)
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_w = max(measure.textlength(t, font=font) for t in lines)

    x0, y0 = origin
    x1 = min(int(x0 + text_w + 2 * pad), img.shape[1] - 1)
    y1 = min(int(y0 + len(lines) * line_h + (len(lines) - 1) * line_gap + 2 * pad),
             img.shape[0] - 1)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)
    cv2.rectangle(img, (x0, y0), (x1, y1), (120, 120, 120), 1, cv2.LINE_AA)

    canvas = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    y = y0 + pad
    for i, text in enumerate(lines):
        b, g, r = highlight_color if i == highlight_index else (245, 245, 245)
        draw.text((x0 + pad, y), text, font=font, fill=(r, g, b))
        y += line_h + line_gap
    img[:] = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    return img


# ---------------------------------------------------------------------------
# Parte 2 — Validação de pupila
# ---------------------------------------------------------------------------
def run_pupil_validation():
    os.makedirs(PUPIL_DIR, exist_ok=True)
    validator = create_pupil_validator()

    print("\n=== Validação de pupila (imagem real) ===")
    img = resize_max(load_image(PUPIL_IMAGE))
    det = detect_pupil_iris(img)
    res = validator.validate(
        pupil_radius=det["pupil_radius"],
        iris_radius=det["iris_radius"],
        pupil_center=det["pupil_center"],
        iris_center=det["iris_center"],
    )

    print(f"  imagem: {PUPIL_IMAGE}")
    for k, v in res.to_dict().items():
        print(f"    {k}: {v}")

    # Anotação
    vis = img.copy()
    ic = (int(det["iris_center"][0]), int(det["iris_center"][1]))
    pc = (int(det["pupil_center"][0]), int(det["pupil_center"][1]))
    cv2.circle(vis, ic, int(det["iris_radius"]), (255, 128, 0), 2)
    cv2.circle(vis, pc, int(det["pupil_radius"]), (0, 255, 0), 2)
    cv2.circle(vis, pc, 3, (0, 255, 0), -1)
    cv2.circle(vis, ic, 3, (255, 128, 0), -1)

    color = {"valid": (0, 200, 0), "borderline": (0, 200, 200),
             "invalid": (0, 0, 255)}[res.quality_label]
    def br(x):
        return f"{x:.2f}".replace(".", ",")

    lines = [
        f"pupila: D = {res.pupil_diameter:.0f} px    íris: D = {res.iris_diameter:.0f} px",
        f"espessura da íris = {res.iris_thickness:.0f} px    razão P/I = {br(res.pupil_iris_ratio)}",
        f"classificação: {res.quality_label} (pontuação = {br(res.quality_score)})",
    ]
    _draw_caption(vis, lines, highlight_index=2, highlight_color=color)

    out_path = os.path.join(PUPIL_DIR, f"{os.path.splitext(PUPIL_IMAGE)[0]}_validation.png")
    cv2.imwrite(out_path, vis)
    print(f"  -> anotação salva em {os.path.relpath(out_path, REPO_ROOT)}")

    # Casos sintéticos para mostrar as fronteiras do classificador
    print("\n=== Validação de pupila (casos sintéticos de teste) ===")
    cases = [
        ("normal",            40.0, 100.0, 0.0),
        ("miose (constrita)", 18.0, 100.0, 0.0),
        ("midriase (dilatada)", 78.0, 100.0, 0.0),
        ("excentrica",        40.0, 100.0, 40.0),
        ("anel fino",         88.0, 100.0, 0.0),
    ]
    synth_rows = []
    for label, pr, ir, off in cases:
        r = validator.validate(pr, ir, pupil_center=(off, 0.0), iris_center=(0.0, 0.0))
        synth_rows.append({
            "caso": label,
            "pupil_iris_ratio": round(r.pupil_iris_ratio, 3),
            "thickness_ratio": round(r.thickness_ratio, 3),
            "concentricity_offset": round(r.concentricity_offset, 3),
            "quality_label": r.quality_label,
            "quality_score": round(r.quality_score, 3),
        })
        print(f"  {label:20s} razao={r.pupil_iris_ratio:.2f} "
              f"espessura={r.thickness_ratio:.2f} offset={r.concentricity_offset:.2f} "
              f"-> {r.quality_label.upper()} ({r.quality_score:.2f})")

    _write_table(synth_rows, os.path.join(PUPIL_DIR, "synthetic_cases.csv"),
                 os.path.join(PUPIL_DIR, "synthetic_cases.md"),
                 title="Classificador de geometria pupila-íris (casos sintéticos)")
    return res


# ---------------------------------------------------------------------------
# Escrita de tabelas (CSV + Markdown)
# ---------------------------------------------------------------------------
def _write_table(rows, csv_path, md_path, title=""):
    if not rows:
        return
    headers = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    with open(md_path, "w", encoding="utf-8") as f:
        if title:
            f.write(f"### {title}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")
    print(f"  -> tabela salva em {os.path.relpath(csv_path, REPO_ROOT)} "
          f"e {os.path.relpath(md_path, REPO_ROOT)}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_vascularization()
    run_pupil_validation()
    print("\nConcluído. Artefatos em results/.")


if __name__ == "__main__":
    main()

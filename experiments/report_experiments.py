"""
Experimentos complementares do relatório de PGC II
==================================================
Reúne as análises usadas no relatório docs/PGC_2, todas executadas sobre o
código do pipeline (src/) e sobre as mesmas imagens públicas da demonstração:

  E0. Painéis comparativos e detalhe ampliado do realce
  E1. Análise por setores angulares do anel da íris (CNR, Wilcoxon, bootstrap)
  E2. Sensibilidade da métrica (limiar de Otsu global vs. restrito ao anel)
  E3. Análise multiescala do filtro de Frangi
  E4. Sensibilidade a parâmetros (Frangi, Gabor, morfologia)
  E5. Robustez a perturbações fotométricas
  E6. Fantoma sintético com verdade de referência (AUC e CNR)
  E7. Custo computacional
  E8. Normalização polar (rubber sheet) do mapa de Frangi e perfil radial
  E9. Validação da pupila: curva de pontuação, mapa de decisão e
      estabilidade da detecção
  E10. Escolha do canal de cor (R, G, B, luminância)
  M.  Figuras de método (banco de Gabor, CLAHE, anel e setores)

Saídas:
  docs/PGC_2/figuras/  figuras do relatório (PNG)
  docs/PGC_2/tabelas/  tabelas LaTeX geradas (incluídas com \\input)
  docs/PGC_2/dados/    séries numéricas para os gráficos pgfplots
  results/report/      CSVs completos e resumo.json

Uso:
    python experiments/report_experiments.py
"""

import csv
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import replace

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Patch
from matplotlib.ticker import ScalarFormatter
import scipy
from scipy import stats
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
import skimage
from skimage.filters import frangi, threshold_otsu
from skimage.util import img_as_float

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import demo_vascularization_pupil as demo  # noqa: E402
from src.preprocessing import VascularizationEnhancer  # noqa: E402
from src.segmentation import create_pupil_validator  # noqa: E402
from src.config import PreprocessingConfig  # noqa: E402

DOC_DIR = os.path.join(REPO_ROOT, "docs", "PGC_2")
FIG_DIR = os.path.join(DOC_DIR, "figuras")
TAB_DIR = os.path.join(DOC_DIR, "tabelas")
DAT_DIR = os.path.join(DOC_DIR, "dados")
OUT_DIR = os.path.join(REPO_ROOT, "results", "report")

IMAGES = {"novak": "iris_petr_novak.jpg", "closeup": "iris_closeup.jpg"}
IMAGE_TEX = {"novak": "Olho completo", "closeup": "Macro da íris"}

BASELINES = ("Brilho +40", "Contraste ×1,8", "CLAHE")
ALGOS = ("Frangi", "Gabor", "Black-hat")
METHOD_TEX = {
    "Original": "Original (verde)",
    "Brilho +40": "Brilho $+40$",
    "Contraste ×1,8": "Contraste $\\times1{,}8$",
    "CLAHE": "CLAHE",
    "Frangi": "Frangi",
    "Gabor": "Gabor",
    "Black-hat": "\\textit{Black-hat}",
}
COLORS = {
    "Original": "#9e9e9e", "Brilho +40": "#b0b0b0", "Contraste ×1,8": "#808080",
    "CLAHE": "#5f5f5f", "Frangi": "#1565c0", "Gabor": "#2e7d32", "Black-hat": "#ef6c00",
}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

SUMMARY = {}


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def ensure_dirs():
    for d in (FIG_DIR, TAB_DIR, DAT_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)


def br(x, nd=2):
    """Número com vírgula decimal para o texto em português."""
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{abs(x):.{nd}f}".replace(".", ",")
    return f"$-${s}" if x < 0 else s


class CommaFormatter(ScalarFormatter):
    """Formatador padrão do matplotlib com vírgula decimal."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def _decimal_comma(fig):
    for ax in fig.axes:
        if ax.name == "polar":
            continue
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale == "linear" and type(axis.get_major_formatter()) is ScalarFormatter:
                axis.set_major_formatter(CommaFormatter())


def save_fig(fig, name):
    _decimal_comma(fig)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura: docs/PGC_2/figuras/{name}")


def write_tex(name, body):
    with open(os.path.join(TAB_DIR, name), "w", encoding="utf-8") as f:
        f.write("% Gerado por experiments/report_experiments.py (não editar à mão).\n")
        f.write(body)
    print(f"  tabela: docs/PGC_2/tabelas/{name}")


def write_csv(name, rows):
    if not rows:
        return
    with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_dat(name, header, rows):
    with open(os.path.join(DAT_DIR, name), "w", encoding="utf-8") as f:
        f.write(" ".join(header) + "\n")
        for r in rows:
            f.write(" ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r) + "\n")
    print(f"  dados: docs/PGC_2/dados/{name}")


def method_table(enh):
    return [
        ("Original", lambda g, im: g),
        ("Brilho +40", lambda g, im: cv2.convertScaleAbs(g, alpha=1.0, beta=40)),
        ("Contraste ×1,8", lambda g, im: cv2.convertScaleAbs(g, alpha=1.8, beta=0)),
        ("CLAHE", lambda g, im: enh._clahe.apply(g)),
        ("Frangi", lambda g, im: enh.enhance_frangi(im)),
        ("Gabor", lambda g, im: enh.enhance_gabor(im)),
        ("Black-hat", lambda g, im: enh.enhance_blackhat(im)),
    ]


def load_case(key, max_dim=demo.MAX_DIM):
    img = demo.resize_max(demo.load_image(IMAGES[key]), max_dim)
    det = demo.detect_pupil_iris(img)
    ann = demo.annulus_mask(img.shape, det)
    disk = demo.iris_disk_mask(img.shape, det)
    return img, det, ann, disk


def run_methods(enh, img):
    green = img[:, :, 1]
    return {name: fn(green, img) for name, fn in method_table(enh)}


def cnr(enh, out, mask):
    return enh.quantify(out, mask=mask)["structure_contrast"]


def cnr_or_nan(out, mask):
    """CNR com o mesmo protocolo de quantify(); NaN quando estrutura ou fundo ficam vazios."""
    thr, _ = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vals = out[mask > 0].astype(np.float64)
    fg, bg = vals[vals > thr], vals[vals <= thr]
    if fg.size == 0 or bg.size == 0:
        return float("nan")
    return float((fg.mean() - bg.mean()) / (bg.std() + 1e-6))


def cnr_otsu_in_mask(out, mask):
    vals = out[mask > 0].astype(np.float64)
    thr = threshold_otsu(vals)
    fg, bg = vals[vals > thr], vals[vals <= thr]
    return float((fg.mean() - bg.mean()) / (bg.std() + 1e-6))


def structure_mask(out, mask):
    thr, _ = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (out > thr) & (mask > 0)


# ---------------------------------------------------------------------------
# E0 — Painéis e métricas principais
# ---------------------------------------------------------------------------
def e0_panels_and_main_table(enh, cases):
    print("\n[E0] Painéis comparativos e tabela principal")
    rows, tex_rows = [], []
    for key, (img, det, ann, disk) in cases.items():
        outs = run_methods(enh, img)
        metrics = {m: enh.quantify(o, mask=ann) for m, o in outs.items()}
        base_cnr = max(metrics[b]["structure_contrast"] for b in BASELINES)
        base_ridge = max(metrics[b]["ridge_energy"] for b in BASELINES)
        first = True
        for m in ("Original",) + BASELINES + ALGOS:
            mm = metrics[m]
            g_c = mm["structure_contrast"] / base_cnr
            g_r = mm["ridge_energy"] / base_ridge
            rows.append({"imagem": key, "metodo": m, "CNR": mm["structure_contrast"],
                         "ridge": mm["ridge_energy"], "entropia": mm["entropy"],
                         "densidade": mm["structure_density"], "CNR_x_base": g_c,
                         "ridge_x_base": g_r})
            best = m == "Frangi"
            cells = [br(mm["structure_contrast"]), br(mm["ridge_energy"], 3),
                     br(mm["entropy"]), br(g_c), br(g_r)]
            if best:
                cells = [f"\\textbf{{{c}}}" for c in cells]
            name = METHOD_TEX[m] + (" (base)" if m in BASELINES else "")
            if best:
                name = f"\\textbf{{{name}}}"
            label = f"\\multirow{{7}}{{*}}{{{IMAGE_TEX[key]}}}" if first else ""
            tex_rows.append(f"{label} & {name} & " + " & ".join(cells) + " \\\\")
            first = False
        tex_rows.append("\\midrule")
        SUMMARY.setdefault("principal", {})[key] = {
            m: {"CNR": metrics[m]["structure_contrast"], "ridge": metrics[m]["ridge_energy"],
                "entropia": metrics[m]["entropy"],
                "CNR_x_base": metrics[m]["structure_contrast"] / base_cnr,
                "ridge_x_base": metrics[m]["ridge_energy"] / base_ridge}
            for m in metrics}
        SUMMARY["principal"][key]["deteccao"] = {
            "iris_estimada": det["iris_estimated"],
            "r_pupila": det["pupil_radius"], "r_iris": det["iris_radius"],
            "dimensao": list(img.shape[:2])}

        # Painel 2 x 3 (realces restritos ao disco da íris)
        order = [("Original", "(a) Original (canal verde)"), ("Brilho +40", "(b) Brilho +40"),
                 ("Contraste ×1,8", "(c) Contraste ×1,8"), ("Frangi", "(d) Frangi"),
                 ("Gabor", "(e) Gabor"), ("Black-hat", "(f) Black-hat")]
        h, w = img.shape[:2]
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 2 * 3.5 * h / w + 0.7))
        for ax, (m, title) in zip(axes.ravel(), order):
            out = outs[m]
            if m in ALGOS:
                out = cv2.bitwise_and(out, out, mask=disk)
            ax.imshow(out, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        save_fig(fig, f"painel_vascularizacao_{key}.png")

        # Histogramas estrutura x fundo (ilustra a CNR)
        if key == "novak":
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
            for ax, m in zip(axes, ("Contraste ×1,8", "Frangi")):
                out = outs[m]
                thr, _ = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                vals = out[ann > 0]
                bins = np.arange(257) - 0.5
                ax.hist(vals[vals <= thr], bins=bins, color="#9e9e9e", label="fundo")
                ax.hist(vals[vals > thr], bins=bins, color="#1565c0", label="estrutura")
                ax.axvline(thr, color="k", ls="--", lw=1)
                ax.set_yscale("log")
                ax.set_xlabel("intensidade no mapa realçado")
                ax.set_ylabel("pixels no anel (escala log)")
                c = metrics[m]["structure_contrast"]
                ax.set_title(f"{m}: CNR = {br(c)} (limiar de Otsu = {thr:.0f})")
                ax.legend(frameon=False)
            fig.tight_layout()
            save_fig(fig, "histograma_estrutura_fundo.png")

        # Detalhe ampliado do estroma (macro da íris)
        if key == "closeup":
            pcx, pcy = det["pupil_center"]
            ir = det["iris_radius"]
            cx, cy, half = int(pcx + 0.60 * ir), int(pcy), int(0.17 * ir)
            sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
            crops = [("(a) Original (verde)", outs["Original"]),
                     ("(b) Contraste ×1,8", outs["Contraste ×1,8"]),
                     ("(c) Frangi", outs["Frangi"]), ("(d) Gabor", outs["Gabor"])]
            fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9))
            for ax, (title, out) in zip(axes, crops):
                ax.imshow(out[sl], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                ax.set_title(title)
                ax.axis("off")
            fig.tight_layout()
            save_fig(fig, "detalhe_estroma_closeup.png")
            SUMMARY["detalhe_recorte"] = {"cx": cx, "cy": cy, "lado": 2 * half}

    write_csv("e0_metricas_principais.csv", rows)
    body = ["\\begin{tabular}{llccccc}", "\\toprule",
            "Imagem & Método & CNR & $E$ & $H$ (bits) & CNR ($\\times$base) & $E$ ($\\times$base) \\\\",
            "\\midrule"] + tex_rows[:-1] + ["\\bottomrule", "\\end{tabular}"]
    write_tex("tab_vasc_principal.tex", "\n".join(body) + "\n")

    # Barras pgfplots (CNR por método e imagem)
    ascii_key = {"Original": "Original", "Brilho +40": "Brilho", "Contraste ×1,8": "Contraste",
                 "CLAHE": "CLAHE", "Frangi": "Frangi", "Gabor": "Gabor", "Black-hat": "Blackhat"}
    hdr = ["metodo", "novak", "closeup"]
    data = [[ascii_key[m], SUMMARY["principal"]["novak"][m]["CNR"],
             SUMMARY["principal"]["closeup"][m]["CNR"]] for m in ("Original",) + BASELINES + ALGOS]
    write_dat("cnr_principal.dat", hdr, data)


# ---------------------------------------------------------------------------
# E1 — Setores angulares
# ---------------------------------------------------------------------------
def sector_masks(shape, det, ann, n=12):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = det["pupil_center"]
    ang = (np.degrees(np.arctan2(cy - yy, xx - cx)) + 360.0) % 360.0
    out = []
    for k in range(n):
        a0, a1 = 360.0 * k / n, 360.0 * (k + 1) / n
        m = np.where((ang >= a0) & (ang < a1) & (ann > 0), 255, 0).astype(np.uint8)
        out.append((a0, a1, m))
    return out


def e1_sectors(enh, cases, n_sectors=12):
    print("\n[E1] Análise por setores angulares")
    rows = []
    per_image = {}
    for key, (img, det, ann, _disk) in cases.items():
        outs = run_methods(enh, img)
        recs = []
        for a0, a1, m in sector_masks(img.shape, det, ann, n_sectors):
            if int((m > 0).sum()) < 500:
                continue
            c = {meth: cnr_or_nan(o, m) for meth, o in outs.items()}
            base_vals = [c[b] for b in BASELINES if np.isfinite(c[b])]
            c["best_base"] = max(base_vals) if base_vals else float("nan")
            recs.append((a0, a1, c))
            rows.append({"imagem": key, "setor_ini": a0, "setor_fim": a1,
                         **{k2: round(v, 4) for k2, v in c.items()}})
        per_image[key] = recs
    write_csv("e1_setores.csv", rows)

    rng = np.random.default_rng(0)

    def summarize(recs):
        res = {}
        for a in ALGOS:
            x_all = np.array([r[2][a] for r in recs])
            b_all = np.array([r[2]["best_base"] for r in recs])
            ok = np.isfinite(x_all) & np.isfinite(b_all)
            x, b = x_all[ok], b_all[ok]
            ratio = x / b
            test = stats.wilcoxon(x - b, alternative="greater")
            idx = rng.integers(0, len(ratio), size=(10000, len(ratio)))
            boot = np.median(ratio[idx], axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            res[a] = {"n": int(len(ratio)), "n_indefinidos": int((~ok).sum()),
                      "mediana_razao": float(np.median(ratio)),
                      "ic95": [float(lo), float(hi)], "vitorias": int((x > b).sum()),
                      "p_wilcoxon": float(test.pvalue),
                      "mediana_cnr": float(np.median(x)),
                      "mediana_base": float(np.median(b))}
        return res

    summ = {key: summarize(recs) for key, recs in per_image.items()}
    summ["todas"] = summarize([r for recs in per_image.values() for r in recs])
    # Sensibilidade: olho completo sem S1-S2 (0-60 graus), setores com cílios sobre a íris
    summ["novak_sem_s1s2"] = summarize([r for r in per_image["novak"] if r[0] >= 60.0])
    summ["indefinidos_por_metodo"] = {
        key: {m: int(sum(not np.isfinite(r[2][m]) for r in recs))
              for m in ("Original",) + BASELINES + ALGOS}
        for key, recs in per_image.items()}
    SUMMARY["setores"] = summ

    lines = ["\\begin{tabular}{llccccc}", "\\toprule",
             "Conjunto & Algoritmo & $n$ & Mediana CNR & Razão mediana (IC 95\\%) & Vitórias & $p$ (Wilcoxon) \\\\",
             "\\midrule"]
    for key, lab in (("novak", IMAGE_TEX["novak"]), ("closeup", IMAGE_TEX["closeup"]),
                     ("todas", "Ambas"), ("novak_sem_s1s2", "\\makecell[l]{Olho completo\\\\(sem S1--S2)}")):
        for i, a in enumerate(ALGOS):
            s = summ[key][a]
            p = s["p_wilcoxon"]
            ptxt = "$<0{,}001$" if p < 0.001 else br(p, 3)
            first = f"\\multirow{{3}}{{*}}{{{lab}}}" if i == 0 else ""
            lines.append(f"{first} & {METHOD_TEX[a]} & {s['n']} & {br(s['mediana_cnr'])} & "
                         f"{br(s['mediana_razao'])} ({br(s['ic95'][0])}--{br(s['ic95'][1])}) & "
                         f"{s['vitorias']}/{s['n']} & {ptxt} \\\\")
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
    write_tex("tab_setores.tex", "\n".join(lines) + "\n")

    # Boxplots por método
    order = ("Original",) + BASELINES + ALGOS
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for ax, key in zip(axes, ("novak", "closeup")):
        data = [[r[2][m] for r in per_image[key] if np.isfinite(r[2][m])] for m in order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops=dict(color="k", lw=1.2))
        for patch, m in zip(bp["boxes"], order):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.85)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(["Original", "Brilho", "Contraste", "CLAHE", "Frangi", "Gabor",
                            "Black-hat"], rotation=30, ha="right")
        ax.set_title(f"{IMAGE_TEX[key]} ({len(per_image[key])} setores de 30°)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("CNR no setor")
    fig.tight_layout()
    save_fig(fig, "setores_cnr_boxplot.png")

    # Perfil polar (CNR por setor)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), subplot_kw={"projection": "polar"})
    for ax, key in zip(axes, ("novak", "closeup")):
        recs = per_image[key]
        th = np.radians([(r[0] + r[1]) / 2 for r in recs])
        th_c = np.append(th, th[0])
        for m, lab in (("best_base", "melhor base"), ("Frangi", "Frangi"), ("Gabor", "Gabor")):
            v = np.array([r[2][m] for r in recs])
            col = "#616161" if m == "best_base" else COLORS[m]
            ax.plot(th_c, np.append(v, v[0]), "-o", ms=3, lw=1.5, color=col, label=lab)
        ax.set_title(IMAGE_TEX[key], pad=14)
        ax.set_rlabel_position(100)
        ax.tick_params(labelsize=8)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.05), frameon=False)
    fig.tight_layout()
    save_fig(fig, "setores_polar.png")


# ---------------------------------------------------------------------------
# E2 — Sensibilidade da métrica (Otsu global vs. restrito ao anel)
# ---------------------------------------------------------------------------
def e2_metric_definition(enh, cases):
    print("\n[E2] Definição do limiar da CNR")
    res = {}
    for key, (img, det, ann, _d) in cases.items():
        outs = run_methods(enh, img)
        res[key] = {m: {"global": cnr(enh, o, ann), "anel": cnr_otsu_in_mask(o, ann)}
                    for m, o in outs.items()}
    SUMMARY["metrica_otsu"] = res
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             " & \\multicolumn{2}{c}{" + IMAGE_TEX["novak"] + "} & \\multicolumn{2}{c}{"
             + IMAGE_TEX["closeup"] + "} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Método & Otsu global & Otsu no anel & Otsu global & Otsu no anel \\\\",
             "\\midrule"]
    for m in ("Original",) + BASELINES + ALGOS:
        a, b = res["novak"][m], res["closeup"][m]
        lines.append(f"{METHOD_TEX[m]} & {br(a['global'])} & {br(a['anel'])} & "
                     f"{br(b['global'])} & {br(b['anel'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex("tab_metrica_otsu.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# E3 — Frangi multiescala
# ---------------------------------------------------------------------------
def e3_frangi_scales(enh, cases):
    print("\n[E3] Análise multiescala do Frangi")
    cfg = enh.config
    res = {}
    for key, (img, det, ann, disk) in cases.items():
        f = img_as_float(enh._prep(img))
        raw = np.stack([frangi(f, sigmas=[s], beta=cfg.frangi_beta, gamma=cfg.frangi_gamma,
                               black_ridges=cfg.frangi_black_ridges)
                        for s in cfg.frangi_sigmas])
        full = frangi(f, sigmas=cfg.frangi_sigmas, beta=cfg.frangi_beta,
                      gamma=cfg.frangi_gamma, black_ridges=cfg.frangi_black_ridges)
        max_abs_diff = float(np.abs(raw.max(0) - full).max())
        multi_disp = enh.enhance_frangi(img)
        struct = structure_mask(multi_disp, ann)
        arg = raw.argmax(0)
        frac = [float((arg[struct] == i).mean()) for i in range(len(cfg.frangi_sigmas))]
        mean_raw = [float(raw[i][ann > 0].mean()) for i in range(len(cfg.frangi_sigmas))]
        cnr_s = [cnr(enh, VascularizationEnhancer._stretch_display(raw[i]), ann)
                 for i in range(len(cfg.frangi_sigmas))]
        res[key] = {"sigmas": list(cfg.frangi_sigmas), "fracao_dominante": frac,
                    "resposta_media": mean_raw, "cnr_escala": cnr_s,
                    "cnr_multiescala": cnr(enh, multi_disp, ann),
                    "max_abs_diff_max_vs_frangi": max_abs_diff,
                    "pixels_estrutura": int(struct.sum()),
                    # efeito da normalização de saída (transformações crescentes) sobre a CNR
                    "normalizacao": {
                        "padrao": cnr(enh, VascularizationEnhancer._stretch_display(full), ann),
                        "sem_gama": cnr(enh, VascularizationEnhancer._stretch_display(full, gamma=1.0), ann),
                        "minmax": cnr(enh, VascularizationEnhancer._to_uint8(full), ann)}}

        if key == "closeup":
            fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2))
            for i, s in enumerate(cfg.frangi_sigmas):
                ax = axes.ravel()[i]
                disp = VascularizationEnhancer._stretch_display(raw[i])
                ax.imshow(cv2.bitwise_and(disp, disp, mask=disk), cmap="gray")
                ax.set_title(f"({'abcd'[i]}) σ = {s:g} (escala única)")
                ax.axis("off")
            ax = axes.ravel()[4]
            ax.imshow(cv2.bitwise_and(multi_disp, multi_disp, mask=disk), cmap="gray")
            ax.set_title("(e) Multiescala: máximo em σ")
            ax.axis("off")
            ax = axes.ravel()[5]
            cmap = ListedColormap(["#1565c0", "#2e7d32", "#f9a825", "#c62828"])
            show = np.ma.masked_where(~struct, arg)
            ax.imshow(np.zeros_like(arg), cmap="gray", vmin=0, vmax=1)
            ax.imshow(show, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
            ax.set_title("(f) Escala dominante nas estruturas")
            ax.axis("off")
            handles = [Patch(color=cmap(i), label=f"σ = {s:g}")
                       for i, s in enumerate(cfg.frangi_sigmas)]
            ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)
            fig.tight_layout()
            save_fig(fig, "frangi_escalas_closeup.png")
    SUMMARY["frangi_escalas"] = res
    lines = ["\\begin{tabular}{ccccccc}", "\\toprule",
             " & \\multicolumn{3}{c}{" + IMAGE_TEX["novak"] + "} & \\multicolumn{3}{c}{"
             + IMAGE_TEX["closeup"] + "} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "$\\sigma$ & Dominância (\\%) & $\\bar{V}_\\sigma$ ($\\times10^{-7}$) & CNR & "
             "Dominância (\\%) & $\\bar{V}_\\sigma$ ($\\times10^{-7}$) & CNR \\\\", "\\midrule"]
    for i, s in enumerate(cfg.frangi_sigmas):
        a, b = res["novak"], res["closeup"]
        lines.append(f"{s:g} & {br(100 * a['fracao_dominante'][i], 1)} & "
                     f"{br(1e7 * a['resposta_media'][i], 2)} & {br(a['cnr_escala'][i])} & "
                     f"{br(100 * b['fracao_dominante'][i], 1)} & "
                     f"{br(1e7 * b['resposta_media'][i], 2)} & {br(b['cnr_escala'][i])} \\\\")
    lines.append("\\midrule")
    lines.append(f"Multiescala & 100,0 & -- & {br(res['novak']['cnr_multiescala'])} & 100,0 & -- & "
                 f"{br(res['closeup']['cnr_multiescala'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex("tab_frangi_escalas.tex", "\n".join(lines) + "\n")

    norm_rows = (("padrao", "Percentis 2 e 99 com gama 0,7 (padrão)"),
                 ("sem_gama", "Percentis 2 e 99, sem gama"),
                 ("minmax", "Mínimo--máximo"))
    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             "Normalização do mapa de Frangi & CNR (" + IMAGE_TEX["novak"] + ") & CNR ("
             + IMAGE_TEX["closeup"] + ") \\\\", "\\midrule"]
    for k, lab in norm_rows:
        lines.append(f"{lab} & {br(res['novak']['normalizacao'][k])} & "
                     f"{br(res['closeup']['normalizacao'][k])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex("tab_frangi_normalizacao.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# E4 — Sensibilidade a parâmetros
# ---------------------------------------------------------------------------
def e4_parameter_sensitivity(cases):
    print("\n[E4] Sensibilidade a parâmetros")
    base = PreprocessingConfig()
    sweeps = []
    for b in (0.25, 0.5, 0.75, 1.0):
        sweeps.append(("Frangi", "$\\beta$", br(b), replace(base, frangi_beta=b),
                       b == base.frangi_beta, "frangi"))
    for c in (0.05, 0.25, 1.0, 15.0):
        sweeps.append(("Frangi", "$c$", br(c), replace(base, frangi_gamma=c),
                       c == base.frangi_gamma, "frangi"))
    for sig in ((1.0,), (1.0, 2.0), (1.0, 2.0, 3.0, 4.0), (2.0, 3.0, 4.0), (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)):
        txt = "\\{" + ", ".join(f"{s:g}" for s in sig) + "\\}"
        sweeps.append(("Frangi", "$\\{\\sigma\\}$", txt, replace(base, frangi_sigmas=sig),
                       sig == base.frangi_sigmas, "frangi"))
    for lam in (6.0, 8.0, 10.0, 14.0):
        sweeps.append(("Gabor", "$\\lambda$", f"{lam:g}", replace(base, gabor_lambda=lam),
                       lam == base.gabor_lambda, "gabor"))
    for sg in (2.0, 3.0, 4.0, 6.0):
        # núcleo com suporte de pelo menos 2,5 sigma; para sigma <= 4 coincide com o padrão (21)
        ks = max(base.gabor_ksize, 2 * int(np.ceil(2.5 * sg)) + 1)
        sweeps.append(("Gabor", "$\\sigma$", f"{sg:g}", replace(base, gabor_sigma=sg, gabor_ksize=ks),
                       sg == base.gabor_sigma, "gabor"))
    for n in (4, 8, 12):
        sweeps.append(("Gabor", "$n_\\theta$", f"{n}", replace(base, gabor_n_orientations=n),
                       n == base.gabor_n_orientations, "gabor"))
    for sc in ((5,), (11,), (21,), (5, 11, 21), (5, 11, 21, 31)):
        txt = "\\{" + ", ".join(str(s) for s in sc) + "\\}"
        sweeps.append(("Morfologia", "escalas", txt, replace(base, morph_scales=sc),
                       sc == base.morph_scales, "blackhat"))

    rows, lines = [], ["\\begin{tabular}{lllcc}", "\\toprule",
                       "Algoritmo & Parâmetro & Valor & CNR (" + IMAGE_TEX["novak"] + ") & CNR ("
                       + IMAGE_TEX["closeup"] + ") \\\\", "\\midrule"]
    last_algo, last_param = None, None
    for algo, param, val, cfg, is_default, meth in sweeps:
        enh = VascularizationEnhancer(cfg)
        vals = []
        for key in ("novak", "closeup"):
            img, det, ann, _d = cases[key]
            fn = {"frangi": enh.enhance_frangi, "gabor": enh.enhance_gabor,
                  "blackhat": enh.enhance_blackhat}[meth]
            vals.append(cnr(enh, fn(img), ann))
        rows.append({"algoritmo": algo, "parametro": param, "valor": val, "padrao": is_default,
                     "cnr_novak": vals[0], "cnr_closeup": vals[1]})
        if last_algo is not None and algo != last_algo:
            lines.append("\\midrule")
        elif last_param is not None and param != last_param:
            lines.append("\\cmidrule(lr){2-5}")
        a_txt = algo if algo != last_algo else ""
        p_txt = param if (param != last_param or algo != last_algo) else ""
        v_txt = f"\\textbf{{{val}}}$^\\ast$" if is_default else val
        c_txt = [f"\\textbf{{{br(v)}}}" if is_default else br(v) for v in vals]
        lines.append(f"{a_txt} & {p_txt} & {v_txt} & {c_txt[0]} & {c_txt[1]} \\\\")
        last_algo, last_param = algo, param
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex("tab_sensibilidade.tex", "\n".join(lines) + "\n")
    write_csv("e4_sensibilidade.csv", rows)
    SUMMARY["sensibilidade"] = rows


# ---------------------------------------------------------------------------
# E5 — Robustez a perturbações fotométricas
# ---------------------------------------------------------------------------
PERTURBATIONS = [
    ("Brilho $-30$", "brilho-"), ("Brilho $+30$", "brilho+"),
    ("Contraste $\\times0{,}7$", "contraste-"), ("Contraste $\\times1{,}3$", "contraste+"),
    ("Gama $0{,}7$", "gama-"), ("Gama $1{,}5$", "gama+"),
    ("Ruído gaussiano ($\\sigma=8$)", "ruido"), ("Desfoque gaussiano ($\\sigma=1{,}5$)", "desfoque"),
    ("Compressão JPEG ($q=30$)", "jpeg"),
]


def perturb(img, kind, seed=0):
    # convertScaleAbs aplica |a*x + b|; para b < 0 isso "dobraria" os tons escuros,
    # por isso o escurecimento usa subtração com saturação em 0.
    if kind == "brilho-":
        return np.clip(img.astype(np.int16) - 30, 0, 255).astype(np.uint8)
    if kind == "brilho+":
        return cv2.convertScaleAbs(img, alpha=1.0, beta=30)
    if kind == "contraste-":
        return cv2.convertScaleAbs(img, alpha=0.7, beta=0)
    if kind == "contraste+":
        return cv2.convertScaleAbs(img, alpha=1.3, beta=0)
    if kind in ("gama-", "gama+"):
        g = 0.7 if kind == "gama-" else 1.5
        lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return cv2.LUT(img, lut)
    if kind == "ruido":
        rng = np.random.default_rng(seed)
        return np.clip(img.astype(np.float64) + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    if kind == "desfoque":
        return cv2.GaussianBlur(img, (0, 0), 1.5)
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 30])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    raise ValueError(kind)


def e5_robustness(enh, cases):
    print("\n[E5] Robustez a perturbações fotométricas")
    meths = ("CLAHE",) + ALGOS
    fns = {"CLAHE": lambda im: enh._clahe.apply(im[:, :, 1]), "Frangi": enh.enhance_frangi,
           "Gabor": enh.enhance_gabor, "Black-hat": enh.enhance_blackhat}
    rows = []
    agg = {m: {"r": [], "dice": [], "cnr_ret": []} for m in meths}
    table = {}
    for tex, kind in PERTURBATIONS:
        table[kind] = {m: {"r": [], "dice": []} for m in meths}
        for key, (img, det, ann, _d) in cases.items():
            pimg = perturb(img, kind)
            for m in meths:
                o0, o1 = fns[m](img), fns[m](pimg)
                sel = ann > 0
                r = float(np.corrcoef(o0[sel].astype(np.float64), o1[sel].astype(np.float64))[0, 1])
                s0, s1 = structure_mask(o0, ann), structure_mask(o1, ann)
                dice = float(2.0 * (s0 & s1).sum() / (s0.sum() + s1.sum() + 1e-9))
                c0, c1 = cnr(enh, o0, ann), cnr(enh, o1, ann)
                rows.append({"perturbacao": kind, "imagem": key, "metodo": m, "pearson_r": r,
                             "dice": dice, "cnr_0": c0, "cnr_p": c1, "retencao_cnr": c1 / c0})
                table[kind][m]["r"].append(r)
                table[kind][m]["dice"].append(dice)
                agg[m]["r"].append(r)
                agg[m]["dice"].append(dice)
                agg[m]["cnr_ret"].append(c1 / c0)
    write_csv("e5_robustez.csv", rows)
    lines = ["\\begin{tabular}{l" + "cc" * len(meths) + "}", "\\toprule",
             "Perturbação & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{METHOD_TEX[m]}}}" for m in meths)
             + " \\\\",
             "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(meths))),
             " & " + " & ".join("$r$ & Dice" for _ in meths) + " \\\\", "\\midrule"]
    for tex, kind in PERTURBATIONS:
        cells = []
        for m in meths:
            cells += [br(float(np.mean(table[kind][m]["r"]))),
                      br(float(np.mean(table[kind][m]["dice"])))]
        lines.append(f"{tex} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    cells = []
    for m in meths:
        cells += [f"\\textbf{{{br(float(np.mean(agg[m]['r'])))}}}",
                  f"\\textbf{{{br(float(np.mean(agg[m]['dice'])))}}}"]
    lines.append("Média & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex("tab_robustez.tex", "\n".join(lines) + "\n")
    SUMMARY["robustez"] = {
        "por_metodo": {m: {k: float(np.mean(v)) for k, v in agg[m].items()} for m in meths},
        "min_r": {m: float(np.min(agg[m]["r"])) for m in meths},
        "min_dice": {m: float(np.min(agg[m]["dice"])) for m in meths},
        "retencao_cnr_min": {m: float(np.min(agg[m]["cnr_ret"])) for m in meths},
        "retencao_cnr_max": {m: float(np.max(agg[m]["cnr_ret"])) for m in meths},
        "por_perturbacao": {k: {m: {"r": float(np.mean(v["r"])), "dice": float(np.mean(v["dice"]))}
                                for m, v in d.items()} for k, d in table.items()},
    }

    # Exemplo visual: Frangi sob desfoque e ruído
    img, det, ann, disk = cases["closeup"]
    pcx, pcy = det["pupil_center"]
    ir = det["iris_radius"]
    cx, cy, half = int(pcx + 0.60 * ir), int(pcy), int(0.17 * ir)
    sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
    shows = [("(a) sem perturbação", img), ("(b) gama 1,5", perturb(img, "gama+")),
             ("(c) ruído σ = 8", perturb(img, "ruido")), ("(d) desfoque σ = 1,5", perturb(img, "desfoque"))]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9))
    for ax, (title, im) in zip(axes, shows):
        ax.imshow(enh.enhance_frangi(im)[sl], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    save_fig(fig, "robustez_frangi_exemplo.png")


# ---------------------------------------------------------------------------
# E6 — Fantoma sintético com verdade de referência
# ---------------------------------------------------------------------------
def make_phantom(seed, size=384, noise_sigma=12.0, n_vessels=28):
    """Fundo com iluminação não uniforme e textura de baixa frequência, mais
    curvas finas claras (larguras de 1 a 3 px) com contraste variável. Devolve
    a imagem BGR e a máscara de verdade de referência dos vasos."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    illum = 70.0 + 50.0 * (xx / size) + 25.0 * np.cos(np.pi * yy / size)
    tex = cv2.GaussianBlur(rng.normal(0, 1, (size, size)), (0, 0), 6)
    tex = tex / (tex.std() + 1e-9) * 12.0
    layer = np.zeros((size, size), np.float64)
    gt = np.zeros((size, size), np.uint8)
    t = np.linspace(0, 1, 300)[:, None]
    for _ in range(n_vessels):
        p0, p1, p2 = rng.uniform(16, size - 16, (3, 2))
        pts = ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2
        pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        width = int(rng.choice([1, 2, 3]))
        contrast = float(rng.uniform(12, 35))
        tmp = np.zeros_like(layer)
        cv2.polylines(tmp, [pts], False, contrast, thickness=width, lineType=cv2.LINE_8)
        layer = np.maximum(layer, tmp)
        cv2.polylines(gt, [pts], False, 1, thickness=width, lineType=cv2.LINE_8)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.7)
    img = illum + tex + layer + rng.normal(0, noise_sigma, (size, size))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.merge([img, img, img]), gt.astype(bool)


def e6_phantom(enh, seeds=range(10), noises=(4.0, 8.0, 12.0, 16.0, 24.0)):
    print("\n[E6] Fantoma sintético com verdade de referência")
    meths = [m for m, _ in method_table(enh)]
    fns = dict(method_table(enh))
    size, border = 384, 12
    fov = np.zeros((size, size), bool)
    fov[border:-border, border:-border] = True
    fov_u8 = fov.astype(np.uint8) * 255
    res = {s: {m: {"auc": [], "cnr_gt": [], "cnr_otsu": []} for m in meths} for s in noises}
    pooled = {m: ([], []) for m in meths}
    for s in noises:
        for seed in seeds:
            bgr, gt = make_phantom(seed, size=size, noise_sigma=s)
            green = bgr[:, :, 1]
            for m in meths:
                out = fns[m](green, bgr)
                y, sc = gt[fov], out[fov].astype(np.float64)
                auc = float(roc_auc_score(y, sc))
                v, bgv = sc[y], sc[~y]
                cgt = float((v.mean() - bgv.mean()) / (bgv.std() + 1e-6))
                res[s][m]["auc"].append(auc)
                res[s][m]["cnr_gt"].append(cgt)
                res[s][m]["cnr_otsu"].append(cnr(enh, out, fov_u8))
                if s == 12.0:
                    pooled[m][0].append(y)
                    pooled[m][1].append(sc)
    rows = []
    for s in noises:
        for m in meths:
            d = res[s][m]
            rows.append({"ruido": s, "metodo": m, "auc_media": float(np.mean(d["auc"])),
                         "auc_dp": float(np.std(d["auc"], ddof=1)),
                         "cnr_gt_media": float(np.mean(d["cnr_gt"])),
                         "cnr_otsu_media": float(np.mean(d["cnr_otsu"]))})
    write_csv("e6_fantoma.csv", rows)

    keymap = {"Original": "Original", "Brilho +40": "Brilho", "Contraste ×1,8": "Contraste",
              "CLAHE": "CLAHE", "Frangi": "Frangi", "Gabor": "Gabor", "Black-hat": "Blackhat"}
    hdr = ["sigma"] + [keymap[m] for m in meths] + [keymap[m] + "_sd" for m in meths]
    data = []
    for s in noises:
        data.append([s] + [float(np.mean(res[s][m]["auc"])) for m in meths]
                    + [float(np.std(res[s][m]["auc"], ddof=1)) for m in meths])
    write_dat("auc_ruido.dat", hdr, data)

    # Tabela no nível de ruído de referência (sigma = 12)
    s = 12.0
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Método & AUC (média $\\pm$ dp) & CNR com verdade & CNR por Otsu \\\\", "\\midrule"]
    best_auc = max(np.mean(res[s][m]["auc"]) for m in meths)
    for m in meths:
        d = res[s][m]
        a = float(np.mean(d["auc"]))
        a_txt = f"{br(a, 3)} $\\pm$ {br(float(np.std(d['auc'], ddof=1)), 3)}"
        if np.isclose(a, best_auc):
            a_txt = f"\\textbf{{{br(a, 3)}}} $\\pm$ \\textbf{{{br(float(np.std(d['auc'], ddof=1)), 3)}}}"
        lines.append(f"{METHOD_TEX[m]} & {a_txt} & {br(float(np.mean(d['cnr_gt'])))} & "
                     f"{br(float(np.mean(d['cnr_otsu'])))} \\\\")
        if m == "CLAHE":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex("tab_fantoma.tex", "\n".join(lines) + "\n")

    # Correlação de Spearman entre CNR por Otsu e AUC (todas as condições)
    xs = [np.mean(res[s2][m]["cnr_otsu"]) for s2 in noises for m in meths]
    ys = [np.mean(res[s2][m]["auc"]) for s2 in noises for m in meths]
    zs = [np.mean(res[s2][m]["cnr_gt"]) for s2 in noises for m in meths]
    rho_auc = stats.spearmanr(xs, ys)
    rho_gt = stats.spearmanr(xs, zs)
    SUMMARY["fantoma"] = {
        "ruidos": list(noises), "n_sementes": len(list(seeds)),
        "auc": {str(s2): {m: [float(np.mean(res[s2][m]["auc"])), float(np.std(res[s2][m]["auc"], ddof=1))]
                          for m in meths} for s2 in noises},
        "cnr_gt": {str(s2): {m: float(np.mean(res[s2][m]["cnr_gt"])) for m in meths} for s2 in noises},
        "cnr_otsu": {str(s2): {m: float(np.mean(res[s2][m]["cnr_otsu"])) for m in meths} for s2 in noises},
        "spearman_cnr_otsu_vs_auc": [float(rho_auc.statistic), float(rho_auc.pvalue)],
        "spearman_cnr_otsu_vs_cnr_gt": [float(rho_gt.statistic), float(rho_gt.pvalue)],
        "fracao_vasos": float(make_phantom(0)[1][fov].mean()),
    }

    # Figura: exemplo do fantoma e mapas
    bgr, gt = make_phantom(0, size=size, noise_sigma=12.0)
    green = bgr[:, :, 1]
    shows = [("(a) Fantoma (σ = 12)", green), ("(b) Verdade de referência", gt.astype(np.uint8) * 255),
             ("(c) Contraste ×1,8", fns["Contraste ×1,8"](green, bgr)),
             ("(d) Frangi", fns["Frangi"](green, bgr)), ("(e) Gabor", fns["Gabor"](green, bgr)),
             ("(f) Black-hat", fns["Black-hat"](green, bgr))]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, (title, im) in zip(axes.ravel(), shows):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    save_fig(fig, "fantoma_exemplo.png")

    # Curvas ROC agregadas (sigma = 12)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for m in meths:
        y = np.concatenate(pooled[m][0])
        sc = np.concatenate(pooled[m][1])
        fpr, tpr, _ = roc_curve(y, sc)
        auc = roc_auc_score(y, sc)
        ls = "--" if m in BASELINES or m == "Original" else "-"
        ax.plot(fpr, tpr, ls, lw=1.6, color=COLORS[m], label=f"{m} (AUC = {br(auc, 3)})")
    ax.plot([0, 1], [0, 1], ":", color="k", lw=0.8)
    ax.set_xlabel("taxa de falsos positivos")
    ax.set_ylabel("taxa de verdadeiros positivos")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fantoma_roc.png")


# ---------------------------------------------------------------------------
# E7 — Custo computacional
# ---------------------------------------------------------------------------
def e7_timing(enh, cases, reps=7):
    print("\n[E7] Custo computacional")
    img, det, ann, _d = cases["novak"]
    validator = create_pupil_validator()
    fns = [(m, (lambda f=f: f(img[:, :, 1], img))) for m, f in method_table(enh) if m != "Original"]
    fns.append(("Detecção pupila/íris", lambda: demo.detect_pupil_iris(img)))
    fns.append(("Validação geométrica", lambda: validator.validate(
        det["pupil_radius"], det["iris_radius"], det["pupil_center"], det["iris_center"])))
    res = {}
    for name, fn in fns:
        fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
        res[name] = {"mediana_ms": float(np.median(ts)), "min_ms": float(np.min(ts)),
                     "max_ms": float(np.max(ts))}
    SUMMARY["tempo"] = {"imagem": list(img.shape[:2]), "repeticoes": reps, "resultados": res}
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Etapa & Mediana (ms) & Mínimo (ms) & Máximo (ms) \\\\", "\\midrule"]
    for name, r in res.items():
        label = METHOD_TEX.get(name, name)
        nd = 3 if r["mediana_ms"] < 0.1 else (2 if r["mediana_ms"] < 10 else 1)
        lines.append(f"{label} & {br(r['mediana_ms'], nd)} & {br(r['min_ms'], nd)} & {br(r['max_ms'], nd)} \\\\")
        if name == "Black-hat":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex("tab_tempo.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# E8 — Normalização polar (rubber sheet) e perfil radial
# ---------------------------------------------------------------------------
def rubber_sheet(gray, det, n_r=64, n_theta=720):
    pcx, pcy = det["pupil_center"]
    pr = det["pupil_radius"]
    icx, icy = det["iris_center"]
    ir = det["iris_radius"]
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)[None, :]
    r = np.linspace(0, 1, n_r)[:, None]
    xp, yp = pcx + pr * np.cos(theta), pcy - pr * np.sin(theta)
    xi, yi = icx + ir * np.cos(theta), icy - ir * np.sin(theta)
    X = ((1 - r) * xp + r * xi).astype(np.float32)
    Y = ((1 - r) * yp + r * yi).astype(np.float32)
    valid = (X >= 0) & (X <= gray.shape[1] - 1) & (Y >= 0) & (Y <= gray.shape[0] - 1)
    return cv2.remap(gray, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), valid


def e8_rubber_sheet(enh, cases):
    print("\n[E8] Normalização polar do mapa de Frangi")
    res = {}
    strips = {}
    for key, (img, det, ann, _d) in cases.items():
        clahe = enh._clahe.apply(img[:, :, 1])
        fr = enh.enhance_frangi(img)
        s_c, valid = rubber_sheet(clahe, det)
        s_f, _ = rubber_sheet(fr, det)
        thr, _ = cv2.threshold(fr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dens = np.array([float((s_f[i][valid[i]] > thr).mean()) if valid[i].any() else np.nan
                         for i in range(s_f.shape[0])])
        mean = np.array([float(s_f[i][valid[i]].mean()) if valid[i].any() else np.nan
                         for i in range(s_f.shape[0])])
        rr = np.linspace(0, 1, s_f.shape[0])
        inner, outer = rr <= 1 / 3, rr >= 1 / 3
        res[key] = {"densidade_interna": float(np.nanmean(dens[inner])),
                    "densidade_externa": float(np.nanmean(dens[outer])),
                    "fracao_valida": float(valid.mean())}
        strips[key] = (s_c, s_f, rr, dens, mean)
        write_dat(f"perfil_radial_{key}.dat", ["r", "densidade", "media"],
                  [[float(a), float(b), float(c)] for a, b, c in zip(rr, dens, mean)])
    SUMMARY["rubber_sheet"] = res

    s_c, s_f, rr, dens, mean = strips["novak"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 3.6))
    for ax, im, title in ((axes[0], s_c, "(a) Canal verde com CLAHE, normalizado (64 × 720)"),
                          (axes[1], s_f, "(b) Mapa de Frangi, normalizado (64 × 720)")):
        ax.imshow(im, cmap="gray", aspect="auto", vmin=0, vmax=255,
                  extent=[0, 360, 1, 0])
        ax.set_title(title, loc="left")
        ax.set_ylabel("r normalizado")
        ax.set_xticks(range(0, 361, 45))
    axes[1].set_xlabel("ângulo θ (graus, anti-horário a partir da direita)")
    fig.tight_layout()
    save_fig(fig, "rubber_sheet_novak.png")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for key, ls in (("novak", "-"), ("closeup", "--")):
        _c, _f, rr, dens, _m = strips[key]
        ax.plot(rr, 100 * dens, ls, lw=1.8, color="#1565c0" if key == "novak" else "#ef6c00",
                label=IMAGE_TEX[key])
    ax.axvspan(0, 1 / 3, color="#90caf9", alpha=0.18, lw=0)
    ax.text(0.165, ax.get_ylim()[1] * 0.95, "zona pupilar\n(aprox.)", ha="center", va="top", fontsize=8)
    ax.set_xlabel("raio normalizado (0 = borda pupilar, 1 = limbo)")
    ax.set_ylabel("pixels de estrutura (%)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "perfil_radial_frangi.png")


# ---------------------------------------------------------------------------
# E9 — Validação da pupila
# ---------------------------------------------------------------------------
def rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def e9_pupil(cases):
    print("\n[E9] Validação da pupila")
    validator = create_pupil_validator()
    cfg = validator.config

    # Figura anotada (código da demonstração, com a legenda corrigida)
    demo.run_pupil_validation()
    shutil.copyfile(os.path.join(demo.PUPIL_DIR, "iris_petr_novak_validation.png"),
                    os.path.join(FIG_DIR, "validacao_pupila_novak.png"))
    print("  figura: docs/PGC_2/figuras/validacao_pupila_novak.png")

    img, det, _a, _d = cases["novak"]
    r0 = validator.validate(det["pupil_radius"], det["iris_radius"], det["pupil_center"], det["iris_center"])
    hvid_mm = 11.71
    px_mm = r0.iris_diameter / hvid_mm
    SUMMARY["pupila_real"] = {**r0.to_dict(), "px_por_mm_estimado": px_mm,
                              "pupila_mm_estimada": r0.pupil_diameter / px_mm,
                              "hvid_mm_referencia": hvid_mm}

    # Casos sintéticos (fronteiras do classificador)
    cases_syn = [("Normal", 40.0, 0.0), ("Miose", 18.0, 0.0), ("Miose extrema", 12.0, 0.0),
                 ("Midríase moderada", 72.0, 0.0), ("Midríase", 78.0, 0.0),
                 ("Excêntrica", 40.0, 40.0), ("Anel fino", 88.0, 0.0)]
    lines = ["\\begin{tabular}{lccccl}", "\\toprule",
             "Caso & $\\rho$ & $\\tau$ & $\\epsilon$ & Pontuação & Rótulo \\\\", "\\midrule"]
    syn = []
    for name, pr, off in cases_syn:
        r = validator.validate(pr, 100.0, pupil_center=(off, 0.0), iris_center=(0.0, 0.0))
        syn.append({"caso": name, "rho": r.pupil_iris_ratio, "tau": r.thickness_ratio,
                    "eps": r.concentricity_offset, "score": r.quality_score, "rotulo": r.quality_label})
        lines.append(f"{name} & {br(r.pupil_iris_ratio)} & {br(r.thickness_ratio)} & "
                     f"{br(r.concentricity_offset)} & {br(r.quality_score)} & \\texttt{{{r.quality_label}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex("tab_pupila_sinteticos.tex", "\n".join(lines) + "\n")
    SUMMARY["pupila_sinteticos"] = syn

    # Curva de pontuação em função de rho (epsilon = 0)
    rhos = np.round(np.arange(0.05, 0.951, 0.0025), 4)
    scores, labels = [], []
    for rho in rhos:
        r = validator.validate(rho * 100.0, 100.0, pupil_center=(0, 0), iris_center=(0, 0))
        scores.append(r.quality_score)
        labels.append(r.quality_label)
    colors = {"valid": "#81c784", "borderline": "#ffd54f", "invalid": "#e57373"}
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    start = 0
    for i in range(1, len(rhos) + 1):
        if i == len(rhos) or labels[i] != labels[start]:
            ax.axvspan(rhos[start] - 0.00125, rhos[i - 1] + 0.00125, color=colors[labels[start]],
                       alpha=0.45, lw=0)
            start = i
    ax.plot(rhos, scores, color="k", lw=1.6)
    pt, = ax.plot([r0.pupil_iris_ratio], [r0.quality_score], "o", color="#1565c0", ms=7,
                  label="imagem real (ρ = " + br(r0.pupil_iris_ratio) + ")")
    for x in (cfg.borderline_low_ratio, cfg.min_pupil_iris_ratio, cfg.max_pupil_iris_ratio,
              cfg.borderline_high_ratio, 1 - cfg.min_thickness_ratio):
        ax.axvline(x, color="k", lw=0.6, ls=":")
    ax.set_xlabel("razão pupila/íris ρ (pupila centrada, ε = 0)")
    ax.set_ylabel("pontuação")
    ax.set_ylim(-0.03, 1.08)
    handles = [Patch(color=colors[k], alpha=0.6, label=k) for k in ("valid", "borderline", "invalid")]
    ax.legend(handles=handles + [pt], frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout()
    save_fig(fig, "pupila_curva_pontuacao.png")

    # Mapa de decisão rho x epsilon
    eps = np.round(np.arange(0.0, 0.501, 0.0025), 4)
    code = {"valid": 2, "borderline": 1, "invalid": 0}
    grid = np.zeros((len(eps), len(rhos)), int)
    for i, e in enumerate(eps):
        for j, rho in enumerate(rhos):
            r = validator.validate(rho * 100.0, 100.0, pupil_center=(e * 100.0, 0.0), iris_center=(0.0, 0.0))
            grid[i, j] = code[r.quality_label]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    cmap = ListedColormap([colors["invalid"], colors["borderline"], colors["valid"]])
    ax.imshow(grid, origin="lower", cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto",
              extent=[rhos[0], rhos[-1], eps[0], eps[-1]])
    label_pos = {"Miose extrema": ((0, 15), "center"), "Midríase moderada": ((0, 15), "center")}
    for s in syn:
        ax.plot(s["rho"], s["eps"], "s", color="k", ms=4, clip_on=False, zorder=5)
        off, ha = label_pos.get(s["caso"], ((4, 5), "left"))
        ax.annotate(s["caso"], (s["rho"], s["eps"]), textcoords="offset points", xytext=off,
                    ha=ha, fontsize=7, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                                                 alpha=0.75))
    ax.plot(r0.pupil_iris_ratio, r0.concentricity_offset, "o", color="#1565c0", ms=7)
    ax.annotate("imagem real", (r0.pupil_iris_ratio, r0.concentricity_offset), textcoords="offset points",
                xytext=(6, -11), fontsize=7, color="#0d47a1",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_xlabel("razão pupila/íris ρ")
    ax.set_ylabel("excentricidade ε")
    handles = [Patch(color=colors[k], label=k) for k in ("valid", "borderline", "invalid")]
    ax.legend(handles=handles, frameon=True, fontsize=8, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "pupila_mapa_decisao.png")

    # Estabilidade da detecção sob perturbações e mudanças de escala/rotação
    full = demo.load_image(IMAGES["novak"])
    conds = [("Original (768 px)", demo.resize_max(full, 768)),
             ("Escala 384 px", demo.resize_max(full, 384)),
             ("Escala 576 px", demo.resize_max(full, 576)),
             ("Escala 1152 px", demo.resize_max(full, 1152)),
             ("Rotação $10^\\circ$", rotate(img, 10)),
             ("Espelhamento horizontal", cv2.flip(img, 1))]
    for tex, kind in PERTURBATIONS:
        conds.append((tex, perturb(img, kind)))
    lines = ["\\begin{tabular}{lcccccl}", "\\toprule",
             "Condição & $D_p$ (px) & $D_i$ (px) & $\\rho$ & $\\tau$ & $\\epsilon$ & Rótulo \\\\", "\\midrule"]
    stab = []
    for tex, im in conds:
        d = demo.detect_pupil_iris(im)
        r = validator.validate(d["pupil_radius"], d["iris_radius"], d["pupil_center"], d["iris_center"])
        stab.append({"condicao": tex, "Dp": r.pupil_diameter, "Di": r.iris_diameter,
                     "rho": r.pupil_iris_ratio, "tau": r.thickness_ratio, "eps": r.concentricity_offset,
                     "rotulo": r.quality_label, "iris_estimada": d["iris_estimated"],
                     "pupila_estimada": d["pupil_estimated"]})
        mark = "$^\\dagger$" if d["iris_estimated"] else ""
        mark_p = "$^\\ddagger$" if d["pupil_estimated"] else ""
        lines.append(f"{tex} & {r.pupil_diameter:.0f}{mark_p} & {r.iris_diameter:.0f}{mark} & "
                     f"{br(r.pupil_iris_ratio)} & {br(r.thickness_ratio)} & {br(r.concentricity_offset)} & "
                     f"\\texttt{{{r.quality_label}}} \\\\")
        if tex.startswith("Espelhamento"):
            lines.append("\\midrule")
    rho_v = np.array([s["rho"] for s in stab])
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex("tab_pupila_estabilidade.tex", "\n".join(lines) + "\n")
    write_csv("e9_pupila_estabilidade.csv", stab)

    # Diagnóstico das falhas da pupila sob escurecimento: máscara escura do detector
    diag = {}
    for name, im in (("original", img), ("brilho-", perturb(img, "brilho-")),
                     ("gama+", perturb(img, "gama+"))):
        g = cv2.medianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 7)
        p3 = float(np.percentile(g, 3))
        _, dark = cv2.threshold(g, p3 + 20.0, 255, cv2.THRESH_BINARY_INV)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=2)
        cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = max(cnts, key=cv2.contourArea)
        _c, rr = cv2.minEnclosingCircle(big)
        diag[name] = {"limiar": p3 + 20.0, "fracao_escura": float((dark > 0).mean()),
                      "area_maior": float(cv2.contourArea(big)),
                      "circularidade_maior": float(cv2.contourArea(big) / (np.pi * rr * rr))}
    SUMMARY["pupila_diagnostico_escurecimento"] = diag
    SUMMARY["pupila_estabilidade"] = {
        "linhas": stab, "rho_media": float(rho_v.mean()), "rho_dp": float(rho_v.std(ddof=1)),
        "rho_cv_pct": float(100 * rho_v.std(ddof=1) / rho_v.mean()),
        "rho_min": float(rho_v.min()), "rho_max": float(rho_v.max()),
        "n_valid": int(sum(s["rotulo"] == "valid" for s in stab)), "n": len(stab),
        "n_iris_estimada": int(sum(bool(s["iris_estimada"]) for s in stab)),
        "n_pupila_estimada": int(sum(bool(s["pupila_estimada"]) for s in stab))}


# ---------------------------------------------------------------------------
# M — Figuras de método (banco de Gabor, CLAHE, anel e setores)
# ---------------------------------------------------------------------------
def m_method_figures(enh, cases):
    print("\n[M] Figuras de método")
    kernels = enh._gabor_kernels
    vmax = max(float(np.abs(k).max()) for k in kernels)
    fig, axes = plt.subplots(1, len(kernels), figsize=(10.5, 1.75))
    for i, (ax, k) in enumerate(zip(axes, kernels)):
        ax.imshow(k, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title("θ = " + f"{180.0 * i / len(kernels):g}".replace(".", ",") + "°", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    save_fig(fig, "banco_gabor.png")

    img, det, ann, disk = cases["novak"]
    green = img[:, :, 1]
    cl = enh._clahe.apply(green)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.9), gridspec_kw={"height_ratios": [2.3, 1]})
    for j, (im, title) in enumerate(((green, "(a) Canal verde"), (cl, "(b) Canal verde após CLAHE"))):
        axes[0, j].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[0, j].set_title(title)
        axes[0, j].axis("off")
        axes[1, j].hist(im[disk > 0], bins=np.arange(257) - 0.5, color="#616161")
        axes[1, j].set_title(f"({'cd'[j]}) Histograma no disco da íris", fontsize=10)
        axes[1, j].set_xlabel("intensidade")
        axes[1, j].set_ylabel("pixels")
    fig.tight_layout()
    save_fig(fig, "clahe_novak.png")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    sel = ann > 0
    rgb[sel] = 0.6 * rgb[sel] + 0.4 * np.array([30.0, 136.0, 229.0])
    pcx, pcy = det["pupil_center"]
    pr, ir = det["pupil_radius"], det["iris_radius"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.imshow(rgb.astype(np.uint8))
    for k in range(12):
        a = np.radians(30.0 * k)
        ax.plot([pcx + pr * np.cos(a), pcx + 1.04 * ir * np.cos(a)],
                [pcy - pr * np.sin(a), pcy - 1.04 * ir * np.sin(a)], color="white", lw=0.9)
        am = np.radians(30.0 * k + 15.0)
        ax.text(pcx + 1.13 * ir * np.cos(am), pcy - 1.13 * ir * np.sin(am), f"S{k + 1}",
                color="white", fontsize=7.5, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55))
    ax.add_patch(Circle(det["pupil_center"], pr, fill=False, ec="#00e676", lw=1.2))
    ax.add_patch(Circle(det["iris_center"], ir, fill=False, ec="#29b6f6", lw=1.2))
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    fig.tight_layout()
    save_fig(fig, "setores_layout_novak.png")


# ---------------------------------------------------------------------------
# E10 — Escolha do canal de cor
# ---------------------------------------------------------------------------
def e10_channels(enh, cases):
    print("\n[E10] Escolha do canal de cor")
    names = (("R", "Vermelho (R)"), ("G", "Verde (G)$^\\ast$"), ("B", "Azul (B)"),
             ("Y", "Luminância (Y)"))
    res = {}
    for key, (img, _det, ann, _d) in cases.items():
        b, g, r = cv2.split(img)
        chans = {"R": r, "G": g, "B": b, "Y": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)}
        res[key] = {}
        for nm, ch in chans.items():
            im3 = cv2.merge([ch, ch, ch])
            res[key][nm] = {"cnr_canal": cnr(enh, ch, ann),
                            "cnr_frangi": cnr(enh, enh.enhance_frangi(im3), ann),
                            "cnr_gabor": cnr(enh, enh.enhance_gabor(im3), ann),
                            "dp_anel": float(ch[ann > 0].std())}
    SUMMARY["canais"] = res
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             " & \\multicolumn{3}{c}{" + IMAGE_TEX["novak"] + "} & \\multicolumn{3}{c}{"
             + IMAGE_TEX["closeup"] + "} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Canal & Canal & Frangi & Gabor & Canal & Frangi & Gabor \\\\", "\\midrule"]
    for nm, tex in names:
        a, b2 = res["novak"][nm], res["closeup"][nm]
        lines.append(f"{tex} & {br(a['cnr_canal'])} & {br(a['cnr_frangi'])} & {br(a['cnr_gabor'])} & "
                     f"{br(b2['cnr_canal'])} & {br(b2['cnr_frangi'])} & {br(b2['cnr_gabor'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex("tab_canais.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
def main():
    ensure_dirs()
    enh = VascularizationEnhancer(PreprocessingConfig())
    cases = {k: load_case(k) for k in IMAGES}
    SUMMARY["ambiente"] = {
        "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
        "scikit-image": skimage.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        "plataforma": platform.platform(), "processador": platform.processor()}
    m_method_figures(enh, cases)
    e0_panels_and_main_table(enh, cases)
    e1_sectors(enh, cases)
    e2_metric_definition(enh, cases)
    e3_frangi_scales(enh, cases)
    e4_parameter_sensitivity(cases)
    e5_robustness(enh, cases)
    e6_phantom(enh)
    e7_timing(enh, cases)
    e8_rubber_sheet(enh, cases)
    e9_pupil(cases)
    e10_channels(enh, cases)
    with open(os.path.join(OUT_DIR, "resumo.json"), "w", encoding="utf-8") as f:
        json.dump(SUMMARY, f, ensure_ascii=False, indent=2, default=float)
    print("\nResumo em results/report/resumo.json")


if __name__ == "__main__":
    main()

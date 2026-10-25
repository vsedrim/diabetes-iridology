"""
Utilitários compartilhados pelos experimentos do PGC III
========================================================
Caminhos de saída, formatação numérica em português (vírgula decimal),
gravação de figuras, tabelas LaTeX, séries de dados e CSVs.
"""

import csv
import json
import os
import platform
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DOC_DIR = os.path.join(REPO_ROOT, "docs", "PGC_3")
FIG_DIR = os.path.join(DOC_DIR, "figuras")
TAB_DIR = os.path.join(DOC_DIR, "tabelas")
DAT_DIR = os.path.join(DOC_DIR, "dados")
OUT_DIR = os.path.join(REPO_ROOT, "results", "pgc3")

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200,
})


def ensure_dirs():
    for d in (FIG_DIR, TAB_DIR, DAT_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)


def br(x, nd=2, sign=False):
    """Número com vírgula decimal para o texto em português."""
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{abs(x):.{nd}f}".replace(".", ",")
    if x < 0 and float(s.replace(",", ".")) != 0:
        return f"$-${s}"
    if sign and x > 0:
        return f"$+${s}"
    return s


def br_pm(mean, sd, nd=3):
    return f"{br(mean, nd)} $\\pm$ {br(sd, nd)}"


def br_p(p):
    """Valor p formatado para tabelas."""
    if p is None or not np.isfinite(p):
        return "--"
    if p < 0.001:
        return "$<$0,001"
    return br(p, 3)


class CommaFormatter(ScalarFormatter):
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def decimal_comma(fig):
    for ax in fig.axes:
        if ax.name == "polar":
            continue
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale == "linear" and type(axis.get_major_formatter()) is ScalarFormatter:
                axis.set_major_formatter(CommaFormatter())


def save_fig(fig, name, comma=True):
    if comma:
        decimal_comma(fig)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura: docs/PGC_3/figuras/{name}")


def write_tex(name, body, script):
    with open(os.path.join(TAB_DIR, name), "w", encoding="utf-8") as f:
        f.write(f"% Gerado por experiments/{script} (não editar à mão).\n")
        f.write(body)
    print(f"  tabela: docs/PGC_3/tabelas/{name}")


def write_csv(name, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_dat(name, header, rows):
    with open(os.path.join(DAT_DIR, name), "w", encoding="utf-8") as f:
        f.write(" ".join(header) + "\n")
        for r in rows:
            f.write(" ".join(f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v)
                             for v in r) + "\n")
    print(f"  dados: docs/PGC_3/dados/{name}")


def write_json(name, obj):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


def environment():
    import cv2
    import scipy
    import skimage
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "scikit-image": skimage.__version__, "opencv": cv2.__version__,
            "matplotlib": matplotlib.__version__, "plataforma": platform.platform(),
            "processador": platform.processor(), "cpus": os.cpu_count()}


def rubber_sheet_circles(image, det, n_r=201, n_theta=720):
    """
    Normalização rubber sheet de Daugman a partir de dois círculos (pupila e
    íris), com interpolação bilinear. A linha 0 é a borda pupilar e a última
    linha é o limbo; a coluna j corresponde ao ângulo 2*pi*j/n_theta.
    Aceita imagem em escala de cinza ou colorida.
    """
    import cv2
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
    return cv2.remap(image, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

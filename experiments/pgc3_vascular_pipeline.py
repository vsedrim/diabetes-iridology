"""
PGC III, parte A: imagens filtradas como entrada direta do pipeline da IC
=========================================================================
Pergunta do orientador: é possível usar as imagens resultantes dos filtros de
realce de vascularização (Frangi, Gabor e black-hat) diretamente no pipeline
de classificação da IC, sem criar descritores manuais de vascularização?

O script reproduz o pipeline da IC (faixa normalizada -> pré-processamento ->
extração de features -> classificador com padronização), troca apenas a
imagem de entrada e avalia todas as combinações com o mesmo protocolo:

  A0. Auditoria da base legada (cópias, espelhamento, costura, cor global)
  A1. Grade principal: 8 representações x 3 conjuntos de features x 5
      classificadores, validação cruzada estratificada repetida (10 x 5)
  A2. Comparação pareada com a imagem original (teste t corrigido de Nadeau
      e Bengio, ajuste de Holm)
  A3. Ranking global (Friedman, Iman-Davenport e diferença crítica de Nemenyi)
  A4. Teste de permutação (regressão logística sobre os pixels)
  A5. Controles: atalho de cor global, descritores manuais de vascularização
      e fusão da imagem original com a filtrada
  A6. Robustez fotométrica (treino na condição original, teste perturbado)
  A7. Replicação no leiaute personBase
  A8. Mapas de diferença entre classes (d de Cohen por pixel, FDR)
  M.  O que o pipeline "vê": representações de uma faixa pública

Saídas:
  docs/PGC_3/figuras/  figuras do relatório
  docs/PGC_3/tabelas/  tabelas LaTeX (incluídas com \\input)
  results/pgc3/        CSVs completos, escores fora do fold e resumo_vascular.json

A base legada não é versionada nem redistribuída; nenhuma figura mostra faixas
dela. As ilustrações usam a fotografia pública de docs/imagens.

Uso:
    python experiments/pgc3_vascular_pipeline.py --data <Data.zip ou Data/Data>
    python experiments/pgc3_vascular_pipeline.py --quick      # execução reduzida
"""

import argparse
import hashlib
import os
import sys
import time
import warnings

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import skeletonize
from skimage.util import img_as_float
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pgc3_common as C  # noqa: E402
from pgc3_common import br, br_p  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from src.classifiers import ClassifierFactory  # noqa: E402
from src.config import (ClassifierConfig, ClassifierType as CT, ColorChannel,  # noqa: E402
                        FeatureType as FT, PhotometricTransform as PT, PreprocessingConfig)
from src.feature_extraction import FeatureExtractor  # noqa: E402
from src.legacy_data import load_legacy_layout, resolve_legacy_source  # noqa: E402
from src import model_comparison as mc  # noqa: E402
from src.preprocessing import ImagePreprocessor, VascularizationEnhancer  # noqa: E402

SCRIPT = "pgc3_vascular_pipeline.py"

# (chave, canal, transformação, rótulo no texto, cor)
REPS = [
    ("original", ColorChannel.GRAY, PT.ORIGINAL, "Original (cinza)", "#424242"),
    ("histograma", ColorChannel.GRAY, PT.HISTOGRAM, "Equalização global", "#9e9e9e"),
    ("clahe", ColorChannel.GRAY, PT.CLAHE, "CLAHE", "#8d6e63"),
    ("verde_clahe", ColorChannel.G, PT.CLAHE, "Verde + CLAHE", "#a1887f"),
    ("frangi", ColorChannel.G, PT.FRANGI, "Frangi", "#1565c0"),
    ("gabor", ColorChannel.G, PT.GABOR, "Gabor", "#2e7d32"),
    ("blackhat", ColorChannel.G, PT.BLACKHAT, "Black-hat", "#ef6c00"),
    ("vessel", ColorChannel.G, PT.VESSEL, "Frangi + black-hat", "#6a1b9a"),
]
REP_KEYS = [r[0] for r in REPS]
REP_LABEL = {r[0]: r[3] for r in REPS}
REP_COLOR = {r[0]: r[4] for r in REPS}
REP_TEX = {"original": "Original (cinza)", "histograma": "Equalização global", "clahe": "CLAHE",
           "verde_clahe": "Verde + CLAHE", "frangi": "Frangi", "gabor": "Gabor",
           "blackhat": "\\textit{Black-hat}", "vessel": "Frangi + \\textit{black-hat}"}
FILTER_KEYS = ["frangi", "gabor", "blackhat", "vessel"]

FEATURE_SETS = [("pixels", "Pixels"), ("textura", "Textura"), ("ic", "Padrão da IC")]
FS_TEX = {"pixels": "Pixels", "textura": "Textura", "ic": "Padrão da IC"}
CLASSIFIERS = [(CT.LR, "RL"), (CT.SVM, "SVM"), (CT.RF, "RF"), (CT.MLP, "MLP"), (CT.ADABOOST, "AdaBoost")]
CLF_TEX = {"RL": "RL", "SVM": "SVM", "RF": "RF", "MLP": "MLP", "AdaBoost": "AdaBoost"}

PERTURBATIONS = [
    ("brilho-", "Brilho $-30$"), ("brilho+", "Brilho $+30$"),
    ("contraste-", "Contraste $\\times0{,}7$"), ("contraste+", "Contraste $\\times1{,}3$"),
    ("gama-", "Gama $0{,}7$"), ("gama+", "Gama $1{,}5$"),
    ("ruido", "Ruído ($\\sigma=8$)"), ("desfoque", "Desfoque ($\\sigma=1{,}5$)"),
    ("jpeg", "JPEG ($q=30$)"),
]
PERT_LABEL = {"brilho-": "Brilho −30", "brilho+": "Brilho +30", "contraste-": "Contraste ×0,7",
              "contraste+": "Contraste ×1,3", "gama-": "Gama 0,7", "gama+": "Gama 1,5",
              "ruido": "Ruído σ=8", "desfoque": "Desfoque σ=1,5", "jpeg": "JPEG q=30"}

METRICS = ["bal_acc", "auc", "acc", "sens", "spec", "prec", "f1"]
SUMMARY = {}


# ---------------------------------------------------------------------------
# Representações e features (executadas nos processos de trabalho)
# ---------------------------------------------------------------------------
_TOOLS = {}


def _tools():
    if "pre" not in _TOOLS:
        cfg = PreprocessingConfig()
        _TOOLS["pre"] = ImagePreprocessor(cfg)
        _TOOLS["fe"] = FeatureExtractor()
        _TOOLS["enh"] = VascularizationEnhancer(cfg)
        _TOOLS["cfg"] = cfg
    return _TOOLS["pre"], _TOOLS["fe"], _TOOLS["enh"], _TOOLS["cfg"]


def perturb(img, kind, seed=0):
    """Perturbações fotométricas (as mesmas da demonstração dos realces, aplicadas à faixa BGR)."""
    if kind is None:
        return img
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
        _ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 30])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    raise ValueError(kind)


def manual_vascular_descriptors(img):
    """
    Descritores manuais de vascularização, como referência para a pergunta do
    orientador: resposta bruta de Frangi (verde + CLAHE), máscara de Otsu da
    própria imagem, esqueleto, pontos de ramificação e de extremidade, e o
    perfil por terços radiais (zona pupilar, intermediária e ciliar).
    """
    _pre, _fe, enh, cfg = _tools()
    gray = enh._prep(img)
    v = frangi(img_as_float(gray), sigmas=cfg.frangi_sigmas, beta=cfg.frangi_beta,
               gamma=cfg.frangi_gamma, black_ridges=cfg.frangi_black_ridges)
    v = v / (v.max() + 1e-12) if v.max() > 0 else v
    try:
        thr = threshold_otsu(v)
    except ValueError:
        thr = float(v.mean())
    mask = v > thr
    skel = skeletonize(mask)
    nb = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
    branch = skel & (nb >= 4)
    ends = skel & (nb == 2)
    area = float(v.size)
    thirds = np.array_split(np.arange(v.shape[0]), 3)
    feats = [v.mean(), v.std(), np.percentile(v, 90), mask.mean(), skel.sum() / area,
             branch.sum() / area, ends.sum() / area]
    feats += [v[t].mean() for t in thirds]
    feats += [mask[t].mean() for t in thirds]
    return np.asarray(feats, dtype=np.float32)


MANUAL_NAMES = ["vesselness média", "vesselness DP", "vesselness p90", "densidade (Otsu)",
                "esqueleto/área", "ramificações/área", "extremidades/área",
                "vesselness terço 1", "vesselness terço 2", "vesselness terço 3",
                "densidade terço 1", "densidade terço 2", "densidade terço 3"]


def color_shortcut_descriptors(img):
    """Médias e desvios globais de B, G, R e do cinza: controle de atalho por cor e iluminação."""
    f = img.reshape(-1, 3).astype(np.float64)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return np.asarray(list(f.mean(axis=0)) + list(f.std(axis=0)) + [gray.mean(), gray.std()],
                      dtype=np.float32)


def image_features(img, perturbation=None, seed=0, with_controls=True):
    """Features de todas as representações para uma faixa (um processo de trabalho)."""
    pre, fe, _enh, _cfg = _tools()
    img = perturb(img, perturbation, seed)
    out = {}
    for key, ch, tr, *_ in REPS:
        rep = pre.preprocess(img, ch, tr)
        pix = fe.extract_features(rep, feature_types=[FT.PIXEL])
        tex = fe.extract_features(rep, feature_types=[FT.LBP, FT.GLCM, FT.INTENSITY_STATS])
        out[key] = (pix, tex)
    if with_controls:
        out["_manual"] = manual_vascular_descriptors(img)
        out["_cor"] = color_shortcut_descriptors(img)
    return out


def compute_features(images, jobs, perturbation=None, with_controls=True, desc=""):
    t0 = time.time()
    res = Parallel(n_jobs=jobs, backend="loky")(
        delayed(image_features)(im, perturbation, 1000 + i, with_controls) for i, im in enumerate(images))
    feats = {}
    for key in REP_KEYS:
        pix = np.stack([r[key][0] for r in res]).astype(np.float32)
        tex = np.stack([r[key][1] for r in res]).astype(np.float32)
        feats[(key, "pixels")] = pix
        feats[(key, "textura")] = tex
        feats[(key, "ic")] = np.hstack([pix, tex])
    if with_controls:
        feats["manual"] = np.stack([r["_manual"] for r in res])
        feats["cor"] = np.stack([r["_cor"] for r in res])
    print(f"  features {desc or 'originais'}: {len(images)} imagens em {time.time() - t0:.0f} s")
    return feats


def cached_features(images, jobs, cache_name, perturbation=None, with_controls=True):
    """Cache local (results/pgc3/cache) para não recalcular as features a cada execução."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(images).tobytes())
    digest.update(f"{perturbation}|{with_controls}|v3".encode())
    path = os.path.join(C.OUT_DIR, "cache", f"{cache_name}_{digest.hexdigest()[:12]}.npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            feats = {}
            for k in z.files:
                if k in ("manual", "cor"):
                    feats[k] = z[k]
                else:
                    rep, fs = k.split("__")
                    feats[(rep, fs)] = z[k]
        print(f"  features {cache_name}: cache {os.path.basename(path)}")
        return feats
    feats = compute_features(images, jobs, perturbation, with_controls, cache_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{(k if isinstance(k, str) else f"{k[0]}__{k[1]}"): v
                                 for k, v in feats.items()})
    return feats


# ---------------------------------------------------------------------------
# Validação cruzada
# ---------------------------------------------------------------------------
def make_splits(y, groups, n_repeats, n_folds, seed0=2026):
    splits = []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed0 + r)
        splits.append([(tr, te) for tr, te in cv.split(np.zeros(len(y)), y, groups)])
    return splits


def _score(model, X):
    if hasattr(model, "decision_function"):
        try:
            return np.asarray(model.decision_function(X), dtype=float)
        except (AttributeError, NotImplementedError):
            pass
    return model.predict_proba(X)[:, 1]


def fold_metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    try:
        auc = roc_auc_score(y, score)
    except ValueError:
        auc = np.nan
    return {"bal_acc": (sens + spec) / 2, "auc": auc, "acc": (tp + tn) / len(y), "sens": sens,
            "spec": spec, "prec": prec, "f1": f1}


def run_repeat(X, y, folds, clf_type, X_tests=None):
    """
    Executa os folds de uma repetição para um classificador do pipeline da IC.
    ``X_tests`` (opcional) mapeia condições -> matrizes de teste alternativas,
    usadas na robustez (o modelo treina em X e é avaliado em cada condição).
    """
    fac = ClassifierFactory(ClassifierConfig())
    rows, oof = [], np.full(len(y), np.nan)
    extra = []
    for f, (tr, te) in enumerate(folds):
        model = fac.create_pipeline(clf_type)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            sc = _score(model, X[te])
            rows.append({"fold": f, **fold_metrics(y[te], pred, sc)})
            oof[te] = sc
            if X_tests:
                for cond, Xc in X_tests.items():
                    p2 = model.predict(Xc[te])
                    extra.append({"fold": f, "condicao": cond, **fold_metrics(y[te], p2, _score(model, Xc[te]))})
    return rows, oof, extra


def run_grid(feats, y, splits, jobs, configs):
    """configs: lista de (nome_rep, conjunto, tipo_classificador, rótulo_classificador)."""
    tasks = []
    for (rep, fs, ctype, clabel) in configs:
        X = feats[(rep, fs)] if fs is not None else feats[rep]
        for r, folds in enumerate(splits):
            tasks.append((rep, fs, clabel, r, X, folds, ctype))
    t0 = time.time()
    out = Parallel(n_jobs=jobs, backend="loky", verbose=0)(
        delayed(run_repeat)(X, y, folds, ctype) for (_rep, _fs, _cl, _r, X, folds, ctype) in tasks)
    rows, oof = [], {}
    for (rep, fs, clabel, r, _X, _folds, _ct), (frows, foof, _e) in zip(tasks, out):
        for fr in frows:
            rows.append({"rep": rep, "fs": fs or "-", "clf": clabel, "repeticao": r, **fr})
        oof[(rep, fs or "-", clabel, r)] = foof
    print(f"  {len(configs)} configurações x {len(splits)} repetições em {time.time() - t0:.0f} s")
    return rows, oof


def metric_matrix(rows, rep, fs, clf, metric):
    vals = [r[metric] for r in rows if r["rep"] == rep and r["fs"] == fs and r["clf"] == clf]
    return np.asarray(vals, dtype=float)


def summarize(rows, n_train, n_test):
    keys = sorted({(r["rep"], r["fs"], r["clf"]) for r in rows},
                  key=lambda k: (REP_KEYS.index(k[0]) if k[0] in REP_KEYS else 99, k[1], k[2]))
    out = []
    for rep, fs, clf in keys:
        row = {"rep": rep, "fs": fs, "clf": clf}
        for m in METRICS:
            v = metric_matrix(rows, rep, fs, clf, m)
            mean, lo, hi = mc.corrected_mean_ci(v, n_train, n_test)
            row[f"{m}_media"] = mean
            row[f"{m}_dp"] = float(np.nanstd(v, ddof=1))
            row[f"{m}_ic_inf"] = lo
            row[f"{m}_ic_sup"] = hi
        row["n_folds"] = int(len(metric_matrix(rows, rep, fs, clf, "acc")))
        out.append(row)
    return out


def lookup(summary, rep, fs, clf):
    for s in summary:
        if s["rep"] == rep and s["fs"] == fs and s["clf"] == clf:
            return s
    raise KeyError((rep, fs, clf))


# ---------------------------------------------------------------------------
# A0 — Auditoria
# ---------------------------------------------------------------------------
def a0_audit(ds, ds_pb, source=None):
    print("\n[A0] Auditoria da base legada")
    md, mdb = ds.metadata, ds_pb.metadata
    n = len(ds)
    gray = ds.images.astype(np.float32).mean(axis=3)[:, ::4, ::4].reshape(n, -1)
    gray -= gray.mean(axis=1, keepdims=True)
    gray /= np.linalg.norm(gray, axis=1, keepdims=True)
    corr = (gray @ gray.T)[np.triu_indices(n, 1)]
    color = {}
    for lab in (0, 1):
        m = ds.labels == lab
        color[lab] = [float(ds.images[m][..., c].mean()) for c in (2, 1, 0)]
    # Lado do olho: desfeito o espelhamento, cada faixa original deve existir em L_split ou R_split.
    side = {}
    if source is not None:
        from src.legacy_data import read_layout_arrays, sha256_bytes
        sets = {lay: {sha256_bytes(x) for x in np.concatenate(read_layout_arrays(source, lay))}
                for lay in ("L_split", "R_split")}
        raw = load_legacy_layout(source, "personBase_invert", repair_seam=False, near_duplicate_threshold=None)
        for key in ("espelhadas_em_R", "espelhadas_em_L", "nao_espelhadas_em_L", "nao_espelhadas_em_R"):
            side[key] = 0
        for im, mir in zip(raw.images, raw.mirrored):
            orig = np.roll(im, -im.shape[1] // 2, axis=1)[:, ::-1] if mir else im
            h = sha256_bytes(orig)
            prefix = "espelhadas" if mir else "nao_espelhadas"
            if h in sets["R_split"]:
                side[f"{prefix}_em_R"] += 1
            if h in sets["L_split"]:
                side[f"{prefix}_em_L"] += 1
        print(f"  lado do olho: {side}")
    SUMMARY["auditoria"] = {
        "personBase_invert": {k: v for k, v in md.items() if k not in ("duplicate_groups",)},
        "personBase": {k: v for k, v in mdb.items() if k not in ("duplicate_groups",)},
        "compartilhadas": int(len(set(ds.hashes.tolist()) & set(ds_pb.hashes.tolist()))),
        "lado": side,
        "correlacao_pares": {"mediana": float(np.median(corr)), "p99": float(np.percentile(corr, 99)),
                             "max": float(corr.max())},
        "cor_media_rgb": {"controle": color[0], "diabetes": color[1]},
    }
    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             "Item & \\texttt{personBase\\_invert} & \\texttt{personBase} \\\\", "\\midrule",
             f"Linhas (controle / diabetes) & {md['rows_control']} / {md['rows_diabetic']} & "
             f"{mdb['rows_control']} / {mdb['rows_diabetic']} \\\\",
             f"Cópias exatas removidas & {md['exact_duplicates_removed']} & {mdb['exact_duplicates_removed']} \\\\",
             f"Imagens únicas (controle / diabetes) & {md['unique_control']} / {md['unique_diabetic']} & "
             f"{mdb['unique_control']} / {mdb['unique_diabetic']} \\\\",
             f"Maior multiplicidade de uma cópia & {max(int(k) for k in md['duplicate_multiplicity'])} & "
             f"{max(int(k) for k in mdb['duplicate_multiplicity'])} \\\\",
             f"Faixas espelhadas (controle / diabetes) & {md['mirrored_control']} / {md['mirrored_diabetic']} & "
             f"{mdb['mirrored_control']} / {mdb['mirrored_diabetic']} \\\\",
             "Coluna nula da costura (coluna: imagens) & "
             + ", ".join(f"{k}: {v}" for k, v in md["seam_columns_repaired"].items()) + " & "
             + ", ".join(f"{k}: {v}" for k, v in mdb["seam_columns_repaired"].items()) + " \\\\",
             f"Pares com correlação $\\geq 0{{,}}97$ & {len(md['near_duplicate_pairs'])} & "
             f"{len(mdb['near_duplicate_pairs'])} \\\\",
             f"Grupos da validação cruzada & {md['cv_groups']} & {mdb['cv_groups']} \\\\",
             "\\bottomrule", "\\end{tabular}"]
    C.write_tex("tab_auditoria.tex", "\n".join(lines) + "\n", SCRIPT)


# ---------------------------------------------------------------------------
# M — O que o pipeline "vê"
# ---------------------------------------------------------------------------
def m_representations_figure():
    print("\n[M] Representações de uma faixa pública (fotografia de Petr Novák)")
    import demo_vascularization_pupil as demo
    img = demo.resize_max(demo.load_image("iris_petr_novak.jpg"), 768)
    det = demo.detect_pupil_iris(img)
    strip = C.rubber_sheet_circles(img, det, n_r=201, n_theta=720)
    # Instâncias locais: o cache _TOOLS do processo principal precisa ficar vazio,
    # porque é serializado junto com as funções enviadas aos processos de trabalho.
    pre, fe = ImagePreprocessor(PreprocessingConfig()), FeatureExtractor()
    fig, axes = plt.subplots(4, 2, figsize=(11.0, 6.4))
    for ax, (key, ch, tr, label, _col) in zip(axes.ravel(), REPS):
        rep = pre.preprocess(strip, ch, tr)
        ax.imshow(rep, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.set_title(label, loc="left", fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    C.save_fig(fig, "representacoes_faixa_novak.png", comma=False)

    rep = pre.preprocess(strip, ColorChannel.G, PT.FRANGI)
    pix = fe.extract_pixel_features(rep).reshape(50, 180)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 1.9))
    axes[0].imshow(rep, cmap="gray", aspect="auto")
    axes[0].set_title("(a) Mapa de Frangi, 201 × 720", loc="left", fontsize=9.5)
    axes[1].imshow(pix, cmap="gray", aspect="auto")
    axes[1].set_title("(b) Vetor de pixels da IC, 50 × 180 = 9000 valores", loc="left", fontsize=9.5)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    C.save_fig(fig, "vetor_pixels_frangi_novak.png", comma=False)


# ---------------------------------------------------------------------------
# A1 a A3 — Grade principal, comparações e ranking
# ---------------------------------------------------------------------------
def grid_configs():
    return [(rep, fs, ct, cl) for rep in REP_KEYS for fs, _ in FEATURE_SETS for ct, cl in CLASSIFIERS]


def a1_grid_tables(summary, layout_tex):
    print("\n[A1] Tabelas da grade principal")
    for fs, fs_label in FEATURE_SETS:
        lines = ["\\begin{tabular}{l" + "c" * len(CLASSIFIERS) + "}", "\\toprule",
                 "Representação & " + " & ".join(CLF_TEX[cl] for _c, cl in CLASSIFIERS) + " \\\\",
                 "\\midrule"]
        best = {cl: max(REP_KEYS, key=lambda r: lookup(summary, r, fs, cl)["bal_acc_media"])
                for _c, cl in CLASSIFIERS}
        for rep in REP_KEYS:
            cells = []
            for _c, cl in CLASSIFIERS:
                s = lookup(summary, rep, fs, cl)
                txt = f"{br(s['bal_acc_media'], 3)} ({br(s['auc_media'], 3)})"
                cells.append(f"\\textbf{{{txt}}}" if best[cl] == rep else txt)
            lines.append(f"{REP_TEX[rep]} & " + " & ".join(cells) + " \\\\")
            if rep == "verde_clahe":
                lines.append("\\midrule")
        lines += ["\\bottomrule", "\\end{tabular}"]
        C.write_tex(f"tab_grade_{fs}.tex", "\n".join(lines) + "\n", SCRIPT)


def a1_heatmap(summary):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0), sharey=True)
    vals_all = [s["bal_acc_media"] for s in summary if s["rep"] in REP_KEYS and s["fs"] != "-"]
    vmin, vmax = min(vals_all), max(vals_all)
    for ax, (fs, fs_label) in zip(axes, FEATURE_SETS):
        M = np.array([[lookup(summary, rep, fs, cl)["bal_acc_media"] for _c, cl in CLASSIFIERS]
                      for rep in REP_KEYS])
        im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.3f}".replace(".", ","), ha="center", va="center", fontsize=7.6,
                        color="white" if M[i, j] < (vmin + vmax) / 2 else "black")
        ax.set_xticks(range(len(CLASSIFIERS)))
        ax.set_xticklabels([cl for _c, cl in CLASSIFIERS], fontsize=8.5)
        ax.set_title(fs_label, fontsize=10)
        ax.axhline(3.5, color="white", lw=1.5)
    axes[0].set_yticks(range(len(REP_KEYS)))
    axes[0].set_yticklabels([REP_LABEL[r] for r in REP_KEYS], fontsize=8.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("acurácia balanceada média (50 folds)")
    cb.formatter = C.CommaFormatter()
    cb.update_ticks()
    C.save_fig(fig, "grade_acuracia_balanceada.png", comma=False)


def a2_pairwise(rows, n_train, n_test):
    print("\n[A2] Comparação pareada com a imagem original (Nadeau-Bengio + Holm)")
    tests = []
    for fs, _ in FEATURE_SETS:
        for _c, cl in CLASSIFIERS:
            base = {m: metric_matrix(rows, "original", fs, cl, m) for m in ("bal_acc", "auc")}
            for rep in REP_KEYS[1:]:
                for m in ("bal_acc", "auc"):
                    t = mc.corrected_paired_ttest(metric_matrix(rows, rep, fs, cl, m), base[m], n_train, n_test)
                    tests.append({"rep": rep, "fs": fs, "clf": cl, "metrica": m, **t})
    for m in ("bal_acc", "auc"):
        idx = [i for i, t in enumerate(tests) if t["metrica"] == m]
        adj = mc.holm_adjust([tests[i]["p"] for i in idx])
        for i, a in zip(idx, adj):
            tests[i]["p_holm"] = float(a)
    C.write_csv("a2_comparacoes_original.csv", tests)
    return tests


def a2b_effect_by_featureset(rows, n_train, n_test, table_name="tab_efeito_por_conjunto.tex"):
    """
    Efeito médio de trocar a imagem de entrada, por conjunto de features: em
    cada fold, a diferença de acurácia balanceada (representação - original) é
    promediada sobre os cinco classificadores; o teste t corrigido é aplicado a
    essas 50 diferenças e o ajuste de Holm é feito dentro de cada conjunto.
    """
    print("\n[A2b] Efeito da representação por conjunto de features (média dos classificadores)")
    out = {}
    for fs, _ in FEATURE_SETS:
        base = np.mean([metric_matrix(rows, "original", fs, cl, "bal_acc") for _c, cl in CLASSIFIERS], axis=0)
        res = {}
        for rep in REP_KEYS[1:]:
            cur = np.mean([metric_matrix(rows, rep, fs, cl, "bal_acc") for _c, cl in CLASSIFIERS], axis=0)
            res[rep] = mc.corrected_paired_ttest(cur, base, n_train, n_test)
        adj = mc.holm_adjust([res[r]["p"] for r in REP_KEYS[1:]])
        for r, a in zip(REP_KEYS[1:], adj):
            res[r]["p_holm"] = float(a)
        out[fs] = res
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Representação & " + " & ".join(FS_TEX[fs] for fs, _ in FEATURE_SETS) + " \\\\", "\\midrule"]
    for rep in REP_KEYS[1:]:
        cells = []
        for fs, _ in FEATURE_SETS:
            t = out[fs][rep]
            txt = f"{br(t['diff'], 3, sign=True)} ({br_p(t['p_holm'])})"
            cells.append(f"\\textbf{{{txt}}}" if t["p_holm"] < 0.05 else txt)
        lines.append(f"{REP_TEX[rep]} & " + " & ".join(cells) + " \\\\")
        if rep == "verde_clahe":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if table_name:
        C.write_tex(table_name, "\n".join(lines) + "\n", SCRIPT)
    for fs in out:
        for rep in out[fs]:
            t = out[fs][rep]
            print(f"  {fs:8s} {rep:12s} delta {t['diff']:+.3f}  IC [{t['ci_low']:+.3f}; {t['ci_high']:+.3f}]  "
                  f"p {t['p']:.4f}  p_holm {t['p_holm']:.4f}")
    return out


def a3_ranking(summary, metric="bal_acc", tag="pbi"):
    blocks = [(fs, cl) for fs, _ in FEATURE_SETS for _c, cl in CLASSIFIERS]
    M = np.array([[lookup(summary, rep, fs, cl)[f"{metric}_media"] for rep in REP_KEYS] for fs, cl in blocks])
    res = mc.friedman_nemenyi(M)
    return res, M


def a3_cd_figure(res, name, title):
    ranks = res["ranks"]
    order = np.argsort(ranks)
    k = len(ranks)
    fig, ax = plt.subplots(figsize=(8.6, 2.5))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(0.28, 1.08)
    ax.axis("off")
    ax.plot([lo, hi], [0.82, 0.82], color="k", lw=1)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [0.82, 0.86], color="k", lw=1)
        ax.text(t, 0.89, str(t), ha="center", va="bottom", fontsize=9)
    cd = res["cd"]
    ax.plot([lo, lo + cd], [1.0, 1.0], color="#c62828", lw=2.2)
    ax.text(lo + cd + 0.08, 1.0, f"CD = {cd:.2f}".replace(".", ","), ha="left", va="center", fontsize=8.5,
            color="#c62828")
    half = int(np.ceil(k / 2))
    for i, idx in enumerate(order):
        r = ranks[idx]
        left = i < half
        y = 0.70 - 0.115 * (i if left else (k - 1 - i))
        xt = lo - 0.25 if left else hi + 0.25
        ax.plot([r, r], [0.82, y], color=REP_COLOR[REP_KEYS[idx]], lw=1.4)
        ax.plot([r, xt], [y, y], color=REP_COLOR[REP_KEYS[idx]], lw=1.4)
        ax.text(xt + (-0.05 if left else 0.05), y, f"{REP_LABEL[REP_KEYS[idx]]} ({r:.2f})".replace(".", ","),
                ha="right" if left else "left", va="center", fontsize=8.5)
    # Barras ligando configurações não distinguíveis (diferença de posto < CD)
    sorted_r = ranks[order]
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and sorted_r[j + 1] - sorted_r[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for n, (i, j) in enumerate(cliques):
        yb = 0.79 - 0.035 * n
        ax.plot([sorted_r[i] - 0.03, sorted_r[j] + 0.03], [yb, yb], color="k", lw=3.2, solid_capstyle="round")
    ax.set_title(title, fontsize=9.5, loc="left", pad=4)
    C.save_fig(fig, name, comma=False)


def a23_tables(summary, tests, res_bal, res_auc, perm):
    lines = ["\\begin{tabular}{lccccccc}", "\\toprule",
             "\\multirow{2}{*}{Representação} & \\multicolumn{2}{c}{Média da grade} & "
             "\\multicolumn{2}{c}{Posto médio} & \\multicolumn{2}{c}{vs.\\ original (Holm)} & "
             "Permutação \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             " & AB & AUC & AB & AUC & $\\uparrow$ / $\\downarrow$ & $\\Delta$AB médio & $p$ \\\\", "\\midrule"]
    for i, rep in enumerate(REP_KEYS):
        sub = [s for s in summary if s["rep"] == rep and s["fs"] in dict(FEATURE_SETS)]
        mb = np.mean([s["bal_acc_media"] for s in sub])
        ma = np.mean([s["auc_media"] for s in sub])
        if rep == "original":
            cmp_txt, dtxt = "--", "--"
        else:
            tt = [t for t in tests if t["rep"] == rep and t["metrica"] == "bal_acc"]
            up = sum(1 for t in tt if t["p_holm"] < 0.05 and t["diff"] > 0)
            down = sum(1 for t in tt if t["p_holm"] < 0.05 and t["diff"] < 0)
            cmp_txt = f"{up} / {down}"
            dtxt = br(np.mean([t["diff"] for t in tt]), 3, sign=True)
        p = perm.get(rep, {}).get("p", np.nan)
        lines.append(f"{REP_TEX[rep]} & {br(mb, 3)} & {br(ma, 3)} & {br(res_bal['ranks'][i])} & "
                     f"{br(res_auc['ranks'][i])} & {cmp_txt} & {dtxt} & {br_p(p)} \\\\")
        if rep == "verde_clahe":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    C.write_tex("tab_resumo_representacoes.tex", "\n".join(lines) + "\n", SCRIPT)

    # Melhores combinações (seleção otimista, apenas descritiva)
    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "Representação & Combinação & AB [IC 95\\%] & AUC [IC 95\\%] & Sensib. & Especif. \\\\",
             "\\midrule"]
    for rep in REP_KEYS:
        sub = [s for s in summary if s["rep"] == rep and s["fs"] in dict(FEATURE_SETS)]
        s = max(sub, key=lambda x: x["bal_acc_media"])
        lines.append(f"{REP_TEX[rep]} & {FS_TEX[s['fs']]} + {CLF_TEX[s['clf']]} & "
                     f"{br(s['bal_acc_media'], 3)} [{br(s['bal_acc_ic_inf'], 3)}; {br(s['bal_acc_ic_sup'], 3)}] & "
                     f"{br(s['auc_media'], 3)} [{br(s['auc_ic_inf'], 3)}; {br(s['auc_ic_sup'], 3)}] & "
                     f"{br(s['sens_media'], 3)} & {br(s['spec_media'], 3)} \\\\")
        if rep == "verde_clahe":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    C.write_tex("tab_melhores_combinacoes.tex", "\n".join(lines) + "\n", SCRIPT)


# ---------------------------------------------------------------------------
# A4 — Teste de permutação
# ---------------------------------------------------------------------------
def _perm_chunk(X, y, groups, seeds, n_folds, cv_seed):
    fac = ClassifierFactory(ClassifierConfig())
    out = []
    for s in seeds:
        yp = y if s < 0 else np.random.default_rng(int(s)).permutation(y)
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
        vals = []
        for tr, te in cv.split(X, yp, groups):
            m = fac.create_pipeline(CT.LR)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X[tr], yp[tr])
            pred = m.predict(X[te])
            tn, fp, fn, tp = confusion_matrix(yp[te], pred, labels=[0, 1]).ravel()
            vals.append(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)))
        out.append(float(np.mean(vals)))
    return out


def a4_permutation(feats, y, groups, n_perm, jobs, n_folds=5, cv_seed=7):
    print(f"\n[A4] Teste de permutação (RL sobre pixels, {n_perm} permutações por representação)")
    res = {}
    chunk = 25
    t0 = time.time()
    for rep in REP_KEYS:
        X = feats[(rep, "pixels")]
        obs = _perm_chunk(X, y, groups, [-1], n_folds, cv_seed)[0]
        seeds = np.arange(n_perm) + 10_000
        parts = Parallel(n_jobs=jobs, backend="loky")(
            delayed(_perm_chunk)(X, y, groups, seeds[i:i + chunk], n_folds, cv_seed)
            for i in range(0, n_perm, chunk))
        null = np.concatenate(parts)
        res[rep] = {"observado": obs, "nulo_media": float(null.mean()), "nulo_dp": float(null.std()),
                    "nulo_p95": float(np.percentile(null, 95)), "p": mc.permutation_pvalue(obs, null),
                    "n_perm": int(n_perm), "nulo": null}
        print(f"  {rep:12s} observado {obs:.3f}  nulo {null.mean():.3f} ± {null.std():.3f}  p = {res[rep]['p']:.4f}")
    print(f"  permutações em {time.time() - t0:.0f} s")

    fig, axes = plt.subplots(2, 4, figsize=(12.0, 4.6), sharex=True, sharey=True)
    for ax, rep in zip(axes.ravel(), REP_KEYS):
        r = res[rep]
        ax.hist(r["nulo"], bins=30, color="#bdbdbd", edgecolor="white")
        ax.axvline(r["observado"], color=REP_COLOR[rep], lw=2.2)
        ax.set_title(REP_LABEL[rep], fontsize=9.5)
        ptxt = "p < 0,001" if r["p"] < 0.001 else f"p = {r['p']:.3f}".replace(".", ",")
        ax.text(0.03, 0.95, ptxt, transform=ax.transAxes, va="top", fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("acurácia balanceada")
    for ax in axes[:, 0]:
        ax.set_ylabel("permutações")
    fig.tight_layout()
    C.save_fig(fig, "permutacao_pixels_rl.png")
    C.write_csv("a4_permutacao.csv", [{"rep": k, **{kk: vv for kk, vv in v.items() if kk != "nulo"}}
                                      for k, v in res.items()])
    return res


# ---------------------------------------------------------------------------
# A5 — Controles: atalho de cor, descritores manuais e fusão
# ---------------------------------------------------------------------------
def a5_controls(feats, y, splits, jobs, n_train, n_test, main_rows):
    print("\n[A5] Controles: cor global, descritores manuais e fusão")
    feats = dict(feats)
    feats[("fusao_frangi", "ic")] = np.hstack([feats[("original", "ic")], feats[("frangi", "ic")]])
    feats[("fusao_gabor", "ic")] = np.hstack([feats[("original", "ic")], feats[("gabor", "ic")]])
    configs = []
    for ct, cl in CLASSIFIERS:
        configs += [("cor", None, ct, cl), ("manual", None, ct, cl),
                    ("fusao_frangi", "ic", ct, cl), ("fusao_gabor", "ic", ct, cl)]
    rows, oof = run_grid(feats, y, splits, jobs, configs)
    for r in rows:
        if r["rep"] in ("cor", "manual"):
            r["fs"] = "-"
    summary = summarize(rows, n_train, n_test)
    C.write_csv("a5_controles_por_fold.csv", rows)
    C.write_csv("a5_controles_resumo.csv", summary)

    names = {"cor": "Cor e iluminação globais (8 valores)",
             "manual": "Descritores manuais de Frangi (13 valores)",
             "fusao_frangi": "Fusão: original + Frangi (padrão da IC)",
             "fusao_gabor": "Fusão: original + Gabor (padrão da IC)"}
    lines = ["\\begin{tabular}{l" + "c" * len(CLASSIFIERS) + "}", "\\toprule",
             "Entrada & " + " & ".join(CLF_TEX[cl] for _c, cl in CLASSIFIERS) + " \\\\", "\\midrule"]
    ref_rows = []
    for rep, fs in (("original", "ic"), ("frangi", "ic"), ("gabor", "ic")):
        cells = []
        for _c, cl in CLASSIFIERS:
            s = lookup(SUMMARY["_main_summary"], rep, fs, cl)
            cells.append(f"{br(s['bal_acc_media'], 3)} ({br(s['auc_media'], 3)})")
        ref_rows.append(f"Referência: {REP_TEX[rep]} (padrão da IC) & " + " & ".join(cells) + " \\\\")
    lines += ref_rows + ["\\midrule"]
    for key in ("cor", "manual", "fusao_frangi", "fusao_gabor"):
        fs = "-" if key in ("cor", "manual") else "ic"
        cells = []
        for _c, cl in CLASSIFIERS:
            s = lookup(summary, key, fs, cl)
            cells.append(f"{br(s['bal_acc_media'], 3)} ({br(s['auc_media'], 3)})")
        lines.append(f"{names[key]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    C.write_tex("tab_controles.tex", "\n".join(lines) + "\n", SCRIPT)

    # Testes: descritores manuais e fusão contra a melhor referência de cada classificador
    tests = []
    for _c, cl in CLASSIFIERS:
        for key, fs, ref in (("manual", "-", ("frangi", "ic")), ("fusao_frangi", "ic", ("original", "ic")),
                             ("fusao_gabor", "ic", ("original", "ic")), ("cor", "-", ("original", "ic"))):
            a = metric_matrix(rows, key, fs, cl, "bal_acc")
            b = metric_matrix(main_rows, ref[0], ref[1], cl, "bal_acc")
            tests.append({"entrada": key, "referencia": f"{ref[0]}/{ref[1]}", "clf": cl,
                          **mc.corrected_paired_ttest(a, b, n_train, n_test)})
    C.write_csv("a5_controles_testes.csv", tests)
    SUMMARY["controles"] = {
        f"{s['rep']}|{s['clf']}": {"bal_acc": s["bal_acc_media"], "auc": s["auc_media"],
                                   "bal_acc_ic": [s["bal_acc_ic_inf"], s["bal_acc_ic_sup"]]}
        for s in summary}
    SUMMARY["controles_testes"] = tests
    return summary


# ---------------------------------------------------------------------------
# A6 — Robustez fotométrica
# ---------------------------------------------------------------------------
def a6_robustness(ds, feats, y, splits, jobs, n_train, n_test, n_repeats_rob):
    print("\n[A6] Robustez fotométrica (treino original, teste perturbado; RL + padrão da IC)")
    pert_feats = {}
    for kind, _tex in PERTURBATIONS:
        pert_feats[kind] = cached_features(ds.images, jobs, f"pbi_{kind}", perturbation=kind,
                                           with_controls=False)
    tasks = []
    for rep in REP_KEYS:
        X = feats[(rep, "ic")]
        X_tests = {kind: pert_feats[kind][(rep, "ic")] for kind, _ in PERTURBATIONS}
        for r, folds in enumerate(splits[:n_repeats_rob]):
            tasks.append((rep, r, X, X_tests, folds))
    out = Parallel(n_jobs=jobs, backend="loky")(
        delayed(run_repeat)(X, y, folds, CT.LR, X_tests) for (_rep, _r, X, X_tests, folds) in tasks)
    rows = []
    for (rep, r, *_), (base_rows, _oof, extra) in zip(tasks, out):
        base = {br_["fold"]: br_ for br_ in base_rows}
        for e in extra:
            b = base[e["fold"]]
            rows.append({"rep": rep, "repeticao": r, "fold": e["fold"], "condicao": e["condicao"],
                         "bal_acc": e["bal_acc"], "bal_acc_original": b["bal_acc"],
                         "delta_bal_acc": e["bal_acc"] - b["bal_acc"], "auc": e["auc"],
                         "delta_auc": e["auc"] - b["auc"]})
    C.write_csv("a6_robustez_por_fold.csv", rows)
    agg = {}
    for rep in REP_KEYS:
        for kind, _ in PERTURBATIONS:
            d = np.array([r["delta_bal_acc"] for r in rows if r["rep"] == rep and r["condicao"] == kind])
            mean, lo, hi = mc.corrected_mean_ci(d, n_train, n_test)
            agg[(rep, kind)] = (mean, lo, hi)
    SUMMARY["robustez"] = {f"{k[0]}|{k[1]}": v for k, v in agg.items()}
    worst = {rep: min(agg[(rep, k)][0] for k, _ in PERTURBATIONS) for rep in REP_KEYS}
    mean_abs = {rep: float(np.mean([abs(agg[(rep, k)][0]) for k, _ in PERTURBATIONS])) for rep in REP_KEYS}
    SUMMARY["robustez_resumo"] = {"pior_queda": worst, "media_abs": mean_abs}

    lines = ["\\begin{tabular}{l" + "c" * len(REP_KEYS) + "}", "\\toprule",
             "Perturbação & " + " & ".join(
                 {"original": "Orig.", "histograma": "Eq.", "clahe": "CLAHE", "verde_clahe": "V+CL",
                  "frangi": "Frangi", "gabor": "Gabor", "blackhat": "\\textit{B-hat}",
                  "vessel": "F+BH"}[r] for r in REP_KEYS) + " \\\\", "\\midrule"]
    for kind, tex in PERTURBATIONS:
        cells = []
        for rep in REP_KEYS:
            mean, lo, hi = agg[(rep, kind)]
            sig = (hi < 0) or (lo > 0)
            txt = br(mean, 3, sign=True)
            cells.append(f"\\textbf{{{txt}}}" if sig else txt)
        lines.append(f"{tex} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append("Média de $|\\Delta|$ & " + " & ".join(br(mean_abs[r], 3) for r in REP_KEYS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    C.write_tex("tab_robustez.tex", "\n".join(lines) + "\n", SCRIPT)

    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    x = np.arange(len(PERTURBATIONS))
    w = 0.8 / len(REP_KEYS)
    for i, rep in enumerate(REP_KEYS):
        m = np.array([agg[(rep, k)][0] for k, _ in PERTURBATIONS])
        lo = np.array([agg[(rep, k)][1] for k, _ in PERTURBATIONS])
        hi = np.array([agg[(rep, k)][2] for k, _ in PERTURBATIONS])
        ax.bar(x - 0.4 + w * (i + 0.5), m, w, color=REP_COLOR[rep], label=REP_LABEL[rep],
               yerr=[m - lo, hi - m], error_kw={"lw": 0.7, "capsize": 1.2})
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([PERT_LABEL[k] for k, _ in PERTURBATIONS], rotation=20, ha="right")
    ax.set_ylabel("Δ acurácia balanceada")
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    C.save_fig(fig, "robustez_fotometrica.png")
    return agg


# ---------------------------------------------------------------------------
# A7 — Replicação no leiaute personBase
# ---------------------------------------------------------------------------
def a7_replication(ds_pb, summary_pbi, res_pbi, jobs, n_repeats, n_folds):
    print("\n[A7] Replicação no leiaute personBase")
    feats = cached_features(ds_pb.images, jobs, "pb", with_controls=False)
    y, groups = ds_pb.labels, ds_pb.groups
    splits = make_splits(y, groups, n_repeats, n_folds)
    n_test = float(np.mean([len(te) for f in splits for _tr, te in f]))
    n_train = len(y) - n_test
    rows, _oof = run_grid(feats, y, splits, jobs, grid_configs())
    summary = summarize(rows, n_train, n_test)
    C.write_csv("a7_personbase_por_fold.csv", rows)
    C.write_csv("a7_personbase_resumo.csv", summary)
    effect_pb = a2b_effect_by_featureset(rows, n_train, n_test, "tab_efeito_por_conjunto_personbase.tex")
    res, _M = a3_ranking(summary)
    keys = [(rep, fs, cl) for rep in REP_KEYS for fs, _ in FEATURE_SETS for _c, cl in CLASSIFIERS]
    a = np.array([lookup(summary_pbi, *k)["bal_acc_media"] for k in keys])
    b = np.array([lookup(summary, *k)["bal_acc_media"] for k in keys])
    rho, p_rho = stats.spearmanr(a, b)
    rank_rho, rank_p = stats.spearmanr(res_pbi["ranks"], res["ranks"])
    SUMMARY["replicacao"] = {"spearman_configs": float(rho), "p": float(p_rho),
                             "spearman_postos": float(rank_rho), "p_postos": float(rank_p),
                             "efeito_por_conjunto": effect_pb,
                             "friedman": {k: v for k, v in res.items() if k != "ranks"},
                             "postos": dict(zip(REP_KEYS, res["ranks"].tolist())),
                             "media_rep": {rep: float(np.mean([lookup(summary, rep, fs, cl)["bal_acc_media"]
                                                               for fs, _ in FEATURE_SETS for _c, cl in CLASSIFIERS]))
                                           for rep in REP_KEYS},
                             "media_rep_fs": {f"{rep}|{fs}": float(np.mean([lookup(summary, rep, fs, cl)["bal_acc_media"]
                                                                           for _c, cl in CLASSIFIERS]))
                                              for rep in REP_KEYS for fs, _ in FEATURE_SETS}}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    ax = axes[0]
    for rep in REP_KEYS:
        idx = [i for i, k in enumerate(keys) if k[0] == rep]
        ax.scatter(a[idx], b[idx], s=22, color=REP_COLOR[rep], label=REP_LABEL[rep], alpha=0.85,
                   edgecolor="white", lw=0.4)
    lim = [min(a.min(), b.min()) - 0.02, max(a.max(), b.max()) + 0.02]
    ax.plot(lim, lim, color="k", lw=0.8, ls=":")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("acurácia balanceada, personBase_invert")
    ax.set_ylabel("acurácia balanceada, personBase")
    ax.set_title(f"(a) 120 configurações; ρ de Spearman = {rho:.2f}".replace(".", ","), loc="left", fontsize=9.5)
    ax.legend(fontsize=7.2, frameon=False, ncol=2, loc="upper left")
    ax = axes[1]
    xx = np.arange(len(REP_KEYS))
    ax.plot(xx, res_pbi["ranks"], "o-", color="#1565c0", label="personBase_invert")
    ax.plot(xx, res["ranks"], "s--", color="#ef6c00", label="personBase")
    ax.set_xticks(xx)
    ax.set_xticklabels([REP_LABEL[r] for r in REP_KEYS], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("posto médio de Friedman (1 = melhor)")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"(b) postos; ρ de Spearman = {rank_rho:.2f}".replace(".", ","), loc="left", fontsize=9.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    C.save_fig(fig, "replicacao_personbase.png")

    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "\\multirow{2}{*}{Representação} & \\multicolumn{2}{c}{\\texttt{personBase\\_invert}} & "
             "\\multicolumn{2}{c}{\\texttt{personBase}} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             " & AB média & Posto & AB média & Posto \\\\", "\\midrule"]
    for i, rep in enumerate(REP_KEYS):
        ma = np.mean([lookup(summary_pbi, rep, fs, cl)["bal_acc_media"] for fs, _ in FEATURE_SETS
                      for _c, cl in CLASSIFIERS])
        mb = SUMMARY["replicacao"]["media_rep"][rep]
        lines.append(f"{REP_TEX[rep]} & {br(ma, 3)} & {br(res_pbi['ranks'][i])} & {br(mb, 3)} & "
                     f"{br(res['ranks'][i])} \\\\")
        if rep == "verde_clahe":
            lines.append("\\midrule")
    lines += ["\\midrule",
              f"Friedman, $\\chi^2_F$ ($p$) & \\multicolumn{{2}}{{c}}{{{br(res_pbi['chi2'], 1)} ({br_p(res_pbi['p_chi2'])})}} & "
              f"\\multicolumn{{2}}{{c}}{{{br(res['chi2'], 1)} ({br_p(res['p_chi2'])})}} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    C.write_tex("tab_replicacao.tex", "\n".join(lines) + "\n", SCRIPT)
    return summary, res


# ---------------------------------------------------------------------------
# A8 — Mapas de diferença entre classes
# ---------------------------------------------------------------------------
def bh_fdr(p):
    p = np.asarray(p)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def a8_maps(feats, y):
    print("\n[A8] Mapas de d de Cohen por pixel (diabetes - controle)")
    keys = ["original", "verde_clahe", "frangi", "gabor", "blackhat", "vessel"]
    fig, axes = plt.subplots(len(keys), 1, figsize=(9.6, 1.45 * len(keys) + 0.4), sharex=True)
    info = {}
    lim = 0.0
    maps = {}
    for key in keys:
        X = feats[(key, "pixels")].reshape(len(y), 50, 180).astype(np.float64)
        a, b = X[y == 1], X[y == 0]
        sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
        d = (a.mean(0) - b.mean(0)) / np.maximum(sp, 1e-9)
        _t, p = stats.ttest_ind(a, b, axis=0, equal_var=False)
        p = np.nan_to_num(p, nan=1.0)
        q = bh_fdr(p.ravel()).reshape(p.shape)
        maps[key] = (d, q)
        lim = max(lim, float(np.nanpercentile(np.abs(d), 99.5)))
        rows = np.array_split(np.arange(50), 3)
        info[key] = {"d_abs_mediano": float(np.median(np.abs(d))), "d_abs_max": float(np.nanmax(np.abs(d))),
                     "frac_fdr05": float((q < 0.05).mean()),
                     "d_abs_terços": [float(np.mean(np.abs(d[r]))) for r in rows]}
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    for ax, key in zip(axes, keys):
        d, q = maps[key]
        im = ax.imshow(d, cmap="RdBu_r", norm=norm, aspect="auto", extent=[0, 360, 1, 0])
        if (q < 0.05).any():
            ax.contour(np.linspace(0, 360, 180), np.linspace(0, 1, 50), (q < 0.05).astype(float), levels=[0.5],
                       colors="k", linewidths=0.6)
        frac = 100 * info[key]["frac_fdr05"]
        ax.set_title(f"{REP_LABEL[key]}: {frac:.1f}% dos pixels com q < 0,05".replace(".", ",", 1)
                     if frac else f"{REP_LABEL[key]}: nenhum pixel com q < 0,05", loc="left", fontsize=9)
        ax.set_ylabel("r norm.", fontsize=8)
    axes[-1].set_xlabel("ângulo θ da faixa normalizada (graus)")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("d de Cohen (diabetes − controle)")
    C.save_fig(fig, "mapas_d_cohen.png")
    SUMMARY["mapas_d"] = info


# ---------------------------------------------------------------------------
# A9 — Recorte radial: estroma versus bordas
# ---------------------------------------------------------------------------
# O vetor de pixels tem 50 linhas, da borda pupilar (r = 0) ao limbo (r = 1).
# As linhas 5 a 42 cobrem r de cerca de 0,10 a 0,86: o estroma sem as duas bordas.
STROMA_ROWS = (5, 43)
REGION_TEX = {"faixa": "Faixa inteira", "estroma": "Só estroma", "bordas": "Só bordas"}


def a9_radial_crop(feats, y, splits, jobs, n_train, n_test):
    print("\n[A9] Recorte radial: estroma (r de 0,10 a 0,86) versus bordas pupilar e limbar")
    rows_idx = np.arange(50)
    stroma = (rows_idx >= STROMA_ROWS[0]) & (rows_idx < STROMA_ROWS[1])
    crop = {}
    for rep in REP_KEYS:
        P = feats[(rep, "pixels")].reshape(len(y), 50, 180)
        crop[(rep, "estroma")] = P[:, stroma, :].reshape(len(y), -1)
        crop[(rep, "bordas")] = P[:, ~stroma, :].reshape(len(y), -1)
    configs = [(rep, region, ct, cl) for rep in REP_KEYS for region in ("estroma", "bordas")
               for ct, cl in CLASSIFIERS]
    rows, _oof = run_grid(crop, y, splits, jobs, configs)
    summary = summarize(rows, n_train, n_test)
    C.write_csv("a9_recorte_por_fold.csv", rows)
    C.write_csv("a9_recorte_resumo.csv", summary)

    res = {}
    for region in ("estroma", "bordas"):
        base = np.mean([metric_matrix(rows, "original", region, cl, "bal_acc") for _c, cl in CLASSIFIERS], axis=0)
        tests = {}
        for rep in REP_KEYS[1:]:
            cur = np.mean([metric_matrix(rows, rep, region, cl, "bal_acc") for _c, cl in CLASSIFIERS], axis=0)
            tests[rep] = mc.corrected_paired_ttest(cur, base, n_train, n_test)
        adj = mc.holm_adjust([tests[r]["p"] for r in REP_KEYS[1:]])
        for r, a in zip(REP_KEYS[1:], adj):
            tests[r]["p_holm"] = float(a)
        means = {rep: float(np.mean([lookup(summary, rep, region, cl)["bal_acc_media"] for _c, cl in CLASSIFIERS]))
                 for rep in REP_KEYS}
        wins = sum(lookup(summary, rep, region, cl)["bal_acc_media"] >
                   lookup(summary, "original", region, cl)["bal_acc_media"]
                   for rep in FILTER_KEYS for _c, cl in CLASSIFIERS)
        res[region] = {"media": means, "efeito": tests, "vitorias": int(wins)}
        print(f"  {region}: " + ", ".join(f"{r} {means[r]:.3f}" for r in REP_KEYS) + f"; vitórias {wins}/20")
    full = {rep: SUMMARY["media_por_rep_fs"][f"{rep}|pixels"] for rep in REP_KEYS}
    res["faixa"] = {"media": full}
    SUMMARY["recorte"] = res

    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             "\\multirow{2}{*}{Representação} & Faixa inteira & \\multicolumn{2}{c}{Só estroma} & "
             "\\multicolumn{2}{c}{Só bordas} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
             " & AB & AB & $\\Delta$ ($p$ Holm) & AB & $\\Delta$ ($p$ Holm) \\\\", "\\midrule"]
    for rep in REP_KEYS:
        cells = [br(full[rep], 3)]
        for region in ("estroma", "bordas"):
            cells.append(br(res[region]["media"][rep], 3))
            if rep == "original":
                cells.append("--")
            else:
                t = res[region]["efeito"][rep]
                txt = f"{br(t['diff'], 3, sign=True)} ({br_p(t['p_holm'])})"
                cells.append(f"\\textbf{{{txt}}}" if t["p_holm"] < 0.05 else txt)
        lines.append(f"{REP_TEX[rep]} & " + " & ".join(cells) + " \\\\")
        if rep == "verde_clahe":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    C.write_tex("tab_recorte_radial.tex", "\n".join(lines) + "\n", SCRIPT)

    fig, ax = plt.subplots(figsize=(10.0, 3.7))
    x = np.arange(len(REP_KEYS))
    w = 0.26
    for i, (region, col) in enumerate((("faixa", "#9e9e9e"), ("estroma", "#1565c0"), ("bordas", "#ef6c00"))):
        vals = [res[region]["media"][r] for r in REP_KEYS]
        ax.bar(x - w + i * w, vals, w, color=col, label=REGION_TEX[region])
    ax.set_xticks(x)
    ax.set_xticklabels([REP_LABEL[r] for r in REP_KEYS], rotation=25, ha="right")
    ax.set_ylabel("AB média (pixels, 5 classificadores)")
    lo = min(min(res[r]["media"].values()) for r in ("faixa", "estroma", "bordas"))
    ax.set_ylim(max(0.4, lo - 0.05), None)
    ax.axvline(3.5, color="k", lw=0.6, ls=":")
    ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    C.save_fig(fig, "recorte_radial.png")
    return res


# ---------------------------------------------------------------------------
# A10 — Desfazer o espelhamento do leiaute personBase_invert
# ---------------------------------------------------------------------------
def unmirror(images, mirrored):
    """Inverte o espelhamento do leiaute (inversão angular + deslocamento de 360 colunas)."""
    out = images.copy()
    for i in np.flatnonzero(mirrored):
        out[i] = np.roll(images[i], -images.shape[2] // 2, axis=1)[:, ::-1]
    return out


def a10_unmirror(ds, feats, y, splits, jobs, n_train, n_test):
    print("\n[A10] Faixas do olho direito devolvidas à orientação original (pixels)")
    images = unmirror(ds.images, ds.mirrored)
    feats_u = cached_features(images, jobs, "pbi_desespelhado", with_controls=False)
    configs = [(rep, "pixels", ct, cl) for rep in REP_KEYS for ct, cl in CLASSIFIERS]
    rows, _oof = run_grid(feats_u, y, splits, jobs, configs)
    summary = summarize(rows, n_train, n_test)
    C.write_csv("a10_desespelhado_por_fold.csv", rows)
    C.write_csv("a10_desespelhado_resumo.csv", summary)
    means = {rep: float(np.mean([lookup(summary, rep, "pixels", cl)["bal_acc_media"] for _c, cl in CLASSIFIERS]))
             for rep in REP_KEYS}
    base = np.mean([metric_matrix(rows, "original", "pixels", cl, "bal_acc") for _c, cl in CLASSIFIERS], axis=0)
    tests = {}
    for rep in REP_KEYS[1:]:
        cur = np.mean([metric_matrix(rows, rep, "pixels", cl, "bal_acc") for _c, cl in CLASSIFIERS], axis=0)
        tests[rep] = mc.corrected_paired_ttest(cur, base, n_train, n_test)
    adj = mc.holm_adjust([tests[r]["p"] for r in REP_KEYS[1:]])
    for r, a in zip(REP_KEYS[1:], adj):
        tests[r]["p_holm"] = float(a)
    wins = sum(lookup(summary, rep, "pixels", cl)["bal_acc_media"] >
               lookup(summary, "original", "pixels", cl)["bal_acc_media"] for rep in FILTER_KEYS for _c, cl in CLASSIFIERS)
    # Mudança de cada representação ao desfazer o espelhamento (mesmos folds)
    change = {}
    main_rows = SUMMARY.get("_main_rows")
    for rep in REP_KEYS:
        after = np.mean([metric_matrix(rows, rep, "pixels", cl, "bal_acc") for _c, cl in CLASSIFIERS], axis=0)
        if main_rows is not None:
            before = np.mean([metric_matrix(main_rows, rep, "pixels", cl, "bal_acc") for _c, cl in CLASSIFIERS], axis=0)
            change[rep] = mc.corrected_paired_ttest(after, before, n_train, n_test)
    SUMMARY["desespelhado"] = {"media": means, "efeito": tests, "vitorias": int(wins), "mudanca": change}
    before_means = {rep: SUMMARY["media_por_rep_fs"][f"{rep}|pixels"] for rep in REP_KEYS}
    pb_means = {rep: SUMMARY["replicacao"]["media_rep_fs"][f"{rep}|pixels"] for rep in REP_KEYS}
    print("  " + ", ".join(f"{r} {before_means[r]:.3f}->{means[r]:.3f}" for r in REP_KEYS) + f"; vitórias {wins}/20")

    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "\\multirow{2}{*}{Representação} & \\multicolumn{3}{c}{AB média com pixels} & "
             "\\multirow{2}{*}{\\makecell{$\\Delta$ ao desfazer\\\\o espelhamento ($p$)}} \\\\",
             "\\cmidrule(lr){2-4}",
             " & \\makecell{\\texttt{personBase\\_invert}\\\\(espelhado)} & "
             "\\makecell{\\texttt{personBase\\_invert}\\\\(desespelhado)} & \\texttt{personBase} & \\\\", "\\midrule"]
    for rep in REP_KEYS:
        ch = change.get(rep)
        ch_txt = f"{br(ch['diff'], 3, sign=True)} ({br_p(ch['p'])})" if ch else "--"
        lines.append(f"{REP_TEX[rep]} & {br(before_means[rep], 3)} & {br(means[rep], 3)} & {br(pb_means[rep], 3)} & "
                     f"{ch_txt} \\\\")
        if rep == "verde_clahe":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    C.write_tex("tab_desespelhado.tex", "\n".join(lines) + "\n", SCRIPT)

    fig, ax = plt.subplots(figsize=(10.0, 3.7))
    x = np.arange(len(REP_KEYS))
    w = 0.26
    for i, (vals, lab, col) in enumerate(((before_means, "personBase_invert (espelhado)", "#9e9e9e"),
                                          (means, "personBase_invert (desespelhado)", "#1565c0"),
                                          (pb_means, "personBase", "#ef6c00"))):
        ax.bar(x - w + i * w, [vals[r] for r in REP_KEYS], w, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([REP_LABEL[r] for r in REP_KEYS], rotation=25, ha="right")
    ax.set_ylabel("AB média (pixels, 5 classificadores)")
    lo = min(min(before_means.values()), min(means.values()), min(pb_means.values()))
    ax.set_ylim(max(0.4, lo - 0.05), None)
    ax.axvline(3.5, color="k", lw=0.6, ls=":")
    ax.legend(frameon=False, ncol=3, fontsize=8.2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    C.save_fig(fig, "desespelhamento.png")
    return SUMMARY["desespelhado"]


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------
def roc_figure(oof, y, summary, n_repeats):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    grid = np.linspace(0, 1, 201)
    for ax, mode in zip(axes, ("fixo", "melhor")):
        for rep in REP_KEYS:
            if mode == "fixo":
                fs, cl = "ic", "RL"
            else:
                s = max([s for s in summary if s["rep"] == rep and s["fs"] in dict(FEATURE_SETS)],
                        key=lambda x: x["bal_acc_media"])
                fs, cl = s["fs"], s["clf"]
            tprs, aucs = [], []
            for r in range(n_repeats):
                sc = oof[(rep, fs, cl, r)]
                fpr, tpr, _ = roc_curve(y, sc)
                tprs.append(np.interp(grid, fpr, tpr))
                aucs.append(roc_auc_score(y, sc))
            ax.plot(grid, np.mean(tprs, axis=0), color=REP_COLOR[rep], lw=1.6,
                    label=f"{REP_LABEL[rep]} ({np.mean(aucs):.3f})".replace(".", ","))
        ax.plot([0, 1], [0, 1], color="k", lw=0.7, ls=":")
        ax.set_xlabel("1 − especificidade")
        ax.set_ylabel("sensibilidade")
        ax.set_title("(a) RL + conjunto padrão da IC (pré-especificado)" if mode == "fixo"
                     else "(b) melhor combinação de cada representação", loc="left", fontsize=9.5)
        ax.legend(fontsize=7.4, frameon=False, loc="lower right", title="AUC agregada por repetição",
                  title_fontsize=7.4)
        ax.set_aspect("equal")
    fig.tight_layout()
    C.save_fig(fig, "roc_representacoes.png")


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="Data.zip ou pasta Data/Data da base legada")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--robust-repeats", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--quick", action="store_true", help="execução reduzida para conferência")
    ap.add_argument("--skip", nargs="*", default=[], help="partes a pular (A4 A6 A7 A8 A9 M)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="executa só as partes indicadas (A0, A9), reaproveitando o cache de features e "
                         "acrescentando os resultados ao resumo_vascular.json existente")
    args = ap.parse_args()
    if args.quick:
        args.repeats, args.permutations, args.robust_repeats = 2, 40, 1

    C.ensure_dirs()
    t_start = time.time()
    source = resolve_legacy_source(args.data)
    print(f"Base legada: {source}")
    ds = load_legacy_layout(source, "personBase_invert")
    ds_pb = load_legacy_layout(source, "personBase")
    y, groups = ds.labels, ds.groups
    print(f"personBase_invert: {len(ds)} imagens únicas ({ds.n_control} controle, {ds.n_diabetic} diabetes)")

    if args.only is not None:
        import json
        with open(os.path.join(C.OUT_DIR, "resumo_vascular.json"), encoding="utf-8") as fh:
            SUMMARY.update(json.load(fh))
        protocol = dict(SUMMARY["protocolo"])
        feats = cached_features(ds.images, args.jobs, "pbi")
        splits = make_splits(y, groups, args.repeats, args.folds)
        n_test = float(np.mean([len(te) for f in splits for _tr, te in f]))
        n_train = len(y) - n_test
        part_times = SUMMARY.setdefault("tempo_partes_s", {})
        if "A0" in args.only:
            a0_audit(ds, ds_pb, source)
        if "A9" in args.only:
            t_part = time.time()
            a9_radial_crop(feats, y, splits, args.jobs, n_train, n_test)
            part_times["A9"] = time.time() - t_part
        if "A10" in args.only:
            import csv
            t_part = time.time()
            with open(os.path.join(C.OUT_DIR, "a1_grade_por_fold.csv"), encoding="utf-8") as fh:
                SUMMARY["_main_rows"] = list(csv.DictReader(fh))
            a10_unmirror(ds, feats, y, splits, args.jobs, n_train, n_test)
            part_times["A10"] = time.time() - t_part
        SUMMARY["protocolo"] = protocol
        C.write_json("resumo_vascular.json", {k: v for k, v in SUMMARY.items() if not k.startswith("_")})
        print(f"\nPartes {args.only} concluídas em {time.time() - t_start:.0f} s.")
        return
    SUMMARY["protocolo"] = {"repeticoes": args.repeats, "folds": args.folds, "permutacoes": args.permutations,
                            "semente_cv": 2026, "classificadores": [c for _t, c in CLASSIFIERS],
                            "conjuntos": [f for f, _ in FEATURE_SETS], "representacoes": REP_KEYS,
                            "ambiente": C.environment(), "rapido": bool(args.quick)}

    a0_audit(ds, ds_pb, source)
    if "M" not in args.skip:
        m_representations_figure()

    print("\n[A1] Grade principal")
    feats = cached_features(ds.images, args.jobs, "pbi")
    splits = make_splits(y, groups, args.repeats, args.folds)
    n_test = float(np.mean([len(te) for f in splits for _tr, te in f]))
    n_train = len(y) - n_test
    SUMMARY["protocolo"].update({"n_treino_medio": n_train, "n_teste_medio": n_test})
    rows, oof = run_grid(feats, y, splits, args.jobs, grid_configs())
    summary = summarize(rows, n_train, n_test)
    SUMMARY["_main_summary"] = summary
    SUMMARY["_main_rows"] = rows
    C.write_csv("a1_grade_por_fold.csv", rows)
    C.write_csv("a1_grade_resumo.csv", summary)
    np.savez_compressed(os.path.join(C.OUT_DIR, "a1_escores_fora_do_fold.npz"),
                        **{f"{k[0]}__{k[1]}__{k[2]}__r{k[3]}": v for k, v in oof.items()}, y=y)
    a1_grid_tables(summary, "personBase\\_invert")
    a1_heatmap(summary)
    roc_figure(oof, y, summary, args.repeats)

    tests = a2_pairwise(rows, n_train, n_test)
    SUMMARY["efeito_por_conjunto"] = a2b_effect_by_featureset(rows, n_train, n_test)
    res_bal, _Mb = a3_ranking(summary, "bal_acc")
    res_auc, _Ma = a3_ranking(summary, "auc")
    a3_cd_figure(res_bal, "diferenca_critica_ab.png",
                 "Posto médio por acurácia balanceada (15 blocos = 3 conjuntos × 5 classificadores)")
    a3_cd_figure(res_auc, "diferenca_critica_auc.png",
                 "Posto médio por AUC (15 blocos = 3 conjuntos × 5 classificadores)")
    SUMMARY["friedman_ab"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res_bal.items()}
    SUMMARY["friedman_auc"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res_auc.items()}

    perm = {}
    if "A4" not in args.skip:
        perm = a4_permutation(feats, y, groups, args.permutations, args.jobs, args.folds)
        SUMMARY["permutacao"] = {k: {kk: vv for kk, vv in v.items() if kk != "nulo"} for k, v in perm.items()}
    a23_tables(summary, tests, res_bal, res_auc, perm)

    a5_controls(feats, y, splits, args.jobs, n_train, n_test, rows)
    if "A6" not in args.skip:
        a6_robustness(ds, feats, y, splits, args.jobs, n_train, n_test, args.robust_repeats)
    if "A7" not in args.skip:
        a7_replication(ds_pb, summary, res_bal, args.jobs, args.repeats, args.folds)
    if "A8" not in args.skip:
        a8_maps(feats, y)

    # Resumo numérico para o texto
    best = max([s for s in summary if s["fs"] in dict(FEATURE_SETS)], key=lambda s: s["bal_acc_media"])
    SUMMARY["melhor_global"] = best
    SUMMARY["media_por_rep"] = {rep: {"bal_acc": float(np.mean([lookup(summary, rep, fs, cl)["bal_acc_media"]
                                                                for fs, _ in FEATURE_SETS for _c, cl in CLASSIFIERS])),
                                      "auc": float(np.mean([lookup(summary, rep, fs, cl)["auc_media"]
                                                            for fs, _ in FEATURE_SETS for _c, cl in CLASSIFIERS]))}
                                for rep in REP_KEYS}
    SUMMARY["media_por_rep_fs"] = {f"{rep}|{fs}": float(np.mean([lookup(summary, rep, fs, cl)["bal_acc_media"]
                                                                 for _c, cl in CLASSIFIERS]))
                                   for rep in REP_KEYS for fs, _ in FEATURE_SETS}
    tests_bal = [t for t in tests if t["metrica"] == "bal_acc"]
    SUMMARY["comparacoes_resumo"] = {
        rep: {"melhores": sum(1 for t in tests_bal if t["rep"] == rep and t["p_holm"] < 0.05 and t["diff"] > 0),
              "piores": sum(1 for t in tests_bal if t["rep"] == rep and t["p_holm"] < 0.05 and t["diff"] < 0),
              "delta_medio": float(np.mean([t["diff"] for t in tests_bal if t["rep"] == rep])),
              "p_bruto_min": float(min(t["p"] for t in tests_bal if t["rep"] == rep))}
        for rep in REP_KEYS[1:]}
    if "A9" not in args.skip:
        a9_radial_crop(feats, y, splits, args.jobs, n_train, n_test)
    if "A10" not in args.skip and "replicacao" in SUMMARY:
        a10_unmirror(ds, feats, y, splits, args.jobs, n_train, n_test)
    SUMMARY["tempo_total_s"] = time.time() - t_start
    SUMMARY["tempo_partes_s"] = {}
    out = {k: v for k, v in SUMMARY.items() if not k.startswith("_")}
    C.write_json("resumo_vascular.json", out)
    print(f"\nConcluído em {time.time() - t_start:.0f} s. Resumo em results/pgc3/resumo_vascular.json")


if __name__ == "__main__":
    main()

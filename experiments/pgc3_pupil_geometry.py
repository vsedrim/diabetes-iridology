"""
PGC III, parte B: a geometria da pupila precisa de features explícitas?
======================================================================
Pergunta do orientador: para a pupila (diâmetro e espessura), será
necessário criar features manuais ou o pipeline só precisaria da foto da íris?

  B1. Invariância da normalização rubber sheet à dilatação da pupila: a
      fotografia pública é reamostrada com pupilas de tamanhos diferentes
      (modelo linear do rubber sheet e um modelo não linear ilustrativo) e as
      faixas normalizadas são comparadas.
  B2. Coorte sintética 2 x 2 (sinal na geometria x sinal na textura), com
      verdade de referência: o pipeline da IC sobre a faixa, a geometria
      medida automaticamente na foto, a fusão das duas e a geometria
      verdadeira (oráculo).
  B3. Erro de medição de rho na coorte sintética: detector por limiar e Hough
      da demonstração contra o detector starburst com refinamento integro-diferencial.

As imagens sintéticas não representam pessoas. A fotografia usada em B1 é a
de Petr Novák (CC BY-SA 2.5); as figuras derivadas dela seguem a mesma licença.

Uso:
    python experiments/pgc3_pupil_geometry.py
    python experiments/pgc3_pupil_geometry.py --quick
"""

import argparse
import os
import sys
import time
import warnings

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from skimage.metrics import structural_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pgc3_common as C  # noqa: E402
from pgc3_common import br  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

import demo_vascularization_pupil as demo  # noqa: E402
from src import model_comparison as mc  # noqa: E402
from src.config import ColorChannel, FeatureType as FT, PhotometricTransform as PT  # noqa: E402
from src.feature_extraction import FeatureExtractor  # noqa: E402
from src.fusion import GEOMETRY_FEATURES, block_classifier, build_fusion_classifier  # noqa: E402
from src.preprocessing import ImagePreprocessor  # noqa: E402
from src.segmentation import PupilValidator, detect_eye_circles  # noqa: E402
from src.synthetic_eye import EyeGeometry, sample_synthetic_eye, simulate_pupil_dilation  # noqa: E402

SCRIPT = "pgc3_pupil_geometry.py"
SUMMARY = {}

SCENARIOS = [
    ("nulo", "Sem sinal", False, False),
    ("geometria", "Sinal só na geometria", True, False),
    ("textura", "Sinal só na textura", False, True),
    ("ambos", "Sinal na geometria e na textura", True, True),
]
RHO_CONTROL, RHO_CASE, RHO_SD = 0.40, 0.34, 0.06
# Efeito de textura: contraste médio das fibras radiais 10% menor nos controles e
# 10% maior nos casos (calibrado para a faixa ter AUC próxima à da geometria).
FIBER_CONTROL, FIBER_CASE, FIBER_NULL = 0.9, 1.1, 1.0
CRYPT_RATE = 7.0
MODELS = [("faixa", "Faixa normalizada (pipeline da IC)", "#616161"),
          ("geometria", "Geometria medida na foto", "#1565c0"),
          ("fusao", "Fusão (empilhamento)", "#2e7d32"),
          ("oraculo", "Geometria verdadeira (oráculo)", "#9e9e9e")]
GEOM_NAMES = list(GEOMETRY_FEATURES)


# ---------------------------------------------------------------------------
# B1 — Invariância da normalização
# ---------------------------------------------------------------------------
def strip_gray(img, det, n_r=201, n_theta=720):
    s = C.rubber_sheet_circles(img, det, n_r=n_r, n_theta=n_theta)
    return cv2.cvtColor(s, cv2.COLOR_BGR2GRAY) if s.ndim == 3 else s


def similarity(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    ssim = float(structural_similarity(a, b, data_range=255.0))
    return r, ssim


def b1_invariance():
    print("\n[B1] Invariância da faixa normalizada à dilatação da pupila")
    rhos = np.round(np.arange(0.22, 0.621, 0.02), 3)

    # (1) Fotografia pública. O limbo deste olho fotografado fora do eixo é elíptico;
    # o círculo do limbo do detector por Hough da demonstração é mantido fixo, como na simulação.
    img = demo.resize_max(demo.load_image("iris_petr_novak.jpg"), 768)
    det_demo = demo.detect_pupil_iris(img)
    det_new = detect_eye_circles(img)
    iris_fixed = {"iris_center": det_demo["iris_center"], "iris_radius": det_demo["iris_radius"]}
    geom0 = EyeGeometry(det_new["pupil_center"], det_new["pupil_radius"], iris_fixed["iris_center"],
                        iris_fixed["iris_radius"])
    ref_strip = strip_gray(img, geom0.as_detection())
    real_rows, examples = [], {}
    for rho in rhos:
        warped, g = simulate_pupil_dilation(img, geom0, float(rho))
        d = detect_eye_circles(warped)
        meas = {"pupil_center": d["pupil_center"], "pupil_radius": d["pupil_radius"], **iris_fixed}
        s_meas = strip_gray(warped, meas)
        r_m, ssim_m = similarity(ref_strip, s_meas)
        real_rows.append({"fonte": "fotografia", "modelo": "linear", "rho": float(rho),
                          "rho_medido": d["pupil_radius"] / iris_fixed["iris_radius"],
                          "r_geometria_medida": r_m, "ssim_geometria_medida": ssim_m})
        if any(abs(rho - x) < 1e-6 for x in (0.24, 0.5)):
            examples[float(rho)] = (warped, s_meas)
    examples[round(geom0.rho, 3)] = (img, ref_strip)

    # (2) Olho sintético com geometria conhecida: modelos linear e não linear.
    rng = np.random.default_rng(11)
    syn, gs0 = sample_synthetic_eye(rng, 0.40, 7.0)
    syn_ref = strip_gray(syn, gs0.as_detection())
    syn_rows = []
    for kappa, model in ((0.0, "linear"), (1.0, "nao_linear")):
        for rho in rhos:
            warped, g = simulate_pupil_dilation(syn, gs0, float(rho), kappa=kappa)
            d = detect_eye_circles(warped)
            r_t, ssim_t = similarity(syn_ref, strip_gray(warped, g.as_detection()))
            r_m, ssim_m = similarity(syn_ref, strip_gray(warped, d))
            syn_rows.append({"fonte": "sintetico", "modelo": model, "rho": float(rho),
                             "rho_medido": d["pupil_radius"] / d["iris_radius"],
                             "r_geometria_verdadeira": r_t, "ssim_geometria_verdadeira": ssim_t,
                             "r_geometria_medida": r_m, "ssim_geometria_medida": ssim_m})
    # Referência: similaridade entre faixas de íris sintéticas diferentes (mesmo rho).
    others = []
    for k in range(12):
        o, go = sample_synthetic_eye(np.random.default_rng(500 + k), 0.40, 7.0)
        others.append(similarity(syn_ref, strip_gray(o, go.as_detection())))
    ssim_other = float(np.mean([s for _r, s in others]))
    r_other = float(np.mean([r for r, _s in others]))
    C.write_csv("b1_invariancia.csv", real_rows + syn_rows)

    lin = [r for r in syn_rows if r["modelo"] == "linear"]
    nl = [r for r in syn_rows if r["modelo"] == "nao_linear"]
    err_real = np.array([r["rho_medido"] - r["rho"] for r in real_rows])
    err_syn = np.array([r["rho_medido"] - r["rho"] for r in lin])
    SUMMARY["b1"] = {
        "foto_rho_referencia": geom0.rho, "foto_raio_pupila": geom0.pupil_radius,
        "foto_raio_iris": geom0.iris_radius,
        "foto_ssim_min": float(min(r["ssim_geometria_medida"] for r in real_rows)),
        "foto_r_min": float(min(r["r_geometria_medida"] for r in real_rows)),
        "foto_rho_mae": float(np.mean(np.abs(err_real))), "foto_rho_erro_max": float(np.max(np.abs(err_real))),
        "sint_linear_ssim_min": float(min(r["ssim_geometria_verdadeira"] for r in lin)),
        "sint_linear_r_min": float(min(r["r_geometria_verdadeira"] for r in lin)),
        "sint_linear_ssim_medida_min": float(min(r["ssim_geometria_medida"] for r in lin)),
        "sint_nao_linear_ssim_min": float(min(r["ssim_geometria_verdadeira"] for r in nl)),
        "sint_nao_linear_r_min": float(min(r["r_geometria_verdadeira"] for r in nl)),
        "sint_rho_mae": float(np.mean(np.abs(err_syn))), "sint_rho_erro_max": float(np.max(np.abs(err_syn))),
        "outras_iris_ssim": ssim_other, "outras_iris_r": r_other, "faixa_rho": [float(rhos.min()), float(rhos.max())],
    }

    keys = sorted(examples)
    fig = plt.figure(figsize=(11.0, 6.6))
    gsp = fig.add_gridspec(3, 3, height_ratios=[1.9, 0.62, 1.7], hspace=0.42, wspace=0.28)
    for j, rho in enumerate(keys):
        im, s = examples[rho]
        ax = fig.add_subplot(gsp[0, j])
        cx, cy = map(int, iris_fixed["iris_center"])
        R = int(iris_fixed["iris_radius"] * 1.12)
        crop = im[max(cy - R, 0):cy + R, max(cx - R, 0):cx + R]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tag = " (original)" if abs(rho - round(geom0.rho, 3)) < 1e-6 else " (simulada)"
        ax.set_title(f"(a{j + 1}) ρ = {rho:.2f}{tag}".replace(".", ","), loc="left", fontsize=9.5)
        ax.axis("off")
        ax2 = fig.add_subplot(gsp[1, j])
        ax2.imshow(s, cmap="gray", aspect="auto", vmin=0, vmax=255)
        ax2.set_title(f"(b{j + 1}) faixa normalizada", loc="left", fontsize=9)
        ax2.axis("off")
    ax = fig.add_subplot(gsp[2, :2])
    ax.plot([r["rho"] for r in lin], [r["ssim_geometria_verdadeira"] for r in lin], "-", color="#1565c0", lw=1.8,
            label="olho sintético, modelo linear")
    ax.plot([r["rho"] for r in nl], [r["ssim_geometria_verdadeira"] for r in nl], "--", color="#1565c0", lw=1.6,
            label="olho sintético, modelo não linear")
    ax.plot([r["rho"] for r in real_rows], [r["ssim_geometria_medida"] for r in real_rows], "-o", ms=2.5,
            color="#2e7d32", lw=1.4, label="fotografia, modelo linear, pupila medida")
    ax.axhline(ssim_other, color="#c62828", lw=1.0, ls=":", label="faixas de íris sintéticas diferentes")
    ax.set_xlabel("razão pupila/íris ρ simulada")
    ax.set_ylabel("SSIM com a faixa de referência")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=7.4, frameon=False, loc="lower left")
    ax.set_title("(c) a faixa praticamente não muda quando a pupila muda", loc="left", fontsize=9.5)
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gsp[2, 2])
    ax.plot([r["rho"] for r in lin], [r["rho_medido"] for r in lin], "o", ms=3, color="#1565c0", label="sintético")
    ax.plot([r["rho"] for r in real_rows], [r["rho_medido"] for r in real_rows], "s", ms=3, color="#2e7d32",
            label="fotografia")
    ax.plot([0.2, 0.65], [0.2, 0.65], color="k", lw=0.7, ls=":")
    ax.set_xlabel("ρ simulado")
    ax.set_ylabel("ρ medido na imagem")
    ax.legend(fontsize=7.4, frameon=False)
    ax.set_title("(d) ρ medido automaticamente", loc="left", fontsize=9.5)
    ax.grid(alpha=0.3)
    C.save_fig(fig, "invariancia_rubber_sheet.png")
    b = SUMMARY["b1"]
    print(f"  SSIM mínimo: sintético linear {b['sint_linear_ssim_min']:.3f}, não linear "
          f"{b['sint_nao_linear_ssim_min']:.3f}, fotografia {b['foto_ssim_min']:.3f}; outras íris {ssim_other:.3f}")
    print(f"  MAE de rho: sintético {b['sint_rho_mae']:.4f}, fotografia {b['foto_rho_mae']:.4f}")


# ---------------------------------------------------------------------------
# B2 — Coorte sintética 2 x 2
# ---------------------------------------------------------------------------
_TOOLS = {}


def _tools():
    if "pre" not in _TOOLS:
        _TOOLS["pre"] = ImagePreprocessor()
        _TOOLS["fe"] = FeatureExtractor()
        _TOOLS["val"] = PupilValidator()
    return _TOOLS["pre"], _TOOLS["fe"], _TOOLS["val"]


def synth_subject(seed, label, geom_signal, tex_signal):
    """Gera um olho sintético, detecta os círculos e extrai features da faixa e da geometria."""
    pre, fe, val = _tools()
    rng = np.random.default_rng(seed)
    mu = RHO_CASE if (geom_signal and label == 1) else RHO_CONTROL
    rho = float(np.clip(rng.normal(mu, RHO_SD), 0.15, 0.75))
    if tex_signal:
        fiber = FIBER_CASE if label == 1 else FIBER_CONTROL
    else:
        fiber = FIBER_NULL
    img, g = sample_synthetic_eye(rng, rho, CRYPT_RATE, fiber_contrast=fiber)
    t0 = time.time()
    det = detect_eye_circles(img)
    t_det = time.time() - t0
    old = demo.detect_pupil_iris(img)
    strip = C.rubber_sheet_circles(img, det, n_r=201, n_theta=720)
    gray = pre.preprocess(strip, ColorChannel.GRAY, PT.ORIGINAL)
    x_strip = fe.extract_features(gray, feature_types=[FT.PIXEL, FT.LBP, FT.GLCM, FT.INTENSITY_STATS])
    vr = val.validate(det["pupil_radius"], det["iris_radius"], det["pupil_center"], det["iris_center"])
    vf = val.extract_features(vr)
    x_geom = np.array([vf[k] for k in GEOM_NAMES], dtype=np.float32)
    x_true = np.array([g.rho, g.tau, g.eps], dtype=np.float32)
    return {"x_strip": x_strip.astype(np.float32), "x_geom": x_geom, "x_true": x_true, "label": label,
            "rho": g.rho, "rho_hat": det["pupil_radius"] / det["iris_radius"],
            "rho_hat_demo": old["pupil_radius"] / old["iris_radius"], "iris_r": g.iris_radius,
            "iris_r_hat": det["iris_radius"], "t_det": t_det, "rotulo_validador": vr.quality_label,
            "demo_iris_estimada": bool(old["iris_estimated"])}


def make_models(n_strip):
    n_geom = len(GEOM_NAMES)
    return {
        "faixa": (block_classifier(range(n_strip)), "strip"),
        "geometria": (block_classifier(range(n_strip, n_strip + n_geom)), "strip+geom"),
        "fusao": (build_fusion_classifier(n_strip, n_geom), "strip+geom"),
        "oraculo": (Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=2000))]),
                    "true"),
    }


def _eval_fold(X_sg, X_true, y, tr, te, name):
    models = make_models(X_sg.shape[1] - len(GEOM_NAMES))
    model, kind = models[name]
    X = X_true if kind == "true" else X_sg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
    return {"auc": roc_auc_score(y[te], p), "bal_acc": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))}


def b2_factorial(n_per_class, jobs, repeats):
    print(f"\n[B2] Coorte sintética 2 x 2 ({n_per_class} olhos por classe e cenário)")
    rows, subjects = [], {}
    t0 = time.time()
    for si, (key, label, gsig, tsig) in enumerate(SCENARIOS):
        seeds = [(100_000 * (si + 1) + i, i % 2) for i in range(2 * n_per_class)]
        subs = Parallel(n_jobs=jobs, backend="loky")(
            delayed(synth_subject)(s, lab, gsig, tsig) for s, lab in seeds)
        subjects[key] = subs
        print(f"  cenário {key}: {len(subs)} olhos gerados e medidos ({time.time() - t0:.0f} s)")
    for key, label, gsig, tsig in SCENARIOS:
        subs = subjects[key]
        y = np.array([s["label"] for s in subs])
        X_sg = np.hstack([np.stack([s["x_strip"] for s in subs]), np.stack([s["x_geom"] for s in subs])])
        X_true = np.stack([s["x_true"] for s in subs])
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=7)
        splits = list(cv.split(X_sg, y))
        tasks = [(m, f, tr, te) for m, _l, _c in MODELS for f, (tr, te) in enumerate(splits)]
        out = Parallel(n_jobs=jobs, backend="loky")(
            delayed(_eval_fold)(X_sg, X_true, y, tr, te, m) for m, f, tr, te in tasks)
        for (m, f, _tr, _te), res in zip(tasks, out):
            rows.append({"cenario": key, "modelo": m, "fold": f, **res})
    C.write_csv("b2_fatorial_por_fold.csv", rows)
    n = 2 * n_per_class
    n_test = n / 5.0
    agg = {}
    for key, *_ in SCENARIOS:
        for m, *_ in MODELS:
            for metric in ("auc", "bal_acc"):
                v = [r[metric] for r in rows if r["cenario"] == key and r["modelo"] == m]
                agg[(key, m, metric)] = mc.corrected_mean_ci(v, n - n_test, n_test)
    tests = {}
    for key, *_ in SCENARIOS:
        a = [r["auc"] for r in rows if r["cenario"] == key and r["modelo"] == "fusao"]
        for ref in ("faixa", "geometria"):
            b = [r["auc"] for r in rows if r["cenario"] == key and r["modelo"] == ref]
            tests[(key, ref)] = mc.corrected_paired_ttest(a, b, n - n_test, n_test)
    theo = float(stats.norm.cdf((RHO_CONTROL - RHO_CASE) / (RHO_SD * np.sqrt(2))))
    SUMMARY["b2"] = {"parametros": {"rho_controle": RHO_CONTROL, "rho_caso": RHO_CASE, "rho_dp": RHO_SD,
                                    "fibras_controle": FIBER_CONTROL, "fibras_caso": FIBER_CASE,
                                    "criptas": CRYPT_RATE,
                                    "olhos_por_classe": n_per_class, "repeticoes": repeats,
                                    "auc_teorica_geometria": theo},
                     "resultados": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in agg.items()},
                     "testes_fusao": {f"{k[0]}|{k[1]}": v for k, v in tests.items()}}

    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Cenário & " + " & ".join({"faixa": "Faixa (IC)", "geometria": "Geometria medida",
                                        "fusao": "Fusão", "oraculo": "Oráculo"}[m] for m, *_ in MODELS)
             + " \\\\", "\\midrule"]
    for key, label, *_ in SCENARIOS:
        cells = []
        for m, *_ in MODELS:
            mean, lo, hi = agg[(key, m, "auc")]
            cells.append(f"{br(mean, 3)} [{br(lo, 2)}; {br(hi, 2)}]")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    C.write_tex("tab_fatorial_pupila.tex", "\n".join(lines) + "\n", SCRIPT)

    fig, ax = plt.subplots(figsize=(10.2, 3.9))
    x = np.arange(len(SCENARIOS))
    w = 0.2
    for i, (m, lab, col) in enumerate(MODELS):
        mean = np.array([agg[(k, m, "auc")][0] for k, *_ in SCENARIOS])
        lo = np.array([agg[(k, m, "auc")][1] for k, *_ in SCENARIOS])
        hi = np.array([agg[(k, m, "auc")][2] for k, *_ in SCENARIOS])
        ax.bar(x - 0.3 + i * w, mean, w, color=col, label=lab, yerr=[mean - lo, hi - mean],
               error_kw={"lw": 0.8, "capsize": 2})
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.axhline(theo, color="#1565c0", lw=0.8, ls="--")
    ax.text(len(SCENARIOS) - 0.45, theo + 0.012, f"AUC teórica da geometria = {theo:.3f}".replace(".", ","),
            ha="right", fontsize=7.5, color="#1565c0")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _k, lab, *_ in SCENARIOS])
    ax.set_ylabel("AUC (média e IC 95%)")
    ax.set_ylim(0.3, 1.02)
    ax.legend(ncol=2, frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    C.save_fig(fig, "fatorial_pupila.png")
    for key, *_ in SCENARIOS:
        print("  " + key + ": " + ", ".join(f"{m} {agg[(key, m, 'auc')][0]:.3f}" for m, *_ in MODELS))
    return subjects


# ---------------------------------------------------------------------------
# B3 — Erro de medição
# ---------------------------------------------------------------------------
def _ic_segmenter_rho(seed, label):
    """Mede rho com o segmentador por contornos ativos do pipeline da IC (cenário nulo)."""
    from src.segmentation import IrisSegmenter
    rng = np.random.default_rng(seed)
    rho = float(np.clip(rng.normal(RHO_CONTROL, RHO_SD), 0.15, 0.75))
    img, g = sample_synthetic_eye(rng, rho, CRYPT_RATE, fiber_contrast=FIBER_NULL)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = IrisSegmenter().segment(img)
    ok = bool(res.success and res.iris_radius > 0)
    return {"rho": g.rho, "rho_hat": (res.pupil_radius / res.iris_radius) if ok else float("nan"),
            "sucesso": ok, "mensagem": res.message, "tempo_s": time.time() - t0}


def b3_measurement(subjects, jobs, n_ic=12):
    print("\n[B3] Erro de medição de rho na coorte sintética")
    allsub = [s for subs in subjects.values() for s in subs]
    rho = np.array([s["rho"] for s in allsub])
    new = np.array([s["rho_hat"] for s in allsub])
    old = np.array([s["rho_hat_demo"] for s in allsub])
    res = {}
    for name, est in (("starburst", new), ("demo", old)):
        e = est - rho
        res[name] = {"mae": float(np.mean(np.abs(e))), "mediana_abs": float(np.median(np.abs(e))),
                     "p95_abs": float(np.percentile(np.abs(e), 95)), "max_abs": float(np.max(np.abs(e))),
                     "vies": float(np.mean(e)), "falhas_003": int(np.sum(np.abs(e) > 0.03)),
                     "n": int(len(e)), "r": float(np.corrcoef(rho, est)[0, 1])}
    res["tempo_medio_s"] = float(np.mean([s["t_det"] for s in allsub]))
    res["demo_iris_estimada"] = int(sum(s["demo_iris_estimada"] for s in allsub))
    res["rotulos_validador"] = {k: int(sum(s["rotulo_validador"] == k for s in allsub))
                                for k in ("valid", "borderline", "invalid")}

    seeds = [(100_000 + i, i % 2) for i in range(n_ic)]
    ic = Parallel(n_jobs=min(jobs, n_ic), backend="loky")(delayed(_ic_segmenter_rho)(s, lab) for s, lab in seeds)
    e_ic = np.array([r["rho_hat"] - r["rho"] for r in ic])
    ok = np.isfinite(e_ic)
    res["contornos_ic"] = {"n": n_ic, "sucesso_declarado": int(sum(r["sucesso"] for r in ic)),
                           "mae": float(np.mean(np.abs(e_ic[ok]))) if ok.any() else float("nan"),
                           "falhas_003": int(np.sum(~ok | (np.abs(np.nan_to_num(e_ic, nan=1.0)) > 0.03))),
                           "tempo_medio_s": float(np.mean([r["tempo_s"] for r in ic])),
                           "tempo_max_s": float(np.max([r["tempo_s"] for r in ic])),
                           "linhas": ic}
    C.write_csv("b3_contornos_ic.csv", ic)
    SUMMARY["b3"] = res
    cic = res["contornos_ic"]
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Medida & Contornos ativos (IC)$^{*}$ & Limiar + Hough & \\textit{Starburst} + refinamento \\\\",
             "\\midrule",
             f"Erro absoluto médio de $\\rho$ & {br(cic['mae'], 3)} & {br(res['demo']['mae'], 4)} & "
             f"{br(res['starburst']['mae'], 4)} \\\\",
             f"Erro absoluto mediano & -- & {br(res['demo']['mediana_abs'], 4)} & "
             f"{br(res['starburst']['mediana_abs'], 4)} \\\\",
             f"Percentil 95 do erro absoluto & -- & {br(res['demo']['p95_abs'], 4)} & "
             f"{br(res['starburst']['p95_abs'], 4)} \\\\",
             f"Maior erro absoluto & -- & {br(res['demo']['max_abs'], 3)} & {br(res['starburst']['max_abs'], 4)} \\\\",
             f"Viés (medido $-$ verdadeiro) & -- & {br(res['demo']['vies'], 4)} & {br(res['starburst']['vies'], 4)} \\\\",
             f"Olhos com erro $>0{{,}}03$ & {cic['falhas_003']} de {cic['n']} & {res['demo']['falhas_003']} de "
             f"{res['demo']['n']} & {res['starburst']['falhas_003']} de {res['starburst']['n']} \\\\",
             f"Correlação com $\\rho$ verdadeiro & -- & {br(res['demo']['r'], 3)} & {br(res['starburst']['r'], 4)} \\\\",
             f"Tempo médio por imagem (s) & {br(cic['tempo_medio_s'], 1)} & -- & {br(res['tempo_medio_s'], 1)} \\\\",
             "\\bottomrule", "\\end{tabular}"]
    C.write_tex("tab_medicao_rho.tex", "\n".join(lines) + "\n", SCRIPT)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharex=True, sharey=True)
    ic_rho = np.array([r["rho"] for r in ic])
    ic_hat = np.array([r["rho_hat"] for r in ic])
    panels = ((axes[0], ic_rho, ic_hat, f"(a) contornos ativos da IC ({n_ic} olhos)", "#6a1b9a"),
              (axes[1], rho, old, "(b) limiar + Hough", "#ef6c00"),
              (axes[2], rho, new, "(c) starburst + refinamento", "#1565c0"))
    for ax, xs, est, title, col in panels:
        ax.plot(xs, np.clip(est, 0.1, 1.05), "o", ms=2.8 if len(xs) > 50 else 4.5, alpha=0.6, color=col)
        ax.plot([0.15, 0.75], [0.15, 0.75], color="k", lw=0.7, ls=":")
        ax.set_title(title, loc="left", fontsize=9.2)
        ax.set_xlabel("ρ verdadeiro")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ρ medido")
    axes[0].set_ylim(0.1, 1.06)
    fig.tight_layout()
    C.save_fig(fig, "medicao_rho.png")
    print(f"  MAE starburst {res['starburst']['mae']:.4f}; demo {res['demo']['mae']:.4f}; "
          f"contornos da IC {cic['mae']:.3f} ({cic['falhas_003']} de {n_ic} com erro > 0,03)")


def b_examples_figure():
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.4))
    for j, (rho, fiber) in enumerate(((0.30, FIBER_CONTROL), (0.42, FIBER_CONTROL), (0.34, FIBER_CASE),
                                      (0.55, FIBER_CASE))):
        img, g = sample_synthetic_eye(rng, rho, CRYPT_RATE, fiber_contrast=fiber)
        det = detect_eye_circles(img)
        vis = img.copy()
        cv2.circle(vis, tuple(int(round(v)) for v in det["pupil_center"]), int(round(det["pupil_radius"])),
                   (0, 220, 0), 2)
        cv2.circle(vis, tuple(int(round(v)) for v in det["iris_center"]), int(round(det["iris_radius"])),
                   (0, 0, 230), 2)
        axes[0, j].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[0, j].set_title(f"ρ = {g.rho:.2f}; ρ medido = {det['pupil_radius'] / det['iris_radius']:.3f}"
                             .replace(".", ","), fontsize=8.5)
        axes[0, j].axis("off")
        strip = cv2.cvtColor(C.rubber_sheet_circles(img, det, 201, 720), cv2.COLOR_BGR2GRAY)
        axes[1, j].imshow(strip, cmap="gray", aspect="auto")
        axes[1, j].set_title(f"faixa normalizada (contraste das fibras ×{fiber:.1f})".replace(".", ","),
                             fontsize=8.5)
        axes[1, j].axis("off")
    fig.tight_layout()
    C.save_fig(fig, "olhos_sinteticos.png", comma=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.per_class, args.repeats = 20, 1
    C.ensure_dirs()
    t0 = time.time()
    SUMMARY["ambiente"] = C.environment()
    b1_invariance()
    b_examples_figure()
    subjects = b2_factorial(args.per_class, args.jobs, args.repeats)
    b3_measurement(subjects, args.jobs, n_ic=4 if args.quick else 12)
    SUMMARY["tempo_total_s"] = time.time() - t0
    C.write_json("resumo_pupila.json", SUMMARY)
    print(f"\nConcluído em {time.time() - t0:.0f} s. Resumo em results/pgc3/resumo_pupila.json")


if __name__ == "__main__":
    main()

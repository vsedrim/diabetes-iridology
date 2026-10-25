"""
Números citados no relatório do PGC III
=======================================
Lê as saídas dos experimentos (results/pgc3/*.json e *.csv) e grava
docs/PGC_3/tabelas/numeros.tex com uma definição por valor, usada no texto
pela macro \\val{chave}. Assim nenhum número do relatório é digitado à mão.

Uso:
    python experiments/pgc3_numbers.py
"""

import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pgc3_common as C  # noqa: E402

REPS = ["original", "histograma", "clahe", "verde_clahe", "frangi", "gabor", "blackhat", "vessel"]
FILTERS = ["frangi", "gabor", "blackhat", "vessel"]
FEATURE_SETS = ["pixels", "textura", "ic"]
CLASSIFIERS = ["RL", "SVM", "RF", "MLP", "AdaBoost"]
SCENARIOS = ["nulo", "geometria", "textura", "ambos"]
MODELS = ["faixa", "geometria", "fusao", "oraculo"]


def num(x, nd=3, sign=False):
    """Número em LaTeX com vírgula decimal (e sinal explícito, se pedido)."""
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{abs(x):.{nd}f}".replace(".", "{,}")
    if x < 0 and float(f"{abs(x):.{nd}f}") != 0:
        return f"$-${s}"
    if sign and x > 0:
        return f"$+${s}"
    return s


def pval(p):
    if p is None or not np.isfinite(p):
        return "--"
    if p < 0.001:
        return "$<$0{,}001"
    return num(p, 3)


def shared_unique_images():
    """Imagens únicas comuns aos dois leiautes por pessoa (requer a base legada)."""
    try:
        from src.legacy_data import load_legacy_layout, resolve_legacy_source
        src = resolve_legacy_source(None)
        a = load_legacy_layout(src, "personBase_invert", near_duplicate_threshold=None)
        b = load_legacy_layout(src, "personBase", near_duplicate_threshold=None)
        return len(set(a.hashes.tolist()) & set(b.hashes.tolist()))
    except (FileNotFoundError, OSError):
        return None


def main():
    vals = {}
    with open(os.path.join(C.OUT_DIR, "resumo_vascular.json"), encoding="utf-8") as fh:
        A = json.load(fh)
    with open(os.path.join(C.OUT_DIR, "resumo_pupila.json"), encoding="utf-8") as fh:
        B = json.load(fh)
    with open(os.path.join(C.OUT_DIR, "a1_grade_resumo.csv"), encoding="utf-8") as fh:
        grid = list(csv.DictReader(fh))

    prot = A["protocolo"]
    vals["prot-rep"] = str(prot["repeticoes"])
    vals["prot-folds"] = str(prot["folds"])
    vals["prot-perm"] = str(prot["permutacoes"])
    vals["prot-ntreino"] = str(int(round(prot["n_treino_medio"])))
    vals["prot-nteste"] = str(int(round(prot["n_teste_medio"])))

    aud = A["auditoria"]["personBase_invert"]
    audb = A["auditoria"]["personBase"]
    vals["aud-linhas"] = str(aud["rows_control"] + aud["rows_diabetic"])
    vals["aud-copias"] = str(aud["exact_duplicates_removed"])
    vals["aud-unicas"] = str(aud["unique_control"] + aud["unique_diabetic"])
    vals["aud-controle"] = str(aud["unique_control"])
    vals["aud-diabetes"] = str(aud["unique_diabetic"])
    vals["aud-pb-controle"] = str(audb["unique_control"])
    vals["aud-pb-diabetes"] = str(audb["unique_diabetic"])
    vals["aud-multmax"] = str(max(int(k) for k in aud["duplicate_multiplicity"]))
    vals["aud-esp-controle"] = str(aud["mirrored_control"])
    vals["aud-esp-diabetes"] = str(aud["mirrored_diabetic"])
    vals["aud-esp-total"] = str(aud["mirrored_control"] + aud["mirrored_diabetic"])
    vals["aud-naoesp-total"] = str(aud["unique_control"] + aud["unique_diabetic"]
                                   - aud["mirrored_control"] - aud["mirrored_diabetic"])
    lado = A["auditoria"].get("lado") or {}
    for key, value in lado.items():
        vals[f"aud-lado-{key.replace('_', '-')}"] = str(value)
    corr = A["auditoria"]["correlacao_pares"]
    vals["aud-corr-mediana"] = num(corr["mediana"], 2)
    vals["aud-corr-max"] = num(corr["max"], 3)
    for lab, key in (("controle", "controle"), ("diabetes", "diabetes")):
        r, g, b = A["auditoria"]["cor_media_rgb"][key]
        vals[f"aud-r-{lab}"] = num(r, 1)
        vals[f"aud-g-{lab}"] = num(g, 1)
        vals[f"aud-b-{lab}"] = num(b, 1)
    shared = A["auditoria"].get("compartilhadas", shared_unique_images())
    vals["aud-compartilhadas"] = str(shared) if shared is not None else "--"

    # Grade principal
    cfg = {(r["rep"], r["fs"], r["clf"]): r for r in grid}
    for (rep, fs, clf), r in cfg.items():
        vals[f"ab-{fs}-{rep}-{clf}"] = num(float(r["bal_acc_media"]))
        vals[f"auc-{fs}-{rep}-{clf}"] = num(float(r["auc_media"]))
    for key, v in A["media_por_rep_fs"].items():
        rep, fs = key.split("|")
        vals[f"media-{fs}-{rep}"] = num(v)
        aucs = [float(cfg[(rep, fs, c)]["auc_media"]) for c in CLASSIFIERS]
        vals[f"mediaauc-{fs}-{rep}"] = num(float(np.mean(aucs)))
    for rep, v in A["media_por_rep"].items():
        vals[f"media-{rep}"] = num(v["bal_acc"])
        vals[f"mediaauc-{rep}"] = num(v["auc"])
    wins = sum(float(cfg[(rep, "pixels", c)]["bal_acc_media"]) > float(cfg[("original", "pixels", c)]["bal_acc_media"])
               for rep in FILTERS for c in CLASSIFIERS)
    vals["pix-vitorias"] = str(wins)
    for fs in FEATURE_SETS:
        fm = {rep: A["media_por_rep_fs"][f"{rep}|{fs}"] for rep in FILTERS}
        vals[f"{fs}-filtros-min"] = num(min(fm.values()))
        vals[f"{fs}-filtros-max"] = num(max(fm.values()))

    rec = A.get("recorte")
    if rec:
        for region in ("faixa", "estroma", "bordas"):
            for rep, v in rec[region]["media"].items():
                vals[f"rec-{region}-{rep}"] = num(v)
        for region in ("estroma", "bordas"):
            vals[f"rec-vit-{region}"] = str(rec[region]["vitorias"])
            for rep, t in rec[region]["efeito"].items():
                vals[f"rec-ef-{region}-{rep}"] = num(t["diff"], 3, sign=True)
                vals[f"rec-efp-{region}-{rep}"] = pval(t["p"])
                vals[f"rec-efh-{region}-{rep}"] = pval(t["p_holm"])
                vals[f"rec-eflo-{region}-{rep}"] = num(t["ci_low"], 3, sign=True)
                vals[f"rec-efhi-{region}-{rep}"] = num(t["ci_high"], 3, sign=True)
            fm = [rec[region]["media"][r] for r in FILTERS]
            vals[f"rec-{region}-filtros-min"] = num(min(fm))
            vals[f"rec-{region}-filtros-max"] = num(max(fm))

    des = A.get("desespelhado")
    if des:
        for rep, v in des["media"].items():
            vals[f"des-{rep}"] = num(v)
        vals["des-vit"] = str(des["vitorias"])
        for rep, t in des["efeito"].items():
            vals[f"des-ef-{rep}"] = num(t["diff"], 3, sign=True)
            vals[f"des-efh-{rep}"] = pval(t["p_holm"])
        for rep, t in des.get("mudanca", {}).items():
            vals[f"des-mud-{rep}"] = num(t["diff"], 3, sign=True)
            vals[f"des-mudp-{rep}"] = pval(t["p"])
            vals[f"des-mudlo-{rep}"] = num(t["ci_low"], 3, sign=True)
            vals[f"des-mudhi-{rep}"] = num(t["ci_high"], 3, sign=True)
        fm = [des["media"][r] for r in FILTERS]
        vals["des-filtros-min"] = num(min(fm))
        vals["des-filtros-max"] = num(max(fm))

    for fs, d in A["efeito_por_conjunto"].items():
        for rep, t in d.items():
            vals[f"ef-{fs}-{rep}"] = num(t["diff"], 3, sign=True)
            vals[f"eflo-{fs}-{rep}"] = num(t["ci_low"], 3, sign=True)
            vals[f"efhi-{fs}-{rep}"] = num(t["ci_high"], 3, sign=True)
            vals[f"efp-{fs}-{rep}"] = pval(t["p"])
            vals[f"efh-{fs}-{rep}"] = pval(t["p_holm"])

    for tag, fr in (("fr", A["friedman_ab"]), ("frauc", A["friedman_auc"])):
        vals[f"{tag}-chi"] = num(fr["chi2"], 1)
        vals[f"{tag}-p"] = pval(fr["p_chi2"])
        vals[f"{tag}-ff"] = num(fr["ff"], 2)
        vals[f"{tag}-cd"] = num(fr["cd"], 2)
        for rep, rk in zip(REPS, fr["ranks"]):
            vals[f"{'rk' if tag == 'fr' else 'rkauc'}-{rep}"] = num(rk, 2)

    perm = A.get("permutacao", {})
    if perm:
        for rep, p in perm.items():
            vals[f"perm-obs-{rep}"] = num(p["observado"])
            vals[f"perm-pv-{rep}"] = pval(p["p"])
        vals["perm-nulo-media"] = num(float(np.mean([p["nulo_media"] for p in perm.values()])))
        vals["perm-nulo-dp"] = num(float(np.mean([p["nulo_dp"] for p in perm.values()])))
        vals["perm-p"] = num(max(p["p"] for p in perm.values()), 3)

    ctl = A.get("controles", {})
    for key in ("cor", "manual", "fusao_frangi", "fusao_gabor"):
        best_clf, best = None, -1.0
        for clf in CLASSIFIERS:
            v = ctl.get(f"{key}|{clf}")
            if v is None:
                continue
            vals[f"ctl-ab-{key}-{clf}"] = num(v["bal_acc"])
            vals[f"ctl-auc-{key}-{clf}"] = num(v["auc"])
            if v["bal_acc"] > best:
                best_clf, best = clf, v["bal_acc"]
        if best_clf:
            vals[f"ctl-abmax-{key}"] = num(best)
            vals[f"ctl-aucmax-{key}"] = num(ctl[f"{key}|{best_clf}"]["auc"])
            vals[f"ctl-clfmax-{key}"] = best_clf
    for t in A.get("controles_testes", []):
        vals[f"ctlt-d-{t['entrada']}-{t['clf']}"] = num(t["diff"], 3, sign=True)
        vals[f"ctlt-p-{t['entrada']}-{t['clf']}"] = pval(t["p"])

    rob = A.get("robustez_resumo")
    if rob:
        for rep in REPS:
            vals[f"rob-mabs-{rep}"] = num(rob["media_abs"][rep])
            vals[f"rob-pior-{rep}"] = num(rob["pior_queda"][rep], 3, sign=True)

    rep_ = A.get("replicacao")
    pb_path = os.path.join(C.OUT_DIR, "a7_personbase_resumo.csv")
    if os.path.exists(pb_path):
        with open(pb_path, encoding="utf-8") as fh:
            pbc = {(r["rep"], r["fs"], r["clf"]): r for r in csv.DictReader(fh)}
        vals["pb-pix-vitorias"] = str(sum(
            float(pbc[(rep, "pixels", c)]["bal_acc_media"]) > float(pbc[("original", "pixels", c)]["bal_acc_media"])
            for rep in FILTERS for c in CLASSIFIERS))
        for (rep, fs, clf), r in pbc.items():
            vals[f"pb-ab-{fs}-{rep}-{clf}"] = num(float(r["bal_acc_media"]))
    if rep_:
        vals["rep-spearman"] = num(rep_["spearman_configs"], 2)
        vals["rep-spearman-p"] = pval(rep_["p"])
        vals["rep-spearman-rank"] = num(rep_["spearman_postos"], 2)
        vals["rep-fr-chi"] = num(rep_["friedman"]["chi2"], 1)
        vals["rep-fr-p"] = pval(rep_["friedman"]["p_chi2"])
        for rep, rk in rep_["postos"].items():
            vals[f"rep-rk-{rep}"] = num(rk, 2)
        for key, v in rep_["media_rep_fs"].items():
            r, fs = key.split("|")
            vals[f"rep-media-{fs}-{r}"] = num(v)
        for fs, d in rep_["efeito_por_conjunto"].items():
            for r, t in d.items():
                vals[f"rep-ef-{fs}-{r}"] = num(t["diff"], 3, sign=True)
                vals[f"rep-efp-{fs}-{r}"] = pval(t["p"])
                vals[f"rep-efh-{fs}-{r}"] = pval(t["p_holm"])
        pix = [rep_["media_rep_fs"][f"{r}|pixels"] for r in FILTERS]
        vals["rep-pix-filtros-min"] = num(min(pix))
        vals["rep-pix-filtros-max"] = num(max(pix))

    for rep, m in A.get("mapas_d", {}).items():
        vals[f"map-frac-{rep}"] = num(100 * m["frac_fdr05"], 1)
        vals[f"map-dmax-{rep}"] = num(m["d_abs_max"], 2)

    vals["tempo-a"] = str(int(round((A["tempo_total_s"] + sum(A.get("tempo_partes_s", {}).values())) / 60)))
    env = prot["ambiente"]
    vals["env-python"] = env["python"]
    vals["env-numpy"] = env["numpy"]
    vals["env-scipy"] = env["scipy"]
    vals["env-sklearn"] = env["scikit-learn"]
    vals["env-skimage"] = env["scikit-image"]
    vals["env-opencv"] = env["opencv"]
    vals["env-cpus"] = str(env.get("cpus", os.cpu_count()))

    # Parte B
    b1 = B["b1"]
    vals["b1-rho-min"] = num(b1["faixa_rho"][0], 2)
    vals["b1-rho-max"] = num(b1["faixa_rho"][1], 2)
    vals["b1-foto-rho"] = num(b1["foto_rho_referencia"], 2)
    vals["b1-foto-ssim-min"] = num(b1["foto_ssim_min"], 2)
    vals["b1-foto-mae"] = num(b1["foto_rho_mae"], 4)
    vals["b1-sint-lin-ssim"] = num(b1["sint_linear_ssim_min"], 3)
    vals["b1-sint-nl-ssim"] = num(b1["sint_nao_linear_ssim_min"], 2)
    vals["b1-sint-mae"] = num(b1["sint_rho_mae"], 4)
    vals["b1-outras-ssim"] = num(b1["outras_iris_ssim"], 2)

    b2 = B["b2"]
    par = b2["parametros"]
    vals["b2-teo"] = num(par["auc_teorica_geometria"], 3)
    vals["b2-rho-c"] = num(par["rho_controle"], 2)
    vals["b2-rho-d"] = num(par["rho_caso"], 2)
    vals["b2-rho-sd"] = num(par["rho_dp"], 2)
    vals["b2-fibra-c"] = num(par["fibras_controle"], 1)
    vals["b2-fibra-d"] = num(par["fibras_caso"], 1)
    vals["b2-criptas"] = str(int(par["criptas"]))
    vals["b2-n"] = str(par["olhos_por_classe"])
    for s in SCENARIOS:
        for m in MODELS:
            mean, lo, hi = b2["resultados"][f"{s}|{m}|auc"]
            vals[f"b2-{s}-{m}"] = num(mean)
            vals[f"b2lo-{s}-{m}"] = num(lo)
            vals[f"b2hi-{s}-{m}"] = num(hi)
            vals[f"b2ab-{s}-{m}"] = num(b2["resultados"][f"{s}|{m}|bal_acc"][0])
    for key, t in b2["testes_fusao"].items():
        s, ref = key.split("|")
        vals[f"b2t-{s}-{ref}-d"] = num(t["diff"], 3, sign=True)
        vals[f"b2t-{s}-{ref}-p"] = pval(t["p"])

    b3 = B["b3"]
    for det in ("starburst", "demo"):
        d = b3[det]
        vals[f"b3-{det}-mae"] = num(d["mae"], 4)
        vals[f"b3-{det}-mediana"] = num(d["mediana_abs"], 4)
        vals[f"b3-{det}-p95"] = num(d["p95_abs"], 4)
        vals[f"b3-{det}-max"] = num(d["max_abs"], 3)
        vals[f"b3-{det}-vies"] = num(d["vies"], 4, sign=True)
        vals[f"b3-{det}-falhas"] = str(d["falhas_003"])
        vals[f"b3-{det}-n"] = str(d["n"])
        vals[f"b3-{det}-r"] = num(d["r"], 4)
    ic = b3["contornos_ic"]
    vals["b3-ic-n"] = str(ic["n"])
    vals["b3-ic-falhas"] = str(ic["falhas_003"])
    vals["b3-ic-mae"] = num(ic["mae"], 3)
    vals["b3-ic-tempo"] = num(ic["tempo_medio_s"], 1)
    vals["b3-tempo"] = num(b3["tempo_medio_s"], 1)
    vals["tempo-b"] = str(int(round(B["tempo_total_s"] / 60)))

    path = os.path.join(C.TAB_DIR, "numeros.tex")
    # Valores p prontos para o texto: "$p=0{,}034$" ou "$p<0{,}001$" (chave com sufixo -txt).
    ptoken = re.compile(r"(^|-)(p|pv|efp|efh|mudp)(-|$)")
    for k, v in list(vals.items()):
        if ptoken.search(k) and v != "--":
            vals[f"{k}-txt"] = f"$p<{v[3:]}$" if v.startswith("$<$") else f"$p={v}$"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("% Gerado por experiments/pgc3_numbers.py (não editar à mão).\n"
                 "% pgcval: vírgula decimal (texto em português); pgcvalen: ponto decimal (abstract).\n"
                 "\\makeatletter\n")
        for k in sorted(vals):
            fh.write(f"\\@namedef{{pgcval@{k}}}{{{vals[k]}}}\n")
            fh.write(f"\\@namedef{{pgcvalen@{k}}}{{{vals[k].replace('{,}', '.')}}}\n")
        fh.write("\\makeatother\n")
    print(f"{len(vals)} valores gravados em docs/PGC_3/tabelas/numeros.tex")


if __name__ == "__main__":
    main()

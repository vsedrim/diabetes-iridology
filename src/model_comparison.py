"""
Comparação estatística de modelos
=================================
Ferramentas para comparar configurações avaliadas por validação cruzada
repetida, seguindo as recomendações usuais da literatura de aprendizado de
máquina:

- intervalo de confiança e teste t pareado com a correção de variância de
  Nadeau e Bengio (2003) para validação cruzada repetida (os folds partilham
  dados de treino, então o teste t comum subestima a variância);
- ajuste de Holm (1979) para comparações múltiplas;
- teste de Friedman, estatística de Iman e Davenport e diferença crítica de
  Nemenyi para comparar várias configurações sobre vários blocos
  (Demšar, 2006).
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats

# Valores críticos q_alpha (alpha = 0,05) do teste de Nemenyi (Demšar, 2006, Tabela 5a),
# indexados pelo número de configurações comparadas.
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
               8: 3.031, 9: 3.102, 10: 3.164}


def corrected_variance_factor(n_train: int, n_test: int, n_folds_total: int) -> float:
    """Fator (1/k + n_teste/n_treino) que multiplica a variância amostral."""
    return 1.0 / n_folds_total + n_test / float(n_train)


def corrected_mean_ci(values: Sequence[float], n_train: int, n_test: int,
                      alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Média e intervalo de confiança de uma métrica obtida em k folds de
    validação cruzada repetida, com a correção de Nadeau e Bengio (2003).
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    k = x.size
    if k < 2:
        m = float(x.mean()) if k else float("nan")
        return m, float("nan"), float("nan")
    mean = float(x.mean())
    se = np.sqrt(corrected_variance_factor(n_train, n_test, k) * x.var(ddof=1))
    half = float(stats.t.ppf(1 - alpha / 2, df=k - 1) * se)
    return mean, mean - half, mean + half


def corrected_paired_ttest(a: Sequence[float], b: Sequence[float], n_train: int,
                           n_test: int) -> Dict[str, float]:
    """
    Teste t pareado corrigido (Nadeau; Bengio, 2003) para as diferenças a - b
    medidas nos mesmos folds. Devolve diferença média, estatística t, valor p
    bilateral e o intervalo de confiança de 95% da diferença.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.isfinite(d)]
    k = d.size
    mean = float(d.mean()) if k else float("nan")
    if k < 2:
        return {"diff": mean, "t": float("nan"), "p": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "k": k}
    var = d.var(ddof=1)
    if var <= 0:
        t = 0.0 if mean == 0 else np.sign(mean) * np.inf
        p = 1.0 if mean == 0 else 0.0
        return {"diff": mean, "t": float(t), "p": float(p),
                "ci_low": mean, "ci_high": mean, "k": k}
    se = np.sqrt(corrected_variance_factor(n_train, n_test, k) * var)
    t = mean / se
    p = 2.0 * stats.t.sf(abs(t), df=k - 1)
    half = stats.t.ppf(0.975, df=k - 1) * se
    return {"diff": mean, "t": float(t), "p": float(p),
            "ci_low": float(mean - half), "ci_high": float(mean + half), "k": k}


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Valores p ajustados pelo procedimento de Holm (1979), na ordem original."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def average_ranks(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """
    Postos médios das colunas (configurações) sobre as linhas (blocos), com
    empates recebendo o posto médio. O posto 1 é o melhor.
    """
    s = np.asarray(scores, dtype=float)
    ranks = np.vstack([stats.rankdata(-row if higher_is_better else row) for row in s])
    return ranks.mean(axis=0)


def friedman_nemenyi(scores: np.ndarray, alpha: float = 0.05,
                     higher_is_better: bool = True) -> Dict[str, object]:
    """
    Teste de Friedman sobre uma matriz (blocos x configurações), estatística
    F de Iman e Davenport e diferença crítica (CD) de Nemenyi.
    """
    s = np.asarray(scores, dtype=float)
    n_blocks, k = s.shape
    ranks = average_ranks(s, higher_is_better)
    chi2 = 12.0 * n_blocks / (k * (k + 1)) * (np.sum(ranks ** 2) - k * (k + 1) ** 2 / 4.0)
    p_chi2 = float(stats.chi2.sf(chi2, df=k - 1))
    denom = n_blocks * (k - 1) - chi2
    if denom > 0:
        ff = (n_blocks - 1) * chi2 / denom
        p_ff = float(stats.f.sf(ff, k - 1, (k - 1) * (n_blocks - 1)))
    else:
        ff, p_ff = float("inf"), 0.0
    if alpha != 0.05 or k not in NEMENYI_Q05:
        # Aproximação pela distribuição da amplitude estudentizada com graus de liberdade infinitos.
        q = float(stats.studentized_range.ppf(1 - alpha, k, 1e6) / np.sqrt(2.0))
    else:
        q = NEMENYI_Q05[k]
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    return {"ranks": ranks, "chi2": float(chi2), "p_chi2": p_chi2, "ff": float(ff),
            "p_ff": p_ff, "cd": float(cd), "q_alpha": float(q), "n_blocks": n_blocks, "k": k}


def permutation_pvalue(observed: float, null_scores: Sequence[float]) -> float:
    """Valor p de permutação (unilateral, com correção +1; Phipson; Smyth, 2010)."""
    null = np.asarray(null_scores, dtype=float)
    return float((1 + np.sum(null >= observed)) / (1 + null.size))

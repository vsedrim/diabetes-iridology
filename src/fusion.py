"""
Fusão da textura da faixa normalizada com a geometria da pupila
===============================================================
A normalização rubber sheet de Daugman leva o anel entre a pupila e o limbo a
uma faixa de tamanho fixo e, por construção, descarta o tamanho da pupila. Por
isso a geometria pupila-íris (razão entre os raios, espessura relativa do
anel e excentricidade) precisa entrar no classificador como um bloco
explícito de features, medido na própria fotografia pela etapa de
segmentação.

O empilhamento (Wolpert, 1992) treina um modelo por bloco e combina as
probabilidades fora do fold com uma regressão logística. Assim o bloco
geométrico, com três valores, não é diluído pelos milhares de valores da
faixa, como aconteceria numa simples concatenação regularizada.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Features invariantes à escala da imagem (não dependem da distância da câmera).
GEOMETRY_FEATURES = ("pupil_iris_ratio", "thickness_ratio", "concentricity_offset")


def geometry_vector(features: Dict[str, float],
                    names: Sequence[str] = GEOMETRY_FEATURES) -> np.ndarray:
    """Vetor geométrico a partir do dicionário de ``PupilValidator.extract_features``."""
    return np.asarray([float(features[n]) for n in names], dtype=np.float32)


def block_classifier(columns: Sequence[int], C: float = 1.0) -> Pipeline:
    """Regressão logística padronizada sobre um subconjunto de colunas."""
    return Pipeline([("selecao", ColumnTransformer([("bloco", "passthrough", list(columns))])),
                     ("padronizacao", StandardScaler()),
                     ("classificador", LogisticRegression(C=C, max_iter=2000))])


def build_fusion_classifier(n_texture: int, n_geometry: int, cv: int = 5) -> StackingClassifier:
    """Empilhamento de um modelo da textura e um modelo da geometria."""
    texture_cols = list(range(n_texture))
    geometry_cols = list(range(n_texture, n_texture + n_geometry))
    return StackingClassifier(
        estimators=[("textura", block_classifier(texture_cols)),
                    ("geometria", block_classifier(geometry_cols))],
        final_estimator=LogisticRegression(max_iter=2000), cv=cv, stack_method="predict_proba")


def evaluate_geometry_fusion(X_texture: np.ndarray, X_geometry: np.ndarray, y: np.ndarray,
                             n_splits: int = 5, n_repeats: int = 5,
                             random_state: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Compara, com a mesma validação cruzada estratificada repetida, a textura
    sozinha, a geometria sozinha e a fusão por empilhamento. Devolve média e
    desvio-padrão da AUC e da acurácia balanceada de cada modelo.
    """
    X = np.hstack([np.asarray(X_texture, dtype=np.float32), np.asarray(X_geometry, dtype=np.float32)])
    y = np.asarray(y)
    n_tex, n_geo = X_texture.shape[1], X_geometry.shape[1]
    models = {
        "textura": lambda: block_classifier(range(n_tex)),
        "geometria": lambda: block_classifier(range(n_tex, n_tex + n_geo)),
        "fusao": lambda: build_fusion_classifier(n_tex, n_geo),
    }
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scores = {name: {"auc": [], "bal_acc": []} for name in models}
    for train, test in cv.split(X, y):
        for name, make in models.items():
            model = make().fit(X[train], y[train])
            proba = model.predict_proba(X[test])[:, 1]
            scores[name]["auc"].append(roc_auc_score(y[test], proba))
            scores[name]["bal_acc"].append(balanced_accuracy_score(y[test], (proba >= 0.5).astype(int)))
    return {name: {f"{m}_media": float(np.mean(v)) for m, v in s.items()}
            | {f"{m}_dp": float(np.std(v, ddof=1)) for m, v in s.items()}
            for name, s in scores.items()}

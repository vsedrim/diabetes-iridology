"""
Leitura segura e auditada da base legada
========================================
A base legada (``Data.zip`` ou pastas ``Data/Data/<layout>``) guarda faixas da
íris já normalizadas pelo modelo *rubber sheet* (201 x 720 x 3, BGR, uint8) em
arquivos *pickle*. Este módulo lê esses arquivos sem executar código arbitrário
e registra o que a auditoria precisa saber antes de qualquer experimento:

- ``pickle`` executa código ao desserializar; o leitor aceita apenas os
  construtores do NumPy necessários para reconstruir um ``ndarray``;
- o leiaute ``personBase`` tem 196 linhas, mas só 123 imagens distintas: as
  cópias exatas são removidas por SHA-256 para não vazar entre treino e teste;
- no leiaute ``personBase_invert`` as faixas do olho direito foram espelhadas
  (inversão angular seguida de deslocamento de 360 colunas); a coluna vazia da
  costura angular denuncia o espelhamento (coluna 360 em vez de 719) e é
  reparada para não virar um marcador artificial do lado do olho;
- a base não traz identificadores de pessoa nem rótulos clínicos verificados,
  e o artigo associado a ela foi retratado pelo IEEE. Os resultados obtidos
  com ela são exploratórios e não sustentam conclusões clínicas.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

LAYOUTS = ("all", "L_split", "R_split", "personBase", "personBase_invert")
CONTROL_FILE = "controlImageArr.p"
DIABETIC_FILE = "diabeteImageArr.p"

ALLOWED_PICKLE_GLOBALS = {
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    # Protocolo 5: o NumPy reconstrói o array a partir do buffer.
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
    # Protocolo 2 gravado pelo Python 3: bytes são serializados como texto + codec.
    ("_codecs", "encode"),
}
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_COMPRESSION_RATIO = 50.0

SOURCE_PUBLICATION_DOI = "10.1109/ICBME.2018.8703564"
RETRACTION_NOTICE_DOI = "10.1109/ICBME45317.2018.10207763"
LIMITATIONS = (
    "A base contém apenas faixas normalizadas, sem as fotografias originais nem os contornos.",
    "O diâmetro da pupila e a espessura do anel não podem ser recuperados das faixas.",
    "Não há identificadores de pessoa; olhos da mesma pessoa podem cair em folds diferentes.",
    "A classe positiva é diabetes informado pela base; o subtipo DM2 não é verificável.",
    "O artigo associado à base foi retratado pelo IEEE.",
)


class LegacyDataError(ValueError):
    """Erro de validação da base legada."""


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Desserializa apenas ``numpy.ndarray``; qualquer outro global é bloqueado."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"global bloqueado: {module}.{name}")
        return super().find_class(module, name)


def restricted_numpy_load(payload: Union[bytes, io.BufferedIOBase]) -> np.ndarray:
    """Carrega um ``ndarray`` de um pickle legado (Python 2, latin-1) com segurança."""
    stream = io.BytesIO(payload) if isinstance(payload, (bytes, bytearray)) else payload
    value = RestrictedNumpyUnpickler(stream, encoding="latin1").load()
    if not isinstance(value, np.ndarray):
        raise pickle.UnpicklingError(f"esperado numpy.ndarray, recebido {type(value).__name__}")
    return value


def sha256_bytes(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _read_zip_member(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.flag_bits & 0x1:
        raise LegacyDataError(f"membro criptografado não suportado: {name}")
    if info.file_size > MAX_MEMBER_BYTES:
        raise LegacyDataError(f"membro acima do limite de tamanho: {name}")
    if info.file_size / max(info.compress_size, 1) > MAX_COMPRESSION_RATIO:
        raise LegacyDataError(f"membro acima do limite de compressão: {name}")
    payload = archive.read(info)
    if len(payload) != info.file_size:
        raise LegacyDataError(f"tamanho inconsistente: {name}")
    return payload


def resolve_legacy_source(explicit: Optional[Union[str, Path]] = None,
                          repo_root: Optional[Union[str, Path]] = None) -> Path:
    """
    Localiza a base legada: argumento explícito, variável ``IRIS_LEGACY_DATA``,
    ``Data/Data`` ou ``Data.zip`` na raiz do repositório, nessa ordem.
    """
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("IRIS_LEGACY_DATA"):
        candidates.append(Path(os.environ["IRIS_LEGACY_DATA"]))
    candidates += [repo_root / "Data" / "Data", repo_root / "Data.zip", repo_root / "Data"]
    for candidate in candidates:
        if candidate.is_file() and zipfile.is_zipfile(candidate):
            return candidate
        if candidate.is_dir() and any((candidate / lay).is_dir() for lay in LAYOUTS):
            return candidate
        if candidate.is_dir() and (candidate / "Data").is_dir():
            return candidate / "Data"
    raise FileNotFoundError(
        "Base legada não encontrada. Informe o caminho do Data.zip ou da pasta Data/Data "
        "(argumento --data ou variável IRIS_LEGACY_DATA)."
    )


def read_layout_member(source: Union[str, Path], layout: str, filename: str) -> np.ndarray:
    """Lê um arquivo de um leiaute (``controlImageArr.p``, ``DR_type.p`` etc.), do zip ou da pasta."""
    if layout not in LAYOUTS:
        raise LegacyDataError(f"leiaute desconhecido: {layout}")
    source = Path(source)
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            return restricted_numpy_load(_read_zip_member(archive, f"Data/{layout}/{filename}"))
    with open(source / layout / filename, "rb") as fh:
        return restricted_numpy_load(fh.read())


def read_layout_arrays(source: Union[str, Path], layout: str) -> Tuple[np.ndarray, np.ndarray]:
    """Lê os tensores de controle e de diabetes de um leiaute (zip ou pasta)."""
    control = read_layout_member(source, layout, CONTROL_FILE)
    diabetic = read_layout_member(source, layout, DIABETIC_FILE)
    if control.ndim != 4 or diabetic.ndim != 4 or control.shape[1:] != diabetic.shape[1:]:
        raise LegacyDataError("tensores de controle e de diabetes incompatíveis")
    if control.dtype != np.uint8 or diabetic.dtype != np.uint8:
        raise LegacyDataError("as faixas legadas devem ser uint8")
    return control, diabetic


def empty_columns(image: np.ndarray) -> np.ndarray:
    """Índices das colunas angulares totalmente nulas (costura da normalização)."""
    flat = image.reshape(image.shape[0], image.shape[1], -1)
    return np.where(np.all(flat == 0, axis=(0, 2)))[0]


def repair_angular_seam(image: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Preenche colunas angulares totalmente nulas com a média das vizinhas
    circulares. A faixa é periódica em theta, então a coluna 719 é vizinha da 0.
    """
    cols = empty_columns(image)
    if cols.size == 0 or cols.size >= image.shape[1] // 2:
        return image, []
    repaired = image.copy()
    width = image.shape[1]
    empty = set(int(c) for c in cols)
    for c in cols:
        left, right = (c - 1) % width, (c + 1) % width
        while left in empty and left != c:
            left = (left - 1) % width
        while right in empty and right != c:
            right = (right + 1) % width
        repaired[:, c] = ((image[:, left].astype(np.uint16) + image[:, right]) // 2).astype(np.uint8)
    return repaired, [int(c) for c in cols]


@dataclass
class LegacyDataset:
    """Imagens únicas de um leiaute, com rótulos, procedência e auditoria."""

    images: np.ndarray            # (N, H, W, 3) uint8, BGR
    labels: np.ndarray            # 0 = controle informado, 1 = diabetes informado
    sample_ids: np.ndarray        # "control-007", "diabetic-031" (linha de origem)
    hashes: np.ndarray            # SHA-256 da faixa original (antes do reparo)
    mirrored: np.ndarray          # True quando a faixa veio espelhada (olho direito)
    groups: np.ndarray            # grupo para a validação cruzada (quase duplicatas juntas)
    layout: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.labels))

    @property
    def n_control(self) -> int:
        return int(np.sum(self.labels == 0))

    @property
    def n_diabetic(self) -> int:
        return int(np.sum(self.labels == 1))


def _deduplicate(images, labels, ids, hashes):
    first: Dict[str, int] = {}
    keep: List[int] = []
    groups: Dict[str, List[str]] = {}
    for idx, digest in enumerate(hashes):
        if digest not in first:
            first[digest] = idx
            keep.append(idx)
            groups[digest] = [str(ids[idx])]
        else:
            if labels[first[digest]] != labels[idx]:
                raise LegacyDataError(
                    f"imagem idêntica com rótulos conflitantes: {ids[first[digest]]}, {ids[idx]}")
            groups[digest].append(str(ids[idx]))
    dup_groups = [{"mantida": v[0], "removidas": v[1:]} for v in groups.values() if len(v) > 1]
    return np.asarray(keep, dtype=np.int64), dup_groups


def near_duplicate_groups(images: np.ndarray, threshold: float = 0.97,
                          step: int = 4) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """
    Agrupa quase duplicatas (correlação de Pearson >= ``threshold`` entre as
    faixas em escala de cinza subamostradas). Devolve o rótulo de grupo de cada
    imagem e os pares encontrados.
    """
    n = len(images)
    gray = images.astype(np.float32).mean(axis=3)[:, ::step, ::step].reshape(n, -1)
    gray -= gray.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(gray, axis=1, keepdims=True)
    gray /= np.maximum(norms, 1e-9)
    corr = gray @ gray.T
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    rows, cols = np.where(np.triu(corr, k=1) >= threshold)
    for i, j in zip(rows.tolist(), cols.tolist()):
        pairs.append((i, j, float(corr[i, j])))
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    roots = [find(i) for i in range(n)]
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids.astype(np.int64), pairs


def load_legacy_layout(source: Union[str, Path], layout: str = "personBase_invert",
                       deduplicate: bool = True, repair_seam: bool = True,
                       near_duplicate_threshold: Optional[float] = 0.97) -> LegacyDataset:
    """
    Carrega um leiaute da base legada já auditado.

    Args:
        source: ``Data.zip`` ou pasta que contém as subpastas dos leiautes.
        layout: um de ``LAYOUTS``.
        deduplicate: remove cópias exatas (SHA-256), mantendo a primeira linha.
        repair_seam: preenche a coluna nula da costura angular.
        near_duplicate_threshold: correlação mínima para agrupar quase
            duplicatas no mesmo grupo de validação cruzada (``None`` desliga).
    """
    control, diabetic = read_layout_arrays(source, layout)
    images = np.concatenate([control, diabetic], axis=0)
    labels = np.concatenate([np.zeros(len(control), np.int64), np.ones(len(diabetic), np.int64)])
    ids = np.asarray([f"control-{i:03d}" for i in range(len(control))]
                     + [f"diabetic-{i:03d}" for i in range(len(diabetic))])
    hashes = np.asarray([sha256_bytes(im) for im in images])
    multiplicity = Counter(Counter(hashes.tolist()).values())

    if deduplicate:
        keep, dup_groups = _deduplicate(images, labels, ids, hashes)
    else:
        keep, dup_groups = np.arange(len(images)), []
    images, labels, ids, hashes = images[keep], labels[keep], ids[keep], hashes[keep]

    width = images.shape[2]
    seams = [empty_columns(im) for im in images]
    mirrored = np.asarray([bool(s.size == 1 and int(s[0]) == width // 2) for s in seams])
    repaired_cols: Counter = Counter()
    if repair_seam:
        fixed = []
        for im in images:
            out, cols = repair_angular_seam(im)
            repaired_cols.update(cols)
            fixed.append(out)
        images = np.stack(fixed)

    pairs: List[Tuple[int, int, float]] = []
    if near_duplicate_threshold is not None and len(images) > 1:
        groups, pairs = near_duplicate_groups(images, near_duplicate_threshold)
    else:
        groups = np.arange(len(images), dtype=np.int64)

    metadata = {
        "layout": layout,
        "source": Path(source).name,
        "rows_control": int(len(control)),
        "rows_diabetic": int(len(diabetic)),
        "unique_control": int(np.sum(labels == 0)),
        "unique_diabetic": int(np.sum(labels == 1)),
        "exact_duplicates_removed": int(sum(len(g["removidas"]) for g in dup_groups)),
        "duplicate_multiplicity": {str(k): int(v) for k, v in sorted(multiplicity.items())},
        "duplicate_groups": dup_groups,
        "mirrored_control": int(np.sum(mirrored & (labels == 0))),
        "mirrored_diabetic": int(np.sum(mirrored & (labels == 1))),
        "seam_columns_repaired": {str(k): int(v) for k, v in sorted(repaired_cols.items())},
        "near_duplicate_threshold": near_duplicate_threshold,
        "near_duplicate_pairs": [(str(ids[i]), str(ids[j]), round(r, 4)) for i, j, r in pairs],
        "cv_groups": int(len(np.unique(groups))),
        "source_publication_doi": SOURCE_PUBLICATION_DOI,
        "retraction_notice_doi": RETRACTION_NOTICE_DOI,
        "identity_status": "sem IDs de pessoa (grupos = imagens únicas ou quase duplicatas)",
        "label_status": "diabetes informado pela base, sem verificação clínica",
        "limitations": list(LIMITATIONS),
    }
    return LegacyDataset(images=images, labels=labels, sample_ids=ids, hashes=hashes,
                         mirrored=mirrored, groups=groups, layout=layout, metadata=metadata)

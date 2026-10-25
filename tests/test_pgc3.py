"""
Testes da etapa preliminar do PGC III:
- leitura segura e auditoria da base legada (src/legacy_data.py)
- comparação estatística de modelos (src/model_comparison.py)
- olhos sintéticos e inversa exata do rubber sheet (src/synthetic_eye.py)
- invariância da faixa normalizada à dilatação da pupila
- detector de pupila e limbo (src/segmentation.py)
- fusão da textura com a geometria da pupila (src/fusion.py)
- integração com o pipeline da IC (main.py)

Não dependem da base legada: usam dados sintéticos e arquivos temporários.
Executar:  python -m unittest discover -s tests
"""

import io
import os
import pickle
import sys
import tempfile
import unittest
import warnings

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import model_comparison as mc  # noqa: E402
from src.classifiers import ClassifierFactory  # noqa: E402
from src.config import ClassifierConfig, ClassifierType  # noqa: E402
from src.fusion import evaluate_geometry_fusion  # noqa: E402
from src.legacy_data import load_legacy_layout, restricted_numpy_load  # noqa: E402
from src.segmentation import detect_eye_circles  # noqa: E402
from src.synthetic_eye import (EyeGeometry, rubber_sheet_coordinates, sample_synthetic_eye,  # noqa: E402
                               simulate_pupil_dilation)


class _Malicious:
    def __reduce__(self):
        return (os.system, ("echo inseguro",))


def _strip(img, geom, n_r=64, n_theta=256):
    import cv2
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)[None, :]
    r = np.linspace(0, 1, n_r)[:, None]
    (pcx, pcy), (icx, icy) = geom.pupil_center, geom.iris_center
    X = ((1 - r) * (pcx + geom.pupil_radius * np.cos(theta)) + r * (icx + geom.iris_radius * np.cos(theta)))
    Y = ((1 - r) * (pcy - geom.pupil_radius * np.sin(theta)) + r * (icy - geom.iris_radius * np.sin(theta)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.remap(gray, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR).astype(float)


def _write_layout(root, layout, control, diabetic):
    folder = os.path.join(root, layout)
    os.makedirs(folder, exist_ok=True)
    for name, arr in (("controlImageArr.p", control), ("diabeteImageArr.p", diabetic),
                      ("DR_type.p", np.arange(len(diabetic), dtype=np.uint8))):
        with open(os.path.join(folder, name), "wb") as fh:
            pickle.dump(arr, fh, protocol=2)


def _fake_strips(n, seed, seam_col=None):
    rng = np.random.default_rng(seed)
    imgs = rng.integers(20, 200, size=(n, 12, 40, 3), dtype=np.uint8)
    if seam_col is not None:
        imgs[:, :, seam_col, :] = 0
    return imgs


class TestLegacyData(unittest.TestCase):
    def test_restricted_unpickler_blocks_code(self):
        with self.assertRaises(pickle.UnpicklingError):
            restricted_numpy_load(pickle.dumps(_Malicious()))
        arr = np.arange(12, dtype=np.uint8).reshape(3, 4)
        for protocol in (2, 3, 4, 5):
            np.testing.assert_array_equal(restricted_numpy_load(pickle.dumps(arr, protocol=protocol)), arr)

    def test_deduplication_mirroring_and_seam(self):
        control = _fake_strips(5, 0, seam_col=39)
        control[1] = control[0]  # cópia exata
        control[3, :, 39, :] = control[3, :, 38, :]
        control[3, :, 20, :] = 0  # costura no meio: faixa espelhada
        diabetic = _fake_strips(4, 1, seam_col=39)
        diabetic[2] = diabetic[1]
        diabetic[3] = diabetic[1]
        with tempfile.TemporaryDirectory() as tmp:
            _write_layout(tmp, "personBase_invert", control, diabetic)
            ds = load_legacy_layout(tmp, "personBase_invert", near_duplicate_threshold=None)
        self.assertEqual((ds.n_control, ds.n_diabetic), (4, 2))
        self.assertEqual(ds.metadata["exact_duplicates_removed"], 3)
        self.assertEqual(int(ds.mirrored.sum()), 1)
        for im in ds.images:
            self.assertFalse(np.any(np.all(im == 0, axis=(0, 2))), "coluna nula não reparada")


class TestModelComparison(unittest.TestCase):
    def test_holm(self):
        np.testing.assert_allclose(mc.holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_corrected_ttest(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.8, 0.05, 50)
        same = mc.corrected_paired_ttest(a, a, 98, 25)
        self.assertEqual(same["diff"], 0.0)
        self.assertEqual(same["p"], 1.0)
        better = mc.corrected_paired_ttest(a + 0.2, a + rng.normal(0, 0.01, 50), 98, 25)
        self.assertLess(better["p"], 0.001)
        naive_se = np.std(a, ddof=1) / np.sqrt(50)
        _m, lo, hi = mc.corrected_mean_ci(a, 98, 25)
        self.assertGreater(hi - lo, 2 * 1.96 * naive_se, "a correção deve alargar o intervalo")

    def test_friedman_ranks(self):
        scores = np.array([[0.9, 0.7, 0.6], [0.8, 0.75, 0.5], [0.85, 0.6, 0.65], [0.95, 0.7, 0.68]])
        res = mc.friedman_nemenyi(scores)
        self.assertEqual(int(np.argmin(res["ranks"])), 0)
        self.assertAlmostEqual(res["ranks"][0], 1.0)
        self.assertGreater(res["cd"], 0)


class TestRubberSheetGeometry(unittest.TestCase):
    def test_inverse_is_exact(self):
        g = EyeGeometry((300.0, 250.0), 70.0, (360.0, 265.0), 200.0)
        rng = np.random.default_rng(1)
        s = rng.uniform(0, 1, 500)
        th = rng.uniform(0, 2 * np.pi, 500)
        x = (1 - s) * (300 + 70 * np.cos(th)) + s * (360 + 200 * np.cos(th))
        y = (1 - s) * (250 - 70 * np.sin(th)) + s * (265 - 200 * np.sin(th))
        s2, th2 = rubber_sheet_coordinates(x, y, g)
        np.testing.assert_allclose(s2, s, atol=1e-9)
        np.testing.assert_allclose(np.angle(np.exp(1j * (th2 - th))), 0, atol=1e-9)

    def test_strip_is_invariant_to_pupil_dilation(self):
        img, g = sample_synthetic_eye(np.random.default_rng(3), 0.40, 7.0)
        ref = _strip(img, g)
        for rho in (0.25, 0.55):
            warped, g2 = simulate_pupil_dilation(img, g, rho)
            r = np.corrcoef(ref.ravel(), _strip(warped, g2).ravel())[0, 1]
            self.assertGreater(r, 0.97, f"faixa mudou com rho={rho}")
            self.assertAlmostEqual(g2.rho, rho, places=6)


class TestEyeDetector(unittest.TestCase):
    def test_measures_rho_on_synthetic_eyes(self):
        rng = np.random.default_rng(5)
        for rho in (0.22, 0.45, 0.66):
            img, g = sample_synthetic_eye(rng, rho, 7.0)
            det = detect_eye_circles(img)
            self.assertLess(abs(det["pupil_radius"] / det["iris_radius"] - g.rho), 0.01)


class TestGeometryFusion(unittest.TestCase):
    def test_fusion_recovers_geometric_signal(self):
        rng = np.random.default_rng(0)
        n = 160
        y = np.repeat([0, 1], n // 2)
        texture = rng.normal(size=(n, 200)).astype(np.float32)
        rho = np.where(y == 1, 0.34, 0.42) + rng.normal(0, 0.04, n)
        geometry = np.column_stack([rho, 1 - rho, rng.uniform(0, 0.04, n)]).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = evaluate_geometry_fusion(texture, geometry, y, n_repeats=1)
        self.assertLess(res["textura"]["auc_media"], 0.65)
        self.assertGreater(res["geometria"]["auc_media"], 0.85)
        self.assertGreater(res["fusao"]["auc_media"], res["textura"]["auc_media"] + 0.2)


class TestPipelineIntegration(unittest.TestCase):
    def test_adaboost_factory_is_version_safe(self):
        model = ClassifierFactory(ClassifierConfig()).create_pipeline(ClassifierType.ADABOOST)
        X = np.random.default_rng(0).normal(size=(30, 4))
        y = np.repeat([0, 1], 15)
        model.fit(X, y)
        self.assertEqual(model.predict(X).shape, (30,))

    def test_main_loads_layout_without_duplicates(self):
        import main
        from src.config import DatasetType, get_default_config
        control = _fake_strips(4, 2, seam_col=39)
        control[2] = control[0]
        diabetic = _fake_strips(3, 3, seam_col=39)
        with tempfile.TemporaryDirectory() as tmp:
            _write_layout(tmp, "personBase", control, diabetic)
            config = get_default_config()
            config.data.data_root = tmp
            with warnings.catch_warnings(), io.StringIO() as buf:
                warnings.simplefilter("ignore")
                stdout, sys.stdout = sys.stdout, buf
                try:
                    pipeline = main.IridologyPipeline(config)
                    c, d, dr = pipeline.load_dataset(DatasetType.PERSON_BASE)
                    config.data.deduplicate = False
                    legacy = main.IridologyPipeline(config).load_dataset(DatasetType.PERSON_BASE)
                finally:
                    sys.stdout = stdout
        self.assertEqual((len(c), len(d), len(dr)), (3, 3, 3))
        self.assertEqual(len(legacy[0]), 4)


if __name__ == "__main__":
    unittest.main()

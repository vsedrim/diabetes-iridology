"""
Testes das novas funcionalidades:
- VascularizationEnhancer (Frangi, Gabor, black-hat, pipeline combinado)
- PupilValidator (classificação da geometria pupila-íris)

Não dependem do dataset: usam imagens sintéticas e geometrias conhecidas.
Executar:  python -m unittest discover -s tests
"""

import os
import sys
import unittest

import numpy as np
import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.preprocessing import create_vascularization_enhancer
from src.segmentation import create_pupil_validator
from src.config import PhotometricTransform


def _synthetic_vessels(size=200, seed=0):
    """Imagem BGR com fundo texturizado e 'fibras' finas claras (realista).

    O fundo com textura/ruído de baixa frequência aproxima o estroma da íris:
    é nesse cenário que os extratores de estrutura mostram vantagem real sobre
    brilho/contraste, que apenas reescalam intensidades.
    """
    rng = np.random.default_rng(seed)
    bg = rng.normal(90, 18, (size, size)).clip(0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    img = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    for x in range(20, size, 25):
        cv2.line(img, (x, 10), (x + 30, size - 10), (205, 205, 205), 2)
    cv2.circle(img, (size // 2, size // 2), 25, (10, 10, 10), -1)  # "pupila"
    return img


class TestVascularizationEnhancer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enhancer = create_vascularization_enhancer()
        cls.img = _synthetic_vessels()

    def test_outputs_shape_and_range(self):
        h, w = self.img.shape[:2]
        for method in [PhotometricTransform.FRANGI, PhotometricTransform.GABOR,
                       PhotometricTransform.BLACKHAT, PhotometricTransform.VESSEL]:
            out = self.enhancer.enhance(self.img, method)
            self.assertEqual(out.shape, (h, w), f"shape incorreto para {method}")
            self.assertEqual(out.dtype, np.uint8, f"dtype incorreto para {method}")
            self.assertGreaterEqual(int(out.min()), 0)
            self.assertLessEqual(int(out.max()), 255)
            self.assertGreater(int(out.max()), 0, f"{method} não respondeu a estruturas")

    def test_accepts_grayscale_input(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        out = self.enhancer.enhance_frangi(gray)
        self.assertEqual(out.shape, gray.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_quantify_metrics(self):
        out = self.enhancer.enhance_blackhat(self.img)
        metrics = self.enhancer.quantify(out)
        for key in ("structure_contrast", "ridge_energy",
                    "structure_density", "rms_contrast", "entropy"):
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)
        self.assertGreaterEqual(metrics["structure_density"], 0.0)
        self.assertLessEqual(metrics["structure_density"], 1.0)
        self.assertGreaterEqual(metrics["entropy"], 0.0)
        self.assertGreaterEqual(metrics["ridge_energy"], 0.0)

    def test_frangi_beats_brightness_on_structure(self):
        """Em fundo texturizado, Frangi separa fibras do fundo melhor que brilho."""
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        bright = cv2.convertScaleAbs(gray, alpha=1.0, beta=40)
        frangi = self.enhancer.enhance_frangi(self.img)
        m_bright = self.enhancer.quantify(bright)
        m_frangi = self.enhancer.quantify(frangi)
        self.assertGreater(m_frangi["structure_contrast"], m_bright["structure_contrast"])
        self.assertGreater(m_frangi["ridge_energy"], m_bright["ridge_energy"])

    def test_constant_image_is_safe(self):
        flat = np.full((64, 64, 3), 127, dtype=np.uint8)
        out = self.enhancer.enhance_frangi(flat)
        self.assertEqual(out.shape, (64, 64))
        self.assertEqual(int(out.max()), 0)  # sem estrutura -> mapa nulo


class TestPupilValidator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = create_pupil_validator()

    def test_normal_geometry_is_valid(self):
        r = self.validator.validate(40.0, 100.0,
                                    pupil_center=(0, 0), iris_center=(0, 0))
        self.assertTrue(r.is_valid)
        self.assertEqual(r.quality_label, "valid")
        self.assertAlmostEqual(r.pupil_iris_ratio, 0.4, places=6)
        self.assertAlmostEqual(r.iris_thickness, 60.0, places=6)
        self.assertAlmostEqual(r.pupil_diameter, 80.0, places=6)

    def test_overdilated_is_invalid(self):
        r = self.validator.validate(90.0, 100.0)
        self.assertFalse(r.is_valid)
        self.assertEqual(r.quality_label, "invalid")

    def test_overconstricted_is_invalid(self):
        r = self.validator.validate(10.0, 100.0)
        self.assertFalse(r.is_valid)
        self.assertEqual(r.quality_label, "invalid")

    def test_eccentric_flagged(self):
        r = self.validator.validate(40.0, 100.0,
                                    pupil_center=(45, 0), iris_center=(0, 0))
        self.assertGreater(r.concentricity_offset, 0.3)
        self.assertNotEqual(r.quality_label, "valid")

    def test_iris_not_larger_than_pupil(self):
        r = self.validator.validate(100.0, 80.0)
        self.assertFalse(r.is_valid)

    def test_feature_vector(self):
        r = self.validator.validate(40.0, 100.0)
        feats = self.validator.extract_features(r)
        self.assertEqual(set(feats.keys()), set(self.validator.FEATURE_NAMES))
        for v in feats.values():
            self.assertIsInstance(v, float)


if __name__ == "__main__":
    unittest.main(verbosity=2)

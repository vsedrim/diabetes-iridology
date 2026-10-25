"""
Olhos sintéticos com geometria conhecida
========================================
Gera fotografias sintéticas de olhos, com textura de íris procedural e
geometria pupila-íris controlada, para experimentos com verdade de referência
e para testes. As imagens não representam nenhuma pessoa.

Também simula a dilatação da pupila numa fotografia real, deslocando o tecido
da íris ao longo dos raios que partem do centro da pupila. O modelo linear é o
mesmo pressuposto da normalização *rubber sheet* de Daugman (o tecido se
estica uniformemente entre a pupila e o limbo); o expoente ``kappa`` produz uma
deformação não linear ilustrativa, em que o tecido próximo da pupila se
comprime mais que o tecido periférico quando a pupila dilata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class EyeGeometry:
    """Círculos da pupila e da íris, em pixels (x para a direita, y para baixo)."""

    pupil_center: Tuple[float, float]
    pupil_radius: float
    iris_center: Tuple[float, float]
    iris_radius: float

    @property
    def rho(self) -> float:
        """Razão entre os raios da pupila e da íris."""
        return self.pupil_radius / self.iris_radius

    @property
    def tau(self) -> float:
        """Espessura relativa do anel, (r_i - r_p) / r_i = 1 - rho."""
        return (self.iris_radius - self.pupil_radius) / self.iris_radius

    @property
    def eps(self) -> float:
        """Distância entre os centros normalizada pelo raio da íris."""
        return float(np.hypot(self.pupil_center[0] - self.iris_center[0],
                              self.pupil_center[1] - self.iris_center[1]) / self.iris_radius)

    def as_detection(self) -> dict:
        """Mesmo formato do detector da demonstração (chaves de centro e raio)."""
        return {"pupil_center": tuple(map(float, self.pupil_center)), "pupil_radius": float(self.pupil_radius),
                "iris_center": tuple(map(float, self.iris_center)), "iris_radius": float(self.iris_radius)}


def _wrapped_angle_diff(a: np.ndarray, b: float) -> np.ndarray:
    return np.angle(np.exp(1j * (a - b)))


def rubber_sheet_coordinates(xx: np.ndarray, yy: np.ndarray,
                             geometry: EyeGeometry) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inversa exata da normalização de Daugman com círculos não concêntricos.

    O ponto de coordenadas normalizadas (s, theta) é
    (1 - s) P(theta) + s I(theta), com P e I nos círculos da pupila e da íris.
    Para s fixo esses pontos formam um círculo de centro c_p + s (c_i - c_p) e
    raio r_p + s (r_i - r_p); logo, dado um pixel, s é a raiz de uma equação
    do segundo grau e theta sai do ângulo em relação ao centro desse círculo.
    Pixels dentro da pupila recebem s < 0 (valor -1).
    """
    cpx, cpy = geometry.pupil_center
    cix, ciy = geometry.iris_center
    a = float(geometry.pupil_radius)
    b = float(geometry.iris_radius - geometry.pupil_radius)
    dx, dy = xx - cpx, yy - cpy
    vx, vy = cix - cpx, ciy - cpy
    A = vx * vx + vy * vy - b * b
    B = -2.0 * (dx * vx + dy * vy + a * b)
    Cc = dx * dx + dy * dy - a * a
    if abs(A) < 1e-9:
        s = -Cc / B
    else:
        disc = np.sqrt(np.maximum(B * B - 4 * A * Cc, 0.0))
        s = np.maximum((-B + disc) / (2 * A), (-B - disc) / (2 * A))
    s = np.where(Cc < 0, -1.0, s)
    cx = cpx + np.clip(s, 0, None) * vx
    cy = cpy + np.clip(s, 0, None) * vy
    theta = np.mod(np.arctan2(-(yy - cy), xx - cx), 2 * np.pi)
    return s, theta


def iris_texture(rng: np.random.Generator, n_r: int = 128, n_theta: int = 512,
                 crypt_rate: float = 7.0, fiber_contrast: float = 1.0,
                 collarette: float = 0.32) -> np.ndarray:
    """
    Textura procedural da íris no domínio normalizado (linhas = raio, da borda
    pupilar ao limbo; colunas = ângulo). Combina fibras radiais (ruído filtrado
    de forma anisotrópica), detalhe fino, colarete ondulado, criptas escuras
    (quantidade de Poisson com média ``crypt_rate``), a franja pigmentar da
    borda pupilar e o anel limbar. Valores em [0, 1].
    """
    s = np.linspace(0.0, 1.0, n_r)[:, None]
    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)[None, :]
    fib = gaussian_filter(rng.standard_normal((n_r, n_theta)), sigma=(10.0, 1.4), mode=("nearest", "wrap"))
    fib /= fib.std() + 1e-9
    fine = gaussian_filter(rng.standard_normal((n_r, n_theta)), sigma=(2.5, 0.8), mode=("nearest", "wrap"))
    fine /= fine.std() + 1e-9
    tex = 0.5 + 0.12 * fiber_contrast * fib + 0.05 * fine + 0.06 * np.cos(np.pi * s)

    phase = rng.uniform(0, 2 * np.pi)
    wobble = gaussian_filter(rng.standard_normal(n_theta), sigma=12, mode="wrap")
    wobble /= np.abs(wobble).max() + 1e-9
    c_s = collarette + 0.03 * np.sin(6 * theta + phase) + 0.02 * wobble[None, :]
    tex += 0.10 * np.exp(-((s - c_s) ** 2) / (2 * 0.02 ** 2))

    for _ in range(rng.poisson(crypt_rate)):
        sk = rng.uniform(0.12, 0.65)
        tk = rng.uniform(0, 2 * np.pi)
        a_s = rng.uniform(0.03, 0.07)
        a_t = rng.uniform(0.04, 0.10)
        depth = rng.uniform(0.15, 0.30)
        tex -= depth * np.exp(-(((s - sk) / a_s) ** 2 + (_wrapped_angle_diff(theta, tk) / a_t) ** 2))

    tex -= 0.15 * np.exp(-((s / 0.03) ** 2))
    tex -= 0.12 * np.exp(-(((1 - s) / 0.05) ** 2))
    return np.clip(tex, 0.0, 1.0).astype(np.float32)


def render_eye(texture: np.ndarray, geometry: EyeGeometry, size: int = 480,
               iris_bgr: Tuple[float, float, float] = (45.0, 75.0, 120.0),
               rng: Optional[np.random.Generator] = None, noise_sigma: float = 3.0,
               blur_sigma: float = 0.8, highlight: bool = True,
               illumination: float = 1.0) -> np.ndarray:
    """
    Renderiza a fotografia sintética (BGR, uint8) de um olho com a textura dada
    mapeada no anel entre a pupila e a íris pelo modelo de Daugman (mesma
    parametrização da normalização), esclera clara, pupila escura e um reflexo
    especular pequeno dentro da pupila.
    """
    rng = rng or np.random.default_rng(0)
    n_r, n_theta = texture.shape
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cp = geometry.pupil_center
    rp = geometry.pupil_radius
    s, theta = rubber_sheet_coordinates(xx, yy, geometry)

    tex_pad = np.hstack([texture, texture[:, :2]]).astype(np.float32)
    map_t = (theta / (2 * np.pi) * n_theta).astype(np.float32)
    map_r = (np.clip(s, 0, 1) * (n_r - 1)).astype(np.float32)
    val = cv2.remap(tex_pad, map_t, map_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    r_img = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    sclera = np.array([214.0, 222.0, 234.0])
    img = np.empty((size, size, 3), np.float32)
    for c in range(3):
        img[..., c] = sclera[c] * (1.0 - 0.18 * r_img ** 2)
    iris = (s >= 0) & (s <= 1)
    for c in range(3):
        img[..., c][iris] = iris_bgr[c] * (0.35 + 1.3 * val[iris])
    pupil = s < 0
    img[pupil] = 12.0
    if highlight:
        hx, hy = cp[0] + 0.35 * rp, cp[1] - 0.35 * rp
        hr = 0.16 * rp
        spot = np.exp(-(((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * (hr / 2) ** 2)))
        img += (240.0 * spot)[..., None] * pupil[..., None]
    img *= illumination
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    if noise_sigma > 0:
        img += rng.normal(0.0, noise_sigma, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def simulate_pupil_dilation(image: np.ndarray, geometry: EyeGeometry, new_rho: float,
                            kappa: float = 0.0) -> Tuple[np.ndarray, EyeGeometry]:
    """
    Redimensiona a pupila de uma fotografia mantendo o limbo fixo. O pixel de
    coordenadas normalizadas (s, theta) na nova geometria busca o ponto
    (s ** gamma, theta) da geometria original, com
    gamma = ((1 - rho_novo) / (1 - rho_original)) ** kappa e a mesma
    parametrização de Daugman usada na normalização. ``kappa = 0`` é o modelo
    linear do rubber sheet; ``kappa > 0`` comprime mais o tecido próximo da
    pupila quando ela dilata. O interior da pupila é escalado.
    """
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cp, ci = geometry.pupil_center, geometry.iris_center
    rp0, ri = geometry.pupil_radius, geometry.iris_radius
    rp1 = float(new_rho) * ri
    new_geom = EyeGeometry(pupil_center=cp, pupil_radius=rp1, iris_center=ci, iris_radius=ri)
    s, theta = rubber_sheet_coordinates(xx, yy, new_geom)
    gamma = ((1.0 - new_rho) / (1.0 - geometry.rho)) ** kappa
    map_x, map_y = xx.copy(), yy.copy()
    ann = (s >= 0) & (s <= 1)
    s0 = s[ann] ** gamma
    cos_t, sin_t = np.cos(theta[ann]), np.sin(theta[ann])
    map_x[ann] = (1 - s0) * (cp[0] + rp0 * cos_t) + s0 * (ci[0] + ri * cos_t)
    map_y[ann] = (1 - s0) * (cp[1] - rp0 * sin_t) + s0 * (ci[1] - ri * sin_t)
    pupil = s < 0
    map_x[pupil] = cp[0] + (xx[pupil] - cp[0]) * (rp0 / rp1)
    map_y[pupil] = cp[1] + (yy[pupil] - cp[1]) * (rp0 / rp1)
    out = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT)
    return out, new_geom


def sample_synthetic_eye(rng: np.random.Generator, rho: float, crypt_rate: float,
                         size: int = 480, eps_max: float = 0.04,
                         fiber_contrast: Optional[float] = None) -> Tuple[np.ndarray, EyeGeometry]:
    """
    Sorteia um olho sintético completo: escala (raio da íris), centro, pequena
    excentricidade, cor da íris, iluminação, ruído e textura. O experimento
    controla ``rho``, ``crypt_rate`` e, opcionalmente, o contraste médio das
    fibras radiais (``fiber_contrast``, com variação individual de 10%).
    """
    r_i = rng.uniform(0.29, 0.35) * size
    cx, cy = size / 2 + rng.uniform(-12, 12), size / 2 + rng.uniform(-12, 12)
    ang = rng.uniform(0, 2 * np.pi)
    off = rng.uniform(0, eps_max) * r_i
    geom = EyeGeometry(pupil_center=(cx + off * np.cos(ang), cy + off * np.sin(ang)),
                       pupil_radius=float(np.clip(rho, 0.12, 0.8)) * r_i,
                       iris_center=(cx, cy), iris_radius=r_i)
    base = np.array([45.0, 75.0, 120.0]) * rng.uniform(0.75, 1.25, size=3)
    fiber = (fiber_contrast * rng.uniform(0.9, 1.1)) if fiber_contrast is not None else rng.uniform(0.8, 1.2)
    tex = iris_texture(rng, crypt_rate=crypt_rate, fiber_contrast=fiber)
    img = render_eye(tex, geom, size=size, iris_bgr=tuple(base), rng=rng,
                     noise_sigma=rng.uniform(1.5, 4.5), blur_sigma=rng.uniform(0.5, 1.1),
                     illumination=rng.uniform(0.85, 1.1))
    return img, geom

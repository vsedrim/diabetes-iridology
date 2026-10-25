"""
Módulo de Segmentação de Íris e Pupila
======================================
Implementa detecção e segmentação da íris e pupila usando active contours.

Técnicas implementadas:
- Detecção de pupila com limiarização adaptativa e active contours
- Detecção de íris com active contours
- Refinamento de contornos com múltiplos canais HSV

Referência:
    "Iridologia sob Lentes Científicas: Avaliação Crítica com Aprendizado de Máquina"
    UFABC - Programa de Iniciação Científica
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, NamedTuple, Dict
from dataclasses import dataclass, field
import warnings

from scipy.spatial import distance
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour

from .config import SegmentationConfig, PupilValidationConfig


@dataclass
class SegmentationResult:
    """
    Resultado da segmentação de íris e pupila.
    
    Attributes:
        pupil_center: Centro da pupila (x, y)
        pupil_radius: Raio da pupila
        pupil_contour: Contorno da pupila (N, 2)
        iris_center: Centro da íris (x, y)
        iris_radius: Raio da íris
        iris_contour: Contorno da íris (N, 2)
        success: Se a segmentação foi bem-sucedida
        message: Mensagem de erro ou sucesso
    """
    pupil_center: Tuple[int, int] = (0, 0)
    pupil_radius: int = 0
    pupil_contour: Optional[np.ndarray] = None
    iris_center: Tuple[int, int] = (0, 0)
    iris_radius: int = 0
    iris_contour: Optional[np.ndarray] = None
    success: bool = False
    message: str = ""
    
    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            'pupil_center': self.pupil_center,
            'pupil_radius': self.pupil_radius,
            'iris_center': self.iris_center,
            'iris_radius': self.iris_radius,
            'success': self.success,
            'message': self.message
        }


@dataclass
class PupilValidationResult:
    """
    Resultado da validação/classificação da geometria pupila-íris.

    Attributes:
        pupil_diameter: Diâmetro da pupila (px)
        iris_diameter: Diâmetro da íris (px)
        iris_thickness: Espessura do anel da íris, r_iris - r_pupila (px)
        pupil_iris_ratio: Razão diâmetro pupila/íris
        thickness_ratio: Espessura relativa, (r_iris - r_pupila)/r_iris
        concentricity_offset: Distância entre centros normalizada pelo raio da íris
        pupil_diameter_mm / iris_diameter_mm: diâmetros em mm (se px/mm informado)
        is_valid: Se a geometria é fisiologicamente plausível
        quality_label: "valid", "borderline" ou "invalid"
        quality_score: Pontuação contínua de qualidade em [0, 1]
        reasons: Lista de motivos que levaram à classificação
    """
    pupil_diameter: float = 0.0
    iris_diameter: float = 0.0
    iris_thickness: float = 0.0
    pupil_iris_ratio: float = 0.0
    thickness_ratio: float = 0.0
    concentricity_offset: float = 0.0
    pupil_diameter_mm: Optional[float] = None
    iris_diameter_mm: Optional[float] = None
    is_valid: bool = False
    quality_label: str = "invalid"
    quality_score: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'pupil_diameter': round(self.pupil_diameter, 2),
            'iris_diameter': round(self.iris_diameter, 2),
            'iris_thickness': round(self.iris_thickness, 2),
            'pupil_iris_ratio': round(self.pupil_iris_ratio, 4),
            'thickness_ratio': round(self.thickness_ratio, 4),
            'concentricity_offset': round(self.concentricity_offset, 4),
            'pupil_diameter_mm': (round(self.pupil_diameter_mm, 2)
                                  if self.pupil_diameter_mm is not None else None),
            'iris_diameter_mm': (round(self.iris_diameter_mm, 2)
                                 if self.iris_diameter_mm is not None else None),
            'is_valid': self.is_valid,
            'quality_label': self.quality_label,
            'quality_score': round(self.quality_score, 3),
            'reasons': list(self.reasons),
        }


class PupilValidator:
    """
    Valida e classifica a geometria pupila-íris.

    Classifica a geometria pupila-íris (validação da pupila usando o diâmetro
    da pupila e a espessura da íris). A classificação é baseada
    em regras fisiológicas interpretáveis sobre três descritores:

    1. Razão diâmetro pupila/íris (dilatação/constrição plausível)
    2. Espessura relativa do anel da íris
    3. Excentricidade (quão concêntricos são pupila e íris)

    Além do rótulo, expõe um vetor de features (extract_features) pronto para
    alimentar um classificador supervisionado (ex.: LogisticRegression/SVM),
    caso se queira aprender os limiares a partir de dados rotulados.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "pupil_diameter",
        "iris_diameter",
        "iris_thickness",
        "pupil_iris_ratio",
        "thickness_ratio",
        "concentricity_offset",
    )

    def __init__(self, config: Optional[PupilValidationConfig] = None):
        self.config = config or PupilValidationConfig()

    def validate(self,
                 pupil_radius: float,
                 iris_radius: float,
                 pupil_center: Optional[Tuple[float, float]] = None,
                 iris_center: Optional[Tuple[float, float]] = None
                 ) -> PupilValidationResult:
        """
        Valida a geometria a partir de raios (e centros opcionais).

        Args:
            pupil_radius: Raio da pupila (px)
            iris_radius: Raio da íris (px)
            pupil_center: Centro da pupila (x, y), opcional
            iris_center: Centro da íris (x, y), opcional

        Returns:
            PupilValidationResult preenchido e classificado.
        """
        cfg = self.config
        result = PupilValidationResult()
        reasons: List[str] = []

        # Geometria básica
        if iris_radius <= 0 or pupil_radius <= 0:
            result.reasons = ["raios inválidos (<= 0)"]
            return result
        if iris_radius <= pupil_radius:
            result.reasons = ["íris menor ou igual à pupila"]
            result.pupil_diameter = 2.0 * pupil_radius
            result.iris_diameter = 2.0 * iris_radius
            return result

        result.pupil_diameter = 2.0 * pupil_radius
        result.iris_diameter = 2.0 * iris_radius
        result.iris_thickness = float(iris_radius - pupil_radius)
        result.pupil_iris_ratio = float(pupil_radius / iris_radius)
        result.thickness_ratio = float((iris_radius - pupil_radius) / iris_radius)

        # Excentricidade (se centros disponíveis)
        if pupil_center is not None and iris_center is not None:
            offset = float(np.hypot(pupil_center[0] - iris_center[0],
                                    pupil_center[1] - iris_center[1]))
            result.concentricity_offset = offset / float(iris_radius)

        # Conversão opcional para mm
        if cfg.pixels_per_mm:
            result.pupil_diameter_mm = result.pupil_diameter / cfg.pixels_per_mm
            result.iris_diameter_mm = result.iris_diameter / cfg.pixels_per_mm

        # --- Regras de classificação ---
        ratio = result.pupil_iris_ratio
        score = 1.0

        # 1) Razão pupila/íris
        if ratio < cfg.borderline_low_ratio or ratio > cfg.borderline_high_ratio:
            reasons.append(
                f"razão pupila/íris {ratio:.2f} fora da faixa plausível "
                f"[{cfg.borderline_low_ratio:.2f}, {cfg.borderline_high_ratio:.2f}]"
            )
            score -= 0.6
        elif ratio < cfg.min_pupil_iris_ratio or ratio > cfg.max_pupil_iris_ratio:
            reasons.append(
                f"razão pupila/íris {ratio:.2f} na faixa de atenção "
                f"[{cfg.min_pupil_iris_ratio:.2f}, {cfg.max_pupil_iris_ratio:.2f}]"
            )
            score -= 0.25

        # 2) Espessura relativa da íris
        if result.thickness_ratio < cfg.min_thickness_ratio:
            reasons.append(
                f"anel da íris fino demais (espessura relativa "
                f"{result.thickness_ratio:.2f} < {cfg.min_thickness_ratio:.2f})"
            )
            score -= 0.3

        # 3) Excentricidade
        if result.concentricity_offset > cfg.max_concentricity_offset:
            reasons.append(
                f"centros pouco concêntricos (offset "
                f"{result.concentricity_offset:.2f} > {cfg.max_concentricity_offset:.2f})"
            )
            score -= 0.3

        score = float(max(0.0, min(1.0, score)))
        result.quality_score = score

        # Rótulo final
        hard_fail = any("fora da faixa plausível" in r for r in reasons)
        if hard_fail or score < 0.5:
            result.quality_label = "invalid"
            result.is_valid = False
        elif score < 0.8 or reasons:
            result.quality_label = "borderline"
            result.is_valid = True
        else:
            result.quality_label = "valid"
            result.is_valid = True

        result.reasons = reasons if reasons else ["geometria dentro das faixas esperadas"]
        return result

    def validate_segmentation(self, seg: "SegmentationResult") -> PupilValidationResult:
        """Valida a partir de um SegmentationResult do IrisSegmenter."""
        if not seg.success:
            res = PupilValidationResult()
            res.reasons = [f"segmentação falhou: {seg.message}"]
            return res
        return self.validate(
            pupil_radius=seg.pupil_radius,
            iris_radius=seg.iris_radius,
            pupil_center=seg.pupil_center,
            iris_center=seg.iris_center,
        )

    def extract_features(self, result: PupilValidationResult) -> Dict[str, float]:
        """Vetor de features (pronto para um classificador supervisionado)."""
        return {
            "pupil_diameter": result.pupil_diameter,
            "iris_diameter": result.iris_diameter,
            "iris_thickness": result.iris_thickness,
            "pupil_iris_ratio": result.pupil_iris_ratio,
            "thickness_ratio": result.thickness_ratio,
            "concentricity_offset": result.concentricity_offset,
        }


class IrisSegmenter:
    """
    Segmentador de íris e pupila usando active contours.
    
    Implementa o algoritmo de segmentação descrito no artigo,
    utilizando:
    1. Pré-processamento morfológico
    2. Limiarização adaptativa para inicialização
    3. Active contours (snakes) para refinamento
    4. Validação geométrica dos resultados
    """
    
    def __init__(self, config: Optional[SegmentationConfig] = None):
        """
        Inicializa o segmentador.
        
        Args:
            config: Configuração de segmentação. Se None, usa padrão.
        """
        self.config = config or SegmentationConfig()
    
    def _create_circle_contour(self, center_x: float, center_y: float, 
                                radius: float, n_points: int = 400) -> np.ndarray:
        """
        Cria um contorno circular.
        
        Args:
            center_x: Coordenada x do centro
            center_y: Coordenada y do centro
            radius: Raio do círculo
            n_points: Número de pontos no contorno
            
        Returns:
            Array (n_points, 2) com coordenadas do contorno
        """
        s = np.linspace(0, 2 * np.pi, n_points)
        x = center_x + radius * np.cos(s)
        y = center_y + radius * np.sin(s)
        return np.column_stack([x, y])
    
    def _get_contour_area(self, contour: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calcula área de um contorno.
        
        Args:
            contour: Contorno (N, 2)
            
        Returns:
            Tupla (área, contorno formatado para OpenCV)
        """
        cnt = np.int32(contour).reshape(-1, 1, 2)
        area = cv2.contourArea(cnt)
        return area, cnt
    
    def _fit_circle_to_contour(self, contour: np.ndarray
                                ) -> Tuple[Tuple[int, int], int, np.ndarray]:
        """
        Ajusta um círculo a um contorno.
        
        Args:
            contour: Contorno (N, 2) ou (N, 1, 2)
            
        Returns:
            Tupla (centro, raio, contorno_circular)
        """
        if len(contour.shape) == 2:
            cnt = contour.reshape(-1, 1, 2).astype(np.float32)
        else:
            cnt = contour.astype(np.float32)
        
        center, radius = cv2.minEnclosingCircle(cnt)
        center = (int(center[0]), int(center[1]))
        radius = int(radius)
        
        # Cria contorno circular
        circle_contour = self._create_circle_contour(center[0], center[1], radius)
        
        return center, radius, circle_contour
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pré-processa imagem para segmentação.
        
        Args:
            image: Imagem BGR
            
        Returns:
            Tupla (imagem_processada, imagem_hsv)
        """
        # Redimensiona se necessário
        h, w = image.shape[:2]
        target_size = self.config.target_size
        
        if (h, w) != target_size:
            image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # Aplica blur e operações morfológicas
        processed = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Downsampling para processamento mais rápido
        for _ in range(self.config.downsample_levels):
            processed = cv2.pyrDown(processed)
        
        # Operações morfológicas para reduzir ruído
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            self.config.morph_kernel_size
        )
        processed = cv2.erode(processed, kernel)
        processed = cv2.dilate(processed, kernel)
        
        # Converte para HSV
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
        return processed, hsv
    
    def _detect_pupil(self, hsv_image: np.ndarray, 
                       processed_image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], int]:
        """
        Detecta a pupila usando limiarização e active contours.
        
        Args:
            hsv_image: Imagem HSV
            processed_image: Imagem pré-processada BGR
            
        Returns:
            Tupla (contorno_pupila, centro, raio)
        """
        # Usa canal V (Value) para detecção inicial
        value_channel = hsv_image[:, :, 2]
        
        h, w = value_channel.shape
        
        # Limiarização adaptativa
        threshold = self.config.pupil_threshold_init
        _, thresh = cv2.threshold(value_channel, threshold, 255, cv2.THRESH_BINARY)
        
        # Converte para float para active contour
        thresh_float = rgb2gray(thresh) if len(thresh.shape) == 3 else thresh.astype(float)
        
        # Inicializa contorno circular no centro da imagem
        init_radius = self.config.pupil_init_radius
        init = self._create_circle_contour(w / 2, h / 2, init_radius)
        
        # Aplica active contour
        snake = active_contour(
            gaussian(thresh_float, 3), 
            init,
            alpha=self.config.snake_alpha,
            beta=self.config.snake_beta
        )
        
        area, cnt = self._get_contour_area(snake)
        
        # Ajusta threshold se área muito pequena
        attempts = 0
        while area < self.config.min_pupil_area and attempts < 10:
            threshold += 10
            _, thresh = cv2.threshold(value_channel, threshold, 255, cv2.THRESH_BINARY_INV)
            thresh_float = thresh.astype(float)
            
            snake = active_contour(
                gaussian(thresh_float, 3),
                init,
                alpha=self.config.snake_alpha,
                beta=self.config.snake_beta
            )
            area, cnt = self._get_contour_area(snake)
            attempts += 1
        
        # Refinamento com canal S (Saturation)
        sat_channel = hsv_image[:, :, 1]
        snake_refined = active_contour(
            gaussian(sat_channel, 3),
            snake,
            alpha=self.config.snake_alpha,
            beta=self.config.snake_beta
        )
        
        area_refined, cnt_refined = self._get_contour_area(snake_refined)
        
        # Usa contorno refinado se área razoável
        if area_refined > 0 and not (area > 5 * area_refined):
            snake = snake_refined
            cnt = cnt_refined
            area = area_refined
        
        # Calcula centro e raio
        center, radius, _ = self._fit_circle_to_contour(snake)
        
        return snake, center, radius
    
    def _detect_iris(self, hsv_image: np.ndarray,
                      pupil_contour: np.ndarray,
                      pupil_center: Tuple[int, int],
                      pupil_radius: int) -> Tuple[np.ndarray, Tuple[int, int], int]:
        """
        Detecta a íris usando active contours expandindo da pupila.
        
        Args:
            hsv_image: Imagem HSV
            pupil_contour: Contorno da pupila
            pupil_center: Centro da pupila
            pupil_radius: Raio da pupila
            
        Returns:
            Tupla (contorno_iris, centro, raio)
        """
        value_channel = hsv_image[:, :, 2]
        value_channel = cv2.medianBlur(value_channel, 5)
        
        h, w = value_channel.shape
        
        # Inicializa contorno maior que a pupila
        if pupil_radius < 40:
            init_radius = 2 * pupil_radius
        else:
            init_radius = 1.4 * pupil_radius
        
        init = self._create_circle_contour(pupil_center[0], pupil_center[1], init_radius)
        
        # Active contour com parâmetros para expansão
        convergence = self.config.iris_convergence
        pupil_area, _ = self._get_contour_area(pupil_contour)
        iris_area = 10**20
        
        attempts = 0
        while iris_area > 10 * pupil_area and attempts < 10:
            snake = active_contour(
                gaussian(value_channel, 3),
                init,
                alpha=self.config.iris_snake_alpha,
                beta=self.config.snake_beta,
                gamma=self.config.iris_snake_gamma,
                convergence=convergence
            )
            iris_area, cnt = self._get_contour_area(snake)
            convergence *= 1.2
            attempts += 1
        
        # Ajusta círculo ao contorno
        center, radius, circle_contour = self._fit_circle_to_contour(snake)
        
        # Verifica se contorno excede limites da imagem
        if center[1] + radius > h:
            # Recalcula usando distância mínima do centro ao contorno
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                center = (cx, cy)
                
                # Calcula raio como distância mínima
                distances = [distance.euclidean(center, (snake[i, 0], snake[i, 1])) 
                            for i in range(len(snake))]
                radius = int(min(distances))
                
                circle_contour = self._create_circle_contour(cx, cy, radius)
        
        # Ajusta se ainda excede
        if center[1] + radius > h:
            diff = center[1] + radius - h + 1
            radius = radius - diff
            circle_contour = self._create_circle_contour(center[0], center[1], radius)
        
        return circle_contour, center, radius
    
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """
        Realiza segmentação completa de íris e pupila.
        
        Args:
            image: Imagem BGR do olho
            
        Returns:
            SegmentationResult com contornos e parâmetros
        """
        result = SegmentationResult()
        
        try:
            # Pré-processamento
            processed, hsv = self._preprocess_image(image)
            
            # Detecta pupila
            pupil_contour, pupil_center, pupil_radius = self._detect_pupil(hsv, processed)
            
            if pupil_radius < self.config.min_pupil_radius:
                result.message = f"Pupila muito pequena: raio={pupil_radius}"
                return result
            
            # Detecta íris
            iris_contour, iris_center, iris_radius = self._detect_iris(
                hsv, pupil_contour, pupil_center, pupil_radius
            )
            
            if iris_radius < pupil_radius:
                result.message = f"Íris menor que pupila: iris={iris_radius}, pupila={pupil_radius}"
                return result
            
            # Preenche resultado
            result.pupil_center = pupil_center
            result.pupil_radius = pupil_radius
            result.pupil_contour = pupil_contour
            result.iris_center = iris_center
            result.iris_radius = iris_radius
            result.iris_contour = iris_contour
            result.success = True
            result.message = "Segmentação bem-sucedida"
            
        except Exception as e:
            result.message = f"Erro na segmentação: {str(e)}"
        
        return result
    
    def segment_batch(self, images: np.ndarray, 
                       verbose: bool = True) -> List[SegmentationResult]:
        """
        Segmenta múltiplas imagens.
        
        Args:
            images: Array de imagens (N, H, W, C)
            verbose: Se True, mostra progresso
            
        Returns:
            Lista de SegmentationResult
        """
        results = []
        n_images = len(images)
        
        for i, img in enumerate(images):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Segmentando imagem {i + 1}/{n_images}...")
            
            result = self.segment(img)
            results.append(result)
        
        success_count = sum(1 for r in results if r.success)
        if verbose:
            print(f"  Segmentação concluída: {success_count}/{n_images} bem-sucedidas")
        
        return results
    
    def visualize(self, image: np.ndarray, 
                  result: SegmentationResult,
                  show: bool = True) -> np.ndarray:
        """
        Visualiza resultado da segmentação.
        
        Args:
            image: Imagem original
            result: Resultado da segmentação
            show: Se True, exibe a imagem
            
        Returns:
            Imagem com contornos desenhados
        """
        import matplotlib.pyplot as plt
        
        vis_image = image.copy()
        
        if result.success:
            # Desenha contorno da pupila (verde)
            if result.pupil_contour is not None:
                pts = result.pupil_contour.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis_image, [pts], True, (0, 255, 0), 2)
            
            # Desenha contorno da íris (vermelho)
            if result.iris_contour is not None:
                pts = result.iris_contour.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis_image, [pts], True, (0, 0, 255), 2)
            
            # Desenha centros
            cv2.circle(vis_image, result.pupil_center, 3, (0, 255, 0), -1)
            cv2.circle(vis_image, result.iris_center, 3, (0, 0, 255), -1)
        
        if show:
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Segmentação: {result.message}")
            plt.axis('off')
            plt.show()
        
        return vis_image


def create_segmenter(config: Optional[SegmentationConfig] = None) -> IrisSegmenter:
    """
    Factory function para criar segmentador.
    
    Args:
        config: Configuração de segmentação
        
    Returns:
        IrisSegmenter configurado
    """
    return IrisSegmenter(config)


def create_pupil_validator(
        config: Optional[PupilValidationConfig] = None) -> PupilValidator:
    """Factory function para criar o validador de pupila."""
    return PupilValidator(config)


# =============================================================================
# Refinamento de fronteiras circulares (operador integro-diferencial)
# =============================================================================

def refine_circle(gray: np.ndarray,
                  center: Tuple[float, float],
                  radius: float,
                  radius_range: Tuple[float, float] = (0.7, 1.3),
                  max_shift: Optional[float] = None,
                  shift_step: float = 1.0,
                  radius_step: float = 0.5,
                  sigma: float = 1.5,
                  n_angles: int = 180,
                  arcs: Optional[List[Tuple[float, float]]] = None,
                  polarity: int = 1,
                  first_peak_fraction: Optional[float] = None) -> Tuple[float, float, float, float]:
    """
    Refina um círculo pelo operador integro-diferencial de Daugman.

    Para cada centro candidato numa vizinhança de ``center`` e cada raio em
    ``radius * radius_range``, calcula a intensidade média ao longo do círculo
    (somente nos arcos indicados) e escolhe o par (centro, raio) que maximiza a
    derivada radial suavizada por uma gaussiana:

        max_{r, x0, y0} | G_sigma(r) * d/dr  (1 / 2 pi r) \\oint I(x, y) ds |

    ``polarity = +1`` procura transições de escuro para claro no sentido do
    raio crescente (pupila -> íris e íris -> esclera); ``-1`` procura o oposto;
    ``0`` aceita as duas. Os arcos são pares (início, fim) em graus, medidos no
    sentido anti-horário a partir do eixo x; ``None`` usa o círculo inteiro.
    Para o limbo, restringir aos arcos laterais evita as pálpebras.

    Com ``first_peak_fraction`` (por exemplo, 0,5), o raio escolhido para cada
    centro é o primeiro máximo local da derivada, a partir do centro, que
    alcança essa fração do maior valor da faixa. Na pupila isso impede que o
    colarete ou outra borda interna da íris, às vezes mais abrupta que a borda
    pupilar, seja tomado pela fronteira da pupila.

    Returns:
        (cx, cy, r, intensidade_da_borda)
    """
    from scipy.ndimage import gaussian_filter1d, map_coordinates

    img = gray.astype(np.float32)
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)

    if arcs is None:
        theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    else:
        parts = [np.deg2rad(np.linspace(a, b, max(8, int(n_angles * (b - a) / 360.0)), endpoint=False))
                 for a, b in arcs]
        theta = np.concatenate(parts)
    cos_t, sin_t = np.cos(theta), -np.sin(theta)

    radii = np.arange(radius * radius_range[0], radius * radius_range[1] + radius_step, radius_step)
    radii = radii[radii > 2]
    if max_shift is None:
        max_shift = max(2.0, 0.08 * radius)
    shifts = np.arange(-max_shift, max_shift + 1e-9, shift_step)

    best = (float(center[0]), float(center[1]), float(radius), -np.inf)
    h, w = img.shape
    for dy in shifts:
        for dx in shifts:
            cx, cy = center[0] + dx, center[1] + dy
            xs = cx + radii[:, None] * cos_t[None, :]
            ys = cy + radii[:, None] * sin_t[None, :]
            inside = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
            vals = map_coordinates(img, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(xs.shape)
            vals = np.where(inside, vals, np.nan)
            profile = np.nanmean(vals, axis=1)
            if np.isnan(profile).any():
                continue
            deriv = np.gradient(gaussian_filter1d(profile, sigma / radius_step), radius_step)
            score = deriv * polarity if polarity != 0 else np.abs(deriv)
            k = int(np.argmax(score))
            if first_peak_fraction is not None and score[k] > 0:
                limit = first_peak_fraction * score[k]
                peaks = [i for i in range(1, len(score) - 1)
                         if score[i] >= limit and score[i] >= score[i - 1] and score[i] >= score[i + 1]]
                if peaks:
                    k = peaks[0]
            if 0 < k < len(radii) - 1 and score[k] > best[3]:
                best = (float(cx), float(cy), float(radii[k]), float(score[k]))
    return best


LIMBUS_ARCS: List[Tuple[float, float]] = [(-45.0, 45.0), (135.0, 225.0)]


def refine_pupil_iris(image: np.ndarray, detection: Dict) -> Dict:
    """
    Refina uma detecção grosseira de pupila e íris (dicionário com
    ``pupil_center``, ``pupil_radius``, ``iris_center`` e ``iris_radius``)
    pelo operador integro-diferencial, em duas passagens (grossa e fina).
    A pupila usa o círculo inteiro; o limbo usa só os arcos laterais, para
    que as pálpebras não deformem a estimativa. Devolve um novo dicionário com
    os círculos refinados e as chaves originais preservadas.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pr0 = float(detection["pupil_radius"])
    ir0 = float(detection["iris_radius"])
    upper = min(1.25, 0.85 * ir0 / pr0) if pr0 > 0 else 1.25
    pcx, pcy, pr, _ = refine_circle(gray, detection["pupil_center"], pr0, radius_range=(0.6, max(upper, 0.7)),
                                    max_shift=min(12.0, max(3.0, 0.15 * pr0)), shift_step=2.0, radius_step=0.5,
                                    first_peak_fraction=0.5)
    pcx, pcy, pr, _ = refine_circle(gray, (pcx, pcy), pr, radius_range=(0.92, 1.08), max_shift=2.0,
                                    shift_step=0.5, radius_step=0.25, first_peak_fraction=0.5)
    icx, icy, ir, _ = refine_circle(gray, detection["iris_center"], ir0, radius_range=(0.85, 1.15),
                                    max_shift=max(3.0, 0.06 * ir0), shift_step=2.0, radius_step=1.0,
                                    arcs=LIMBUS_ARCS)
    icx, icy, ir, _ = refine_circle(gray, (icx, icy), ir, radius_range=(0.96, 1.04), max_shift=2.0,
                                    shift_step=0.5, radius_step=0.5, arcs=LIMBUS_ARCS)
    refined = dict(detection)
    refined.update({"pupil_center": (pcx, pcy), "pupil_radius": pr,
                    "iris_center": (icx, icy), "iris_radius": ir, "refined": True})
    return refined


def fit_circle_least_squares(points: np.ndarray) -> Tuple[float, float, float]:
    """Ajuste algébrico de círculo (Kåsa) por mínimos quadrados: (cx, cy, r)."""
    pts = np.asarray(points, dtype=float)
    A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    b = pts[:, 0] ** 2 + pts[:, 1] ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    return float(cx), float(cy), float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def ransac_circle(points: np.ndarray, n_iter: int = 300, tol: float = 2.0,
                  seed: int = 0) -> Tuple[Tuple[float, float, float], np.ndarray]:
    """
    Ajuste robusto de círculo por RANSAC (Fischler; Bolles, 1981): sorteia
    trios de pontos, conta os inliers a menos de ``tol`` pixels do círculo e
    reajusta por mínimos quadrados sobre o maior consenso.
    """
    pts = np.asarray(points, dtype=float)
    rng = np.random.default_rng(seed)
    best = np.zeros(len(pts), dtype=bool)
    if len(pts) < 3:
        return (float("nan"),) * 3, best
    for _ in range(n_iter):
        p = pts[rng.choice(len(pts), 3, replace=False)]
        try:
            cx, cy, r = fit_circle_least_squares(p)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(r) or r <= 0:
            continue
        inl = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < tol
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 3:
        best = np.ones(len(pts), dtype=bool)
    return fit_circle_least_squares(pts[best]), best


def _ray_profiles(img: np.ndarray, center: Tuple[float, float], angles: np.ndarray,
                  radii: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.ndimage import map_coordinates
    xs = center[0] + radii[None, :] * np.cos(angles)[:, None]
    ys = center[1] - radii[None, :] * np.sin(angles)[:, None]
    prof = map_coordinates(img, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(xs.shape)
    return prof, xs, ys


def detect_eye_circles(image: np.ndarray, n_rays: int = 72) -> Dict:
    """
    Detecta a pupila e o limbo numa fotografia do olho sem depender da faixa
    de razões pupila/íris assumida pelo detector da demonstração:

    1. semente: ponto mais escuro, após suavização, na região central;
    2. pupila: raios em estrela a partir da semente (*starburst*; Li, Winfield
       e Parkhurst, 2005); em cada raio, o primeiro máximo local da derivada
       de escuro para claro que alcança 30% da maior derivada do raio; círculo
       por RANSAC e refinamento integro-diferencial (Daugman, 1993);
    3. limbo: raios nos arcos laterais a partir do centro da pupila; em cada
       raio, a maior derivada de escuro para claro além de 1,25 r_p; círculo
       por RANSAC e refinamento integro-diferencial nos mesmos arcos.

    Devolve um dicionário no formato do detector da demonstração, com
    ``detector = "starburst"`` e as frações de inliers dos dois ajustes.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0)

    y0, y1, x0, x1 = int(0.15 * h), int(0.85 * h), int(0.15 * w), int(0.85 * w)
    heavy = cv2.GaussianBlur(g, (0, 0), 6.0)
    iy, ix = np.unravel_index(int(np.argmin(heavy[y0:y1, x0:x1])), heavy[y0:y1, x0:x1].shape)
    seed = (float(x0 + ix), float(y0 + iy))

    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    r_max = 0.45 * min(h, w)
    radii = np.arange(1.0, r_max, 0.5)
    prof, xs, ys = _ray_profiles(g, seed, angles, radii)
    deriv_p = np.gradient(cv2.GaussianBlur(prof, (0, 0), sigmaX=3.0, sigmaY=0.01), axis=1)
    pts = []
    for i in range(n_rays):
        di = deriv_p[i]
        top = float(di.max())
        if top <= 0:
            continue
        peaks = np.flatnonzero((di[1:-1] >= 0.3 * top) & (di[1:-1] >= di[:-2]) & (di[1:-1] >= di[2:])) + 1
        if peaks.size:
            pts.append((xs[i, peaks[0]], ys[i, peaks[0]]))
    (pcx, pcy, pr), inl_p = ransac_circle(np.asarray(pts), tol=2.0) if len(pts) >= 3 else \
        ((seed[0], seed[1], 0.08 * min(h, w)), np.zeros(0, bool))
    # Um ajuste degenerado (pontos quase colineares) produz raios enormes; nesse caso
    # volta-se à distância mediana dos pontos de borda até a semente.
    if (not np.isfinite(pr) or pr < 3 or pr > r_max
            or np.hypot(pcx - seed[0], pcy - seed[1]) > r_max):
        dist = np.hypot(np.asarray(pts)[:, 0] - seed[0], np.asarray(pts)[:, 1] - seed[1]) if pts else []
        pcx, pcy = seed
        pr = float(np.clip(np.median(dist), 3, r_max)) if len(dist) else 0.08 * min(h, w)
    pcx, pcy, pr, _ = refine_circle(gray, (pcx, pcy), pr, radius_range=(0.75, 1.3),
                                    max_shift=min(12.0, max(2.0, 0.1 * pr)), shift_step=2.0, radius_step=0.5,
                                    first_peak_fraction=0.5)
    pcx, pcy, pr, _ = refine_circle(gray, (pcx, pcy), pr, radius_range=(0.93, 1.07), max_shift=1.5,
                                    shift_step=0.5, radius_step=0.25, first_peak_fraction=0.5)

    lat = np.deg2rad(np.concatenate([np.linspace(-40, 40, n_rays // 3), np.linspace(140, 220, n_rays // 3)]))
    radii_i = np.arange(1.25 * pr, max(1.3 * pr + 2, min(6.0 * pr, 0.6 * max(h, w))), 0.5)
    prof_i, xs_i, ys_i = _ray_profiles(g, (pcx, pcy), lat, radii_i)
    valid = (xs_i >= 1) & (xs_i <= w - 2) & (ys_i >= 1) & (ys_i <= h - 2)
    deriv = np.gradient(cv2.GaussianBlur(prof_i, (0, 0), sigmaX=4.0, sigmaY=0.01), axis=1)
    deriv[~valid] = -np.inf
    k = np.argmax(deriv, axis=1)
    pts_i = np.column_stack([xs_i[np.arange(len(lat)), k], ys_i[np.arange(len(lat)), k]])
    (icx, icy, ir), inl_i = ransac_circle(pts_i, tol=3.0)
    r_max_iris = 0.75 * max(h, w)
    if (not np.isfinite(ir) or ir <= pr or ir > r_max_iris
            or np.hypot(icx - pcx, icy - pcy) > ir):
        icx, icy, ir = pcx, pcy, float(np.median(np.hypot(pts_i[:, 0] - pcx, pts_i[:, 1] - pcy)))
    icx, icy, ir, _ = refine_circle(gray, (icx, icy), ir, radius_range=(0.9, 1.1),
                                    max_shift=min(12.0, max(3.0, 0.05 * ir)), shift_step=2.0, radius_step=0.5,
                                    arcs=LIMBUS_ARCS)
    icx, icy, ir, _ = refine_circle(gray, (icx, icy), ir, radius_range=(0.97, 1.03), max_shift=1.5,
                                    shift_step=0.5, radius_step=0.25, arcs=LIMBUS_ARCS)
    return {"pupil_center": (pcx, pcy), "pupil_radius": pr, "iris_center": (icx, icy), "iris_radius": ir,
            "pupil_estimated": len(pts) < n_rays // 3, "iris_estimated": int(inl_i.sum()) < len(lat) // 3,
            "pupil_inlier_fraction": float(inl_p.mean()) if inl_p.size else 0.0,
            "iris_inlier_fraction": float(inl_i.mean()) if inl_i.size else 0.0,
            "detector": "starburst"}

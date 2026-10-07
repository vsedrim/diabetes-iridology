"""
Módulo de Pré-processamento de Imagens
======================================
Implementa transformações fotométricas e normalização de imagens de íris.

Transformações implementadas conforme metodologia do artigo:
- Equalização global de histograma
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian blur

Referência:
    "Iridologia sob Lentes Científicas: Avaliação Crítica com Aprendizado de Máquina"
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union, Dict
from enum import Enum

from skimage.filters import frangi
from skimage.util import img_as_float
from skimage.measure import shannon_entropy

from .config import PreprocessingConfig, PhotometricTransform, ColorChannel


class VascularizationEnhancer:
    """
    Realce de vascularização e estroma da íris por algoritmos consolidados.

    Motivação: ir além de brilho/contraste simples e
    equalização, usando métodos já estabelecidos em visão computacional e
    imagem médica para evidenciar estruturas tubulares/fibrosas (vasos,
    fibras radiais, criptas):

    1. Frangi vesselness (realce Hessiano multiescala)
       Frangi et al., "Multiscale vessel enhancement filtering", MICCAI 1998.
       Resposta construída a partir dos autovalores da matriz Hessiana em
       várias escalas; alta para estruturas tubulares, baixa para blobs/planos.

    2. Banco de filtros de Gabor orientados
       Resposta máxima entre orientações realça fibras/vasos direcionais;
       base do reconhecimento de íris de Daugman e de realce de textura.

    3. Top-hat / black-hat morfológico multiescala
       Soares et al. e literatura de retina: extrai estruturas finas, claras
       (top-hat) ou escuras (black-hat), em múltiplas escalas de elemento
       estruturante, independente de iluminação de fundo.

    Pré-processamento comum: canal verde (maior contraste de vasos em imagens
    RGB) e CLAHE, prática padrão em realce de fundo de olho/íris.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size
        )
        self._gabor_kernels = self._build_gabor_bank()

    # ----------------------------- utilitários -----------------------------
    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        """Normaliza min-max para [0, 255] uint8 de forma segura."""
        img = image.astype(np.float64)
        mn, mx = float(img.min()), float(img.max())
        if mx - mn < 1e-12:
            return np.zeros(img.shape, dtype=np.uint8)
        norm = (img - mn) / (mx - mn)
        return (norm * 255.0).astype(np.uint8)

    @staticmethod
    def _stretch_display(response: np.ndarray,
                         p_low: float = 2.0, p_high: float = 99.0,
                         gamma: float = 0.7) -> np.ndarray:
        """
        Normalização de exibição por percentil + gama para respostas de filtro.

        A resposta do Frangi é esparsa e concentrada em valores baixos; o
        recorte por percentil e a correção de gama tornam visíveis as cristas
        reais sem alterar o que foi detectado (prática padrão de visualização).
        """
        r = response.astype(np.float64)
        lo = float(np.percentile(r, p_low))
        hi = float(np.percentile(r, p_high))
        if hi - lo < 1e-12:
            return VascularizationEnhancer._to_uint8(r)
        r = np.clip((r - lo) / (hi - lo), 0.0, 1.0)
        if gamma and gamma != 1.0:
            r = np.power(r, gamma)
        return (r * 255.0).astype(np.uint8)

    def _prep(self, image: np.ndarray, equalize: bool = True) -> np.ndarray:
        """Usa o canal verde (padrão na literatura de retina) e aplica CLAHE."""
        if image.ndim == 3:
            gray = image[:, :, 1]  # canal verde no BGR do OpenCV
        else:
            gray = image
        gray = gray.astype(np.uint8)
        if equalize:
            gray = self._clahe.apply(gray)
        return gray

    def _build_gabor_bank(self) -> list:
        kernels = []
        n = max(1, int(self.config.gabor_n_orientations))
        ksize = int(self.config.gabor_ksize)
        for i in range(n):
            theta = np.pi * i / n
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                self.config.gabor_sigma,
                theta,
                self.config.gabor_lambda,
                self.config.gabor_gamma,
                0,
                ktype=cv2.CV_32F
            )
            kernel -= kernel.mean()  # resposta DC nula
            kernels.append(kernel)
        return kernels

    # ------------------------------ algoritmos ------------------------------
    def enhance_frangi(self, image: np.ndarray) -> np.ndarray:
        """Realce de vascularização de Frangi (vesselness Hessiano multiescala)."""
        gray = self._prep(image)
        f = img_as_float(gray)
        vessels = frangi(
            f,
            sigmas=self.config.frangi_sigmas,
            beta=self.config.frangi_beta,
            gamma=self.config.frangi_gamma,
            black_ridges=self.config.frangi_black_ridges
        )
        return self._stretch_display(vessels)

    def enhance_gabor(self, image: np.ndarray) -> np.ndarray:
        """Resposta máxima de um banco de filtros de Gabor orientados."""
        gray = self._prep(image).astype(np.float32)
        response = np.zeros_like(gray)
        for kernel in self._gabor_kernels:
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            np.maximum(response, np.abs(filtered), out=response)
        return self._to_uint8(response)

    def enhance_blackhat(self, image: np.ndarray, mode: str = "both") -> np.ndarray:
        """Top-hat/black-hat morfológico multiescala (estruturas finas)."""
        gray = self._prep(image)
        acc = np.zeros(gray.shape, dtype=np.float32)
        for scale in self.config.morph_scales:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(scale), int(scale)))
            if mode in ("both", "black"):
                acc += cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
            if mode in ("both", "top"):
                acc += cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
        return self._to_uint8(acc)

    def enhance_vessels(self, image: np.ndarray) -> np.ndarray:
        """Pipeline combinado padrão: Frangi (0.6) + black-hat (0.4)."""
        frangi_map = self.enhance_frangi(image).astype(np.float32) / 255.0
        morph_map = self.enhance_blackhat(image).astype(np.float32) / 255.0
        combined = 0.6 * frangi_map + 0.4 * morph_map
        return self._to_uint8(combined)

    def enhance(self, image: np.ndarray, method: PhotometricTransform) -> np.ndarray:
        """Despacha para o algoritmo de realce correspondente."""
        dispatch = {
            PhotometricTransform.FRANGI: self.enhance_frangi,
            PhotometricTransform.GABOR: self.enhance_gabor,
            PhotometricTransform.BLACKHAT: self.enhance_blackhat,
            PhotometricTransform.VESSEL: self.enhance_vessels,
        }
        if method not in dispatch:
            raise ValueError(f"Método de realce não suportado: {method}")
        return dispatch[method](image)

    # ------------------------------ métricas --------------------------------
    @staticmethod
    def quantify(enhanced: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Métricas objetivas de visibilidade da vascularização (reportadas como
        medidas, sem juízo de valor):

        - structure_contrast: CNR (contrast-to-noise ratio) entre as estruturas
          realçadas e o fundo, (média_estrutura - média_fundo) / desvio_fundo.
          É a medida direta de "quão bem as fibras/vasos se separam do fundo",
          o que de fato significa "mostrar melhor a vascularização".
        - ridge_energy: energia de cristas/bordas (variância do Laplaciano),
          normalizada; quanto das estruturas finas foi evidenciado.
        - structure_density: fração de pixels acima do limiar de Otsu.
        - rms_contrast: desvio-padrão das intensidades normalizado em [0,1].
        - entropy: entropia de Shannon (riqueza de informação).
        """
        img = enhanced if enhanced.dtype == np.uint8 else VascularizationEnhancer._to_uint8(enhanced)
        zero = {"structure_contrast": 0.0, "ridge_energy": 0.0,
                "structure_density": 0.0, "rms_contrast": 0.0, "entropy": 0.0}
        if mask is not None:
            sel = mask > 0
            values = img[sel]
        else:
            sel = np.ones(img.shape, dtype=bool)
            values = img.ravel()
        if values.size == 0:
            return dict(zero)

        otsu_thr, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        density = float(np.mean(values > otsu_thr))
        rms_contrast = float(np.std(values) / 255.0)
        entropy = float(shannon_entropy(values))

        # CNR estrutura-vs-fundo (dentro da máscara)
        fg = img[sel & (img > otsu_thr)]
        bg = img[sel & (img <= otsu_thr)]
        if fg.size > 0 and bg.size > 0:
            structure_contrast = float((float(fg.mean()) - float(bg.mean()))
                                       / (float(np.std(bg)) + 1e-6))
        else:
            structure_contrast = 0.0

        # Energia de cristas (variância do Laplaciano) normalizada, na máscara
        lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
        ridge_energy = float(np.var(lap[sel]) / (255.0 ** 2))

        return {
            "structure_contrast": structure_contrast,
            "ridge_energy": ridge_energy,
            "structure_density": density,
            "rms_contrast": rms_contrast,
            "entropy": entropy,
        }


class ImagePreprocessor:
    """
    Classe para pré-processamento de imagens de íris.
    
    Implementa transformações fotométricas padronizadas para teste de robustez
    do pipeline conforme descrito na metodologia do artigo.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Inicializa o pré-processador.
        
        Args:
            config: Configuração de pré-processamento. Se None, usa padrão.
        """
        self.config = config or PreprocessingConfig()
        
        # Inicializa CLAHE
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size
        )

        # Realce de vascularização (Frangi / Gabor / black-hat)
        self._vessel_enhancer = VascularizationEnhancer(self.config)
    
    def extract_channel(self, image: np.ndarray, 
                        channel: Optional[ColorChannel] = None) -> np.ndarray:
        """
        Extrai um canal específico da imagem.
        
        Args:
            image: Imagem BGR (OpenCV format)
            channel: Canal a extrair. Se None, usa o configurado.
            
        Returns:
            Imagem em escala de cinza do canal selecionado.
        """
        channel = channel or self.config.color_channel
        
        if channel == ColorChannel.GRAY:
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
        
        elif channel in [ColorChannel.H, ColorChannel.S, ColorChannel.V]:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channel_idx = {'h': 0, 's': 1, 'v': 2}[channel.value]
            return hsv[:, :, channel_idx]
        
        elif channel in [ColorChannel.B, ColorChannel.G, ColorChannel.R]:
            channel_idx = {'b': 0, 'g': 1, 'r': 2}[channel.value]
            return image[:, :, channel_idx]
        
        else:
            raise ValueError(f"Canal não suportado: {channel}")
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica equalização global de histograma.
        
        Esta transformação redistribui os valores de intensidade para
        ocupar todo o range disponível, aumentando o contraste global.
        
        Args:
            image: Imagem em escala de cinza
            
        Returns:
            Imagem equalizada
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(image)
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        CLAHE divide a imagem em tiles e aplica equalização adaptativa
        em cada um, com limite de contraste para evitar amplificação
        excessiva de ruído.
        
        Args:
            image: Imagem em escala de cinza
            
        Returns:
            Imagem com CLAHE aplicado
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self._clahe.apply(image)
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica Gaussian blur (suavização gaussiana).
        
        Útil para reduzir ruído de alta frequência e avaliar
        se o modelo depende de detalhes finos ou padrões mais grosseiros.
        
        Args:
            image: Imagem (pode ser colorida ou escala de cinza)
            
        Returns:
            Imagem suavizada
        """
        return cv2.GaussianBlur(
            image, 
            self.config.blur_kernel_size, 
            self.config.blur_sigma
        )
    
    def apply_transform(self, image: np.ndarray, 
                        transform: PhotometricTransform) -> np.ndarray:
        """
        Aplica uma transformação fotométrica específica.
        
        Args:
            image: Imagem de entrada
            transform: Tipo de transformação a aplicar
            
        Returns:
            Imagem transformada
        """
        if transform == PhotometricTransform.ORIGINAL:
            return image.copy()
        
        elif transform == PhotometricTransform.HISTOGRAM:
            return self.apply_histogram_equalization(image)
        
        elif transform == PhotometricTransform.CLAHE:
            return self.apply_clahe(image)
        
        elif transform == PhotometricTransform.BLUR:
            return self.apply_gaussian_blur(image)

        elif transform in (PhotometricTransform.FRANGI,
                           PhotometricTransform.GABOR,
                           PhotometricTransform.BLACKHAT,
                           PhotometricTransform.VESSEL):
            return self._vessel_enhancer.enhance(image, transform)

        else:
            raise ValueError(f"Transformação não suportada: {transform}")
    
    def preprocess(self, image: np.ndarray, 
                   channel: Optional[ColorChannel] = None,
                   transform: PhotometricTransform = PhotometricTransform.ORIGINAL
                   ) -> np.ndarray:
        """
        Pipeline completo de pré-processamento.
        
        Args:
            image: Imagem BGR de entrada
            channel: Canal de cor a extrair
            transform: Transformação fotométrica a aplicar
            
        Returns:
            Imagem pré-processada em escala de cinza
        """
        # Extrai canal
        gray = self.extract_channel(image, channel)
        
        # Aplica transformação fotométrica
        processed = self.apply_transform(gray, transform)
        
        return processed
    
    def preprocess_batch(self, images: np.ndarray,
                         channel: Optional[ColorChannel] = None,
                         transform: PhotometricTransform = PhotometricTransform.ORIGINAL
                         ) -> np.ndarray:
        """
        Pré-processa um lote de imagens.
        
        Args:
            images: Array de imagens (N, H, W, C) ou (N, H, W)
            channel: Canal de cor a extrair
            transform: Transformação fotométrica a aplicar
            
        Returns:
            Array de imagens pré-processadas (N, H, W)
        """
        n_images = images.shape[0]
        
        # Processa primeira imagem para determinar shape
        first_processed = self.preprocess(
            np.squeeze(images[0]), channel, transform
        )
        
        # Cria array de saída
        output = np.zeros((n_images,) + first_processed.shape, dtype=np.uint8)
        output[0] = first_processed
        
        # Processa restante
        for i in range(1, n_images):
            output[i] = self.preprocess(
                np.squeeze(images[i]), channel, transform
            )
        
        return output


class IrisNormalizer:
    """
    Normalização geométrica da íris (rubber sheet transform).
    
    Mapeia a região anular da íris para uma representação retangular
    em coordenadas polares, permitindo comparação ponto-a-ponto
    mesmo com variações de dilatação pupilar.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Inicializa o normalizador.
        
        Args:
            config: Configuração de pré-processamento
        """
        self.config = config or PreprocessingConfig()
        self.height = self.config.normalized_height
        self.width = self.config.normalized_width
    
    def normalize(self, image: np.ndarray,
                  pupil_center: Tuple[int, int],
                  pupil_contour: np.ndarray,
                  iris_contour: np.ndarray) -> np.ndarray:
        """
        Aplica normalização rubber sheet.
        
        Mapeia a região anular entre pupila e íris para uma grade retangular.
        
        Args:
            image: Imagem de entrada (BGR ou grayscale)
            pupil_center: Centro da pupila (x, y)
            pupil_contour: Contorno da pupila
            iris_contour: Contorno da íris
            
        Returns:
            Imagem normalizada (height x width)
        """
        # Garante que a imagem tem 3 canais
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Cria imagem de saída
        normalized = np.zeros(
            (self.height, self.width, image.shape[2]), 
            dtype=np.uint8
        )
        
        # Varre todos os ângulos
        for col, theta_deg in enumerate(np.linspace(0, 360, self.width, endpoint=False)):
            theta_rad = np.deg2rad(theta_deg)
            cos_t = np.cos(theta_rad)
            sin_t = np.sin(theta_rad)
            
            # Encontra ponto na borda da pupila
            px, py = float(pupil_center[0]), float(pupil_center[1])
            while cv2.pointPolygonTest(pupil_contour, (int(px), int(py)), False) == 1:
                px += cos_t
                py += sin_t
            pupil_edge = (int(px - cos_t), int(py - sin_t))
            
            # Encontra ponto na borda da íris
            ix, iy = float(pupil_center[0]), float(pupil_center[1])
            while cv2.pointPolygonTest(iris_contour, (int(ix), int(iy)), False) == 1:
                ix += cos_t
                iy += sin_t
            iris_edge = (int(ix - cos_t), int(iy - sin_t))
            
            # Interpola entre pupila e íris
            for row, r in enumerate(np.linspace(0, 1, self.height)):
                x = int((1 - r) * pupil_edge[0] + r * iris_edge[0])
                y = int((1 - r) * pupil_edge[1] + r * iris_edge[1])
                
                # Verifica limites
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    normalized[row, col] = image[y, x]
        
        return normalized
    
    def crop_upper_region(self, normalized_image: np.ndarray) -> np.ndarray:
        """
        Remove região superior (pálpebra) da imagem normalizada.
        
        Args:
            normalized_image: Imagem normalizada
            
        Returns:
            Imagem com região superior removida
        """
        return normalized_image[self.config.upper_cut_height:, :]


def create_preprocessor(config: Optional[PreprocessingConfig] = None) -> ImagePreprocessor:
    """Factory function para criar pré-processador."""
    return ImagePreprocessor(config)


def create_vascularization_enhancer(
        config: Optional[PreprocessingConfig] = None) -> VascularizationEnhancer:
    """Factory function para criar o realçador de vascularização."""
    return VascularizationEnhancer(config)


def create_normalizer(config: Optional[PreprocessingConfig] = None) -> IrisNormalizer:
    """Factory function para criar normalizador."""
    return IrisNormalizer(config)


# =============================================================================
# Funções utilitárias
# =============================================================================

def apply_all_transforms(image: np.ndarray, 
                         config: Optional[PreprocessingConfig] = None
                         ) -> dict:
    """
    Aplica todas as transformações fotométricas a uma imagem.
    
    Útil para análise comparativa de robustez.
    
    Args:
        image: Imagem de entrada
        config: Configuração de pré-processamento
        
    Returns:
        Dicionário com imagens transformadas
    """
    preprocessor = ImagePreprocessor(config)
    
    results = {}
    for transform in PhotometricTransform:
        results[transform.value] = preprocessor.apply_transform(image, transform)
    
    return results


def normalize_intensity(image: np.ndarray, 
                        target_mean: float = 128.0,
                        target_std: float = 50.0) -> np.ndarray:
    """
    Normaliza intensidade da imagem para média e desvio padrão alvo.
    
    Args:
        image: Imagem de entrada
        target_mean: Média alvo
        target_std: Desvio padrão alvo
        
    Returns:
        Imagem normalizada
    """
    current_mean = np.mean(image)
    current_std = np.std(image)
    
    if current_std == 0:
        return np.full_like(image, target_mean, dtype=np.uint8)
    
    normalized = (image - current_mean) / current_std * target_std + target_mean
    return np.clip(normalized, 0, 255).astype(np.uint8)

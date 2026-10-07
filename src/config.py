"""
Configuração do Pipeline de Análise de Iridologia
=================================================
Módulo de configuração central com todos os parâmetros do experimento.
Baseado na metodologia descrita no artigo de pesquisa.

Referência:
    "Iridologia sob Lentes Científicas: Avaliação Crítica com Aprendizado de Máquina"
    UFABC - Programa de Iniciação Científica
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

# =============================================================================
# Enumerações
# =============================================================================

class DatasetType(Enum):
    """
    Esquemas de particionamento para controle de data leakage.
    
    - ALL: utiliza todos os dados sem separação específica
    - L_SPLIT: utiliza apenas olhos esquerdos
    - R_SPLIT: utiliza apenas olhos direitos
    - PERSON_BASE: separa por identidade (controle principal contra vazamento)
    - PERSON_BASE_INVERT: separação por identidade invertida
    """
    ALL = "all"
    L_SPLIT = "L_split"
    R_SPLIT = "R_split"
    PERSON_BASE = "personBase"
    PERSON_BASE_INVERT = "personBase_invert"


class ClassifierType(Enum):
    """
    Classificadores avaliados no estudo.
    
    Conforme metodologia do artigo:
    - LR: Regressão Logística
    - SVM: Support Vector Machine
    - RF: Random Forest
    - MLP: Multi-Layer Perceptron
    - ADABOOST: AdaBoost
    """
    LR = "LogisticRegression"
    SVM = "SVM"
    RF = "RandomForest"
    MLP = "MLP"
    ADABOOST = "AdaBoost"


class PhotometricTransform(Enum):
    """
    Transformações fotométricas e realces de vascularização.

    Robustez (baseline):
    - ORIGINAL: sem transformação
    - HISTOGRAM: equalização global de histograma
    - CLAHE: Contrast Limited Adaptive Histogram Equalization
    - BLUR: Gaussian blur

    Realce de vascularização/estroma (algoritmos consolidados em imagem médica):
    - FRANGI: filtro de vascularização de Frangi (vesselness Hessiano multiescala)
    - GABOR: resposta máxima de um banco de filtros de Gabor orientados
    - BLACKHAT: top-hat/black-hat morfológico multiescala
    - VESSEL: combinação ponderada padrão (0.6 Frangi + 0.4 black-hat)
    """
    ORIGINAL = "original"
    HISTOGRAM = "histogram"
    CLAHE = "clahe"
    BLUR = "blur"
    FRANGI = "frangi"
    GABOR = "gabor"
    BLACKHAT = "blackhat"
    VESSEL = "vessel"


class FeatureType(Enum):
    """
    Tipos de features extraídas das imagens.
    
    - PIXEL: vetor de intensidades (baseline)
    - LBP: Local Binary Patterns
    - GLCM: Grey-Level Co-occurrence Matrix (Haralick)
    - INTENSITY_STATS: estatísticas de intensidade
    - GABOR: filtros de Gabor
    - HOG: Histogram of Oriented Gradients
    """
    PIXEL = "pixel"
    LBP = "lbp"
    GLCM = "glcm"
    INTENSITY_STATS = "intensity_stats"
    GABOR = "gabor"
    HOG = "hog"


class ColorChannel(Enum):
    """Canais de cor suportados."""
    GRAY = "gray"
    H = "h"  # Hue (HSV)
    S = "s"  # Saturation (HSV)
    V = "v"  # Value (HSV)
    R = "r"  # Red (BGR)
    G = "g"  # Green (BGR)
    B = "b"  # Blue (BGR)


# =============================================================================
# Configurações de Segmentação
# =============================================================================

@dataclass
class SegmentationConfig:
    """Configuração da segmentação de íris e pupila."""
    
    # Tamanho alvo para processamento
    target_size: Tuple[int, int] = (853, 1280)  # (height, width)
    
    # Níveis de downsampling para processamento
    downsample_levels: int = 2
    
    # Kernel morfológico
    morph_kernel_size: Tuple[int, int] = (7, 7)
    
    # Parâmetros de detecção de pupila
    pupil_threshold_init: int = 40
    pupil_init_radius: int = 100
    min_pupil_area: int = 2500
    min_pupil_radius: int = 10
    
    # Parâmetros de active contour (snake)
    snake_alpha: float = 0.2   # Elasticidade
    snake_beta: float = 100.0  # Rigidez
    
    # Parâmetros específicos para íris
    iris_snake_alpha: float = -0.1  # Negativo para expansão
    iris_snake_gamma: float = 0.001
    iris_convergence: float = 0.35


# =============================================================================
# Configuração da Validação de Pupila
# =============================================================================

@dataclass
class PupilValidationConfig:
    """
    Limiares para validação/classificação da geometria pupila-íris.

    Base fisiológica (olho humano adulto, iluminação normal):
    - diâmetro da pupila ~ 2-8 mm, diâmetro da íris ~ 11-12 mm
    - razão pupila/íris ~ 0.2-0.7 fora de dilatação/constrição extrema
    - espessura relativa da íris = (r_iris - r_pupila) / r_iris
    - excentricidade (offset dos centros) normalizada pelo raio da íris
    """
    # Razão diâmetro pupila/íris aceitável
    min_pupil_iris_ratio: float = 0.20
    max_pupil_iris_ratio: float = 0.70
    # Faixa "borderline" (válida com ressalva)
    borderline_low_ratio: float = 0.15
    borderline_high_ratio: float = 0.80
    # Espessura relativa mínima da íris (anel)
    min_thickness_ratio: float = 0.25
    # Excentricidade máxima (offset dos centros / raio da íris)
    max_concentricity_offset: float = 0.30
    # Escala opcional para converter pixels em milímetros (px/mm); None = desligado
    pixels_per_mm: Optional[float] = None


# =============================================================================
# Configurações de Normalização (Rubber Sheet)
# =============================================================================

@dataclass
class NormalizationConfig:
    """Configuração da normalização de íris (rubber sheet model)."""
    
    # Dimensões da imagem normalizada
    normalized_height: int = 201  # Resolução radial
    normalized_width: int = 720   # Resolução angular (720px = 0.5°/pixel)
    
    # Método de interpolação
    interpolation_method: str = "bilinear"


# =============================================================================
# Configurações de Dados
# =============================================================================

@dataclass
class DataConfig:
    """Configuração dos dados e caminhos."""
    
    # Caminhos
    data_root: str = "Data/Data"
    raw_data_root: str = "Data/Raw"  # Para imagens não processadas
    output_dir: str = "results"
    
    # Datasets disponíveis
    datasets: List[DatasetType] = field(default_factory=lambda: [
        DatasetType.ALL,
        DatasetType.L_SPLIT,
        DatasetType.R_SPLIT,
        DatasetType.PERSON_BASE,
        DatasetType.PERSON_BASE_INVERT
    ])
    
    # Arquivos de dados
    control_file: str = "controlImageArr.p"
    diabetic_file: str = "diabeteImageArr.p"
    dr_type_file: str = "DR_type.p"
    control_manner_file: str = "controllingManner.p"
    
    def get_dataset_path(self, dataset: DatasetType) -> str:
        """Retorna o caminho completo para um dataset."""
        return os.path.join(self.data_root, dataset.value)


# =============================================================================
# Configurações de Pré-processamento
# =============================================================================

@dataclass
class PreprocessingConfig:
    """Configuração do pré-processamento de imagens."""
    
    # Canal de cor padrão
    color_channel: ColorChannel = ColorChannel.GRAY
    
    # Transformações fotométricas
    transforms: List[PhotometricTransform] = field(default_factory=lambda: [
        PhotometricTransform.ORIGINAL,
        PhotometricTransform.HISTOGRAM,
        PhotometricTransform.CLAHE,
        PhotometricTransform.BLUR
    ])
    
    # Parâmetros CLAHE
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    
    # Parâmetros Gaussian Blur
    blur_kernel_size: Tuple[int, int] = (5, 5)
    blur_sigma: float = 0

    # --- Realce de vascularização (algoritmos consolidados) ---
    # Frangi (vesselness Hessiano multiescala)
    frangi_sigmas: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    frangi_beta: float = 0.5        # sensibilidade à "blobness" (plate vs line)
    frangi_gamma: float = 15.0      # sensibilidade ao contraste de estrutura
    frangi_black_ridges: bool = False  # False: realça fibras/cristas claras
    # Banco de Gabor (fibras radiais/orientadas)
    gabor_n_orientations: int = 8
    gabor_ksize: int = 21
    gabor_sigma: float = 4.0
    gabor_lambda: float = 10.0
    gabor_gamma: float = 0.5
    # Top-hat / black-hat morfológico multiescala
    morph_scales: Tuple[int, ...] = (5, 11, 21)

    # Normalização da íris (rubber sheet)
    normalized_height: int = 201
    normalized_width: int = 720
    
    # Região superior a ser cortada (pálpebra)
    upper_cut_height: int = 10


# =============================================================================
# Configurações de Extração de Features
# =============================================================================

@dataclass
class FeatureConfig:
    """Configuração da extração de features."""
    
    # Features a extrair
    features: List[FeatureType] = field(default_factory=lambda: [
        FeatureType.PIXEL,
        FeatureType.LBP,
        FeatureType.GLCM,
        FeatureType.INTENSITY_STATS
    ])
    
    # Parâmetros Pixel Features
    pixel_downsample_size: int = 4
    
    # Parâmetros LBP
    lbp_n_points: int = 16
    lbp_radius: int = 2
    lbp_method: str = "uniform"
    
    # Parâmetros GLCM
    glcm_distances: List[int] = field(default_factory=lambda: [1, 2, 3])
    glcm_angles: List[float] = field(default_factory=lambda: [0, 0.785, 1.571, 2.356])  # 0, 45, 90, 135 graus
    glcm_levels: int = 256
    glcm_properties: List[str] = field(default_factory=lambda: [
        'contrast', 'dissimilarity', 'homogeneity', 
        'energy', 'correlation', 'ASM'
    ])
    
    # Parâmetros Gabor
    gabor_frequencies: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    gabor_n_orientations: int = 8
    gabor_sigma: float = 1.0
    
    # Parâmetros HOG
    hog_orientations: int = 6
    hog_pixels_per_cell: Tuple[int, int] = (32, 32)
    hog_cells_per_block: Tuple[int, int] = (1, 1)
    
    # Parâmetros estatísticas de intensidade
    histogram_bins: int = 32


# =============================================================================
# Configurações de Classificação
# =============================================================================

@dataclass
class ClassifierConfig:
    """Configuração dos classificadores."""
    
    # Classificadores a avaliar
    classifiers: List[ClassifierType] = field(default_factory=lambda: [
        ClassifierType.LR,
        ClassifierType.SVM,
        ClassifierType.RF,
        ClassifierType.MLP,
        ClassifierType.ADABOOST
    ])
    
    # Validação cruzada
    n_folds: int = 5  # K=5 folds conforme metodologia
    stratified: bool = True  # Validação cruzada estratificada
    
    # Random state para reprodutibilidade
    random_state: int = 42
    
    # Parâmetros Regressão Logística
    lr_C: float = 1.0
    lr_max_iter: int = 1000
    lr_solver: str = "lbfgs"
    
    # Parâmetros SVM
    svm_C: float = 1.0
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"
    
    # Parâmetros Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    
    # Parâmetros MLP
    mlp_hidden_layers: Tuple[int, ...] = (100, 50)
    mlp_max_iter: int = 2000
    mlp_activation: str = "relu"
    mlp_solver: str = "adam"
    
    # Parâmetros AdaBoost
    adaboost_n_estimators: int = 50
    adaboost_algorithm: str = "SAMME.R"


# =============================================================================
# Configurações de Análise Local
# =============================================================================

@dataclass
class LocalAnalysisConfig:
    """Configuração das análises locais por região."""
    
    # Análise por setores angulares
    sector_step_degrees: int = 10  # Passo de 10° conforme metodologia
    sector_width_degrees: int = 50  # Largura da janela
    
    # Análise de região pancreática (malha 12x12)
    pancreatic_grid_size: Tuple[int, int] = (12, 12)
    
    # Regiões predefinidas para análise (ângulos em pixels para íris de 720px)
    # Conversão: 1° = 2 pixels (720/360)
    predefined_regions: Dict[str, List[Tuple[int, int]]] = field(default_factory=lambda: {
        "cross_andreas": [(395, 445), (35, 85), (275, 325), (635, 685)],
        "pancreas_left": [(430, 470)],   # ~215-235° para olho esquerdo
        "pancreas_right": [(250, 290)],  # ~125-145° para olho direito
        "our_regions": [(75, 125), (375, 425), (235, 285), (615, 665)]
    })


# =============================================================================
# Configurações de Métricas
# =============================================================================

@dataclass
class MetricsConfig:
    """Configuração das métricas de avaliação."""
    
    # Métricas a calcular
    metrics: List[str] = field(default_factory=lambda: [
        'accuracy',
        'sensitivity',  # recall da classe DM2
        'specificity',
        'precision',
        'f1_score'
    ])
    
    # Matriz de confusão
    compute_confusion_matrix: bool = True
    
    # Intervalo de confiança
    compute_confidence_interval: bool = True
    confidence_level: float = 0.95


# =============================================================================
# Configuração Global
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuração completa do experimento."""
    
    # Nome do experimento
    experiment_name: str = "iridology_dm2_analysis"
    
    # Descrição
    description: str = "Avaliação crítica da iridologia para DM2 com ML"
    
    # Componentes
    data: DataConfig = field(default_factory=DataConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    local_analysis: LocalAnalysisConfig = field(default_factory=LocalAnalysisConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    
    # Opções de execução
    verbose: bool = True
    save_results: bool = True
    generate_plots: bool = True
    
    def to_dict(self) -> dict:
        """Converte configuração para dicionário para logging."""
        import dataclasses
        return dataclasses.asdict(self)


# =============================================================================
# Instância padrão
# =============================================================================

def get_default_config() -> ExperimentConfig:
    """Retorna a configuração padrão do experimento."""
    return ExperimentConfig()


def get_minimal_config() -> ExperimentConfig:
    """Retorna configuração mínima para testes rápidos."""
    config = ExperimentConfig()
    config.data.datasets = [DatasetType.PERSON_BASE]
    config.classifier.classifiers = [ClassifierType.LR, ClassifierType.MLP]
    config.preprocessing.transforms = [PhotometricTransform.ORIGINAL]
    config.features.features = [FeatureType.PIXEL]
    return config


def get_full_config() -> ExperimentConfig:
    """Retorna configuração completa para análise final."""
    return ExperimentConfig()


# Alias para compatibilidade
DEFAULT_CONFIG = get_default_config()

"""
Iridology Analysis Package
==========================
Módulos para análise de iridologia e detecção de DM2.

Referência:
    "Iridologia sob Lentes Científicas: Avaliação Crítica com Aprendizado de Máquina"
"""

from .config import (
    ExperimentConfig,
    DatasetType,
    ClassifierType,
    PhotometricTransform,
    FeatureType,
    ColorChannel,
    SegmentationConfig,
    NormalizationConfig,
    PupilValidationConfig,
    get_default_config
)

from .segmentation import (
    IrisSegmenter,
    SegmentationResult,
    PupilValidator,
    PupilValidationResult,
    create_segmenter,
    create_pupil_validator
)

from .normalization import (
    IrisNormalizer,
    NormalizationResult,
    IrisPreprocessingPipeline,
    create_normalizer,
    create_preprocessing_pipeline
)

from .preprocessing import (
    ImagePreprocessor,
    VascularizationEnhancer,
    create_preprocessor,
    create_vascularization_enhancer
)

from .feature_extraction import (
    FeatureExtractor,
    create_feature_extractor
)

from .classifiers import (
    ClassifierFactory,
    CrossValidator,
    ClassificationResult,
    create_classifier,
    create_validator
)

from .metrics import (
    MetricsCalculator,
    MetricsResult,
    create_calculator,
    format_metrics_table,
    format_confusion_matrix
)

from .local_analysis import (
    SectorAnalyzer,
    PancreaticRegionAnalyzer,
    LocalAnalysisPipeline,
    SectorAnalysisResult,
    GridAnalysisResult
)

from .results_generator import (
    ResultsGenerator,
    create_results_generator
)

__version__ = "2.0.0"
__author__ = "UFABC Research Team"

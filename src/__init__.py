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
    create_pupil_validator,
    detect_eye_circles,
    refine_circle,
    ransac_circle
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

from .legacy_data import (
    LegacyDataset,
    load_legacy_layout,
    resolve_legacy_source,
    restricted_numpy_load
)

from .fusion import (
    GEOMETRY_FEATURES,
    build_fusion_classifier,
    evaluate_geometry_fusion
)

from .synthetic_eye import (
    EyeGeometry,
    rubber_sheet_coordinates,
    sample_synthetic_eye,
    simulate_pupil_dilation
)

__version__ = "2.0.0"
__author__ = "UFABC Research Team"

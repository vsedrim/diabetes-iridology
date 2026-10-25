"""
Módulo de Classificação
=======================
Implementa classificadores e validação cruzada estratificada.

Classificadores conforme metodologia do artigo:
- LR: Regressão Logística
- SVM: Support Vector Machine
- RF: Random Forest
- MLP: Multi-Layer Perceptron
- AdaBoost

Referência:
    "Iridologia sob Lentes Científicas: Avaliação Crítica com Aprendizado de Máquina"
"""

import inspect
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    StratifiedKFold, 
    cross_val_score, 
    cross_val_predict,
    cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

from .config import ClassifierConfig, ClassifierType


@dataclass
class ClassificationResult:
    """
    Resultado de classificação com métricas detalhadas.
    
    Armazena resultados por fold e métricas agregadas.
    """
    classifier_name: str
    dataset_name: str
    
    # Métricas por fold
    accuracy_per_fold: List[float] = field(default_factory=list)
    sensitivity_per_fold: List[float] = field(default_factory=list)
    specificity_per_fold: List[float] = field(default_factory=list)
    precision_per_fold: List[float] = field(default_factory=list)
    f1_per_fold: List[float] = field(default_factory=list)
    
    # Predições para matriz de confusão
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    
    # Probabilidades (se disponível)
    y_proba: Optional[np.ndarray] = None
    
    @property
    def accuracy_mean(self) -> float:
        return np.mean(self.accuracy_per_fold) if self.accuracy_per_fold else 0.0
    
    @property
    def accuracy_std(self) -> float:
        return np.std(self.accuracy_per_fold) if self.accuracy_per_fold else 0.0
    
    @property
    def sensitivity_mean(self) -> float:
        return np.mean(self.sensitivity_per_fold) if self.sensitivity_per_fold else 0.0
    
    @property
    def specificity_mean(self) -> float:
        return np.mean(self.specificity_per_fold) if self.specificity_per_fold else 0.0
    
    @property
    def precision_mean(self) -> float:
        return np.mean(self.precision_per_fold) if self.precision_per_fold else 0.0
    
    @property
    def f1_mean(self) -> float:
        return np.mean(self.f1_per_fold) if self.f1_per_fold else 0.0
    
    def to_dict(self) -> dict:
        """Converte resultado para dicionário."""
        return {
            'classifier': self.classifier_name,
            'dataset': self.dataset_name,
            'accuracy_mean': self.accuracy_mean,
            'accuracy_std': self.accuracy_std,
            'sensitivity_mean': self.sensitivity_mean,
            'specificity_mean': self.specificity_mean,
            'precision_mean': self.precision_mean,
            'f1_mean': self.f1_mean,
            'n_folds': len(self.accuracy_per_fold)
        }


class ClassifierFactory:
    """
    Factory para criação de classificadores configurados.
    
    Centraliza a criação de classificadores com parâmetros padronizados
    conforme a metodologia do artigo.
    """
    
    def __init__(self, config: Optional[ClassifierConfig] = None):
        """
        Inicializa a factory.
        
        Args:
            config: Configuração dos classificadores
        """
        self.config = config or ClassifierConfig()
    
    def create_classifier(self, classifier_type: ClassifierType) -> BaseEstimator:
        """
        Cria um classificador configurado.
        
        Args:
            classifier_type: Tipo de classificador a criar
            
        Returns:
            Instância do classificador
        """
        if classifier_type == ClassifierType.LR:
            return LogisticRegression(
                C=self.config.lr_C,
                max_iter=self.config.lr_max_iter,
                solver=self.config.lr_solver,
                random_state=self.config.random_state
            )
        
        elif classifier_type == ClassifierType.SVM:
            return SVC(
                C=self.config.svm_C,
                kernel=self.config.svm_kernel,
                gamma=self.config.svm_gamma,
                random_state=self.config.random_state,
                probability=True
            )
        
        elif classifier_type == ClassifierType.RF:
            return RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                random_state=self.config.random_state,
                n_jobs=1
            )
        
        elif classifier_type == ClassifierType.MLP:
            return MLPClassifier(
                hidden_layer_sizes=self.config.mlp_hidden_layers,
                max_iter=self.config.mlp_max_iter,
                activation=self.config.mlp_activation,
                solver=self.config.mlp_solver,
                random_state=self.config.random_state
            )
        
        elif classifier_type == ClassifierType.ADABOOST:
            params = {
                "n_estimators": self.config.adaboost_n_estimators,
                "random_state": self.config.random_state,
            }
            # O scikit-learn 1.8 removeu o parâmetro "algorithm" (só resta o SAMME);
            # ele só é repassado às versões que ainda o aceitam.
            if "algorithm" in inspect.signature(AdaBoostClassifier.__init__).parameters:
                params["algorithm"] = self.config.adaboost_algorithm
            return AdaBoostClassifier(**params)
        
        else:
            raise ValueError(f"Tipo de classificador não suportado: {classifier_type}")
    
    def create_pipeline(self, classifier_type: ClassifierType,
                        scale: bool = True) -> Pipeline:
        """
        Cria um pipeline com pré-processamento e classificador.
        
        Args:
            classifier_type: Tipo de classificador
            scale: Se True, inclui normalização StandardScaler
            
        Returns:
            Pipeline sklearn
        """
        steps = []
        
        if scale:
            steps.append(('scaler', StandardScaler()))
        
        steps.append(('classifier', self.create_classifier(classifier_type)))
        
        return Pipeline(steps)
    
    def create_all_classifiers(self, scale: bool = True) -> Dict[str, Pipeline]:
        """
        Cria todos os classificadores configurados.
        
        Args:
            scale: Se True, inclui normalização
            
        Returns:
            Dicionário de pipelines por nome
        """
        classifiers = {}
        for clf_type in self.config.classifiers:
            classifiers[clf_type.value] = self.create_pipeline(clf_type, scale)
        return classifiers


class CrossValidator:
    """
    Validador cruzado estratificado para classificação.
    
    Implementa validação cruzada K-fold estratificada conforme
    metodologia do artigo (K=5 por padrão).
    """
    
    def __init__(self, config: Optional[ClassifierConfig] = None):
        """
        Inicializa o validador.
        
        Args:
            config: Configuração de classificação
        """
        self.config = config or ClassifierConfig()
        self.factory = ClassifierFactory(config)
    
    def _create_cv_splitter(self) -> StratifiedKFold:
        """Cria splitter de validação cruzada estratificada."""
        return StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
    
    def validate(self, X: np.ndarray, y: np.ndarray,
                 classifier_type: ClassifierType,
                 dataset_name: str = "unknown",
                 scale: bool = True) -> ClassificationResult:
        """
        Executa validação cruzada para um classificador.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            classifier_type: Tipo de classificador
            dataset_name: Nome do dataset para logging
            scale: Se True, normaliza features
            
        Returns:
            ClassificationResult com métricas detalhadas
        """
        result = ClassificationResult(
            classifier_name=classifier_type.value,
            dataset_name=dataset_name
        )
        
        # Cria pipeline e splitter
        pipeline = self.factory.create_pipeline(classifier_type, scale)
        cv = self._create_cv_splitter()
        
        # Validação cruzada com múltiplas métricas
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',  # sensibilidade
            'f1': 'f1'
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            cv_results = cross_validate(
                pipeline, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=1
            )
        
        # Armazena resultados por fold
        result.accuracy_per_fold = cv_results['test_accuracy'].tolist()
        result.precision_per_fold = cv_results['test_precision'].tolist()
        result.sensitivity_per_fold = cv_results['test_recall'].tolist()
        result.f1_per_fold = cv_results['test_f1'].tolist()
        
        # Calcula especificidade manualmente por fold
        y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=1)
        result.y_true = y
        result.y_pred = y_pred
        
        # Especificidade por fold
        for train_idx, test_idx in cv.split(X, y):
            y_test_fold = y[test_idx]
            y_pred_fold = y_pred[test_idx]
            
            # TN / (TN + FP)
            tn = np.sum((y_test_fold == 0) & (y_pred_fold == 0))
            fp = np.sum((y_test_fold == 0) & (y_pred_fold == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            result.specificity_per_fold.append(specificity)
        
        return result
    
    def validate_all(self, X: np.ndarray, y: np.ndarray,
                     dataset_name: str = "unknown",
                     scale: bool = True) -> Dict[str, ClassificationResult]:
        """
        Executa validação cruzada para todos os classificadores.
        
        Args:
            X: Features
            y: Labels
            dataset_name: Nome do dataset
            scale: Se True, normaliza features
            
        Returns:
            Dicionário de resultados por classificador
        """
        results = {}
        
        for clf_type in self.config.classifiers:
            print(f"  Validando {clf_type.value}...")
            results[clf_type.value] = self.validate(
                X, y, clf_type, dataset_name, scale
            )
        
        return results


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Classificador ensemble por votação.
    
    Combina múltiplos classificadores usando votação majoritária
    (hard voting) ou probabilística (soft voting).
    """
    
    def __init__(self, classifiers: Dict[str, BaseEstimator],
                 voting: str = 'hard'):
        """
        Inicializa o ensemble.
        
        Args:
            classifiers: Dicionário de classificadores
            voting: 'hard' para votação majoritária, 'soft' para probabilística
        """
        self.classifiers = classifiers
        self.voting = voting
        self.fitted_classifiers_ = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleClassifier':
        """
        Treina todos os classificadores.
        
        Args:
            X: Features de treino
            y: Labels de treino
            
        Returns:
            self
        """
        for name, clf in self.classifiers.items():
            self.fitted_classifiers_[name] = clf.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz usando votação.
        
        Args:
            X: Features de teste
            
        Returns:
            Predições
        """
        if self.voting == 'hard':
            predictions = np.array([
                clf.predict(X) for clf in self.fitted_classifiers_.values()
            ])
            # Votação majoritária
            return np.round(np.mean(predictions, axis=0)).astype(int)
        
        else:  # soft voting
            probas = np.array([
                clf.predict_proba(X)[:, 1] 
                for clf in self.fitted_classifiers_.values()
            ])
            return (np.mean(probas, axis=0) >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna probabilidades médias.
        
        Args:
            X: Features de teste
            
        Returns:
            Probabilidades (n_samples, 2)
        """
        probas = np.array([
            clf.predict_proba(X) 
            for clf in self.fitted_classifiers_.values()
        ])
        return np.mean(probas, axis=0)


# =============================================================================
# Funções utilitárias
# =============================================================================

def create_classifier(classifier_type: ClassifierType,
                      config: Optional[ClassifierConfig] = None) -> BaseEstimator:
    """Factory function para criar classificador."""
    factory = ClassifierFactory(config)
    return factory.create_classifier(classifier_type)


def create_validator(config: Optional[ClassifierConfig] = None) -> CrossValidator:
    """Factory function para criar validador."""
    return CrossValidator(config)


def quick_validate(X: np.ndarray, y: np.ndarray,
                   classifier_type: ClassifierType = ClassifierType.LR,
                   n_folds: int = 5) -> float:
    """
    Validação rápida retornando apenas acurácia média.
    
    Args:
        X: Features
        y: Labels
        classifier_type: Tipo de classificador
        n_folds: Número de folds
        
    Returns:
        Acurácia média
    """
    config = ClassifierConfig()
    config.n_folds = n_folds
    
    validator = CrossValidator(config)
    result = validator.validate(X, y, classifier_type)
    
    return result.accuracy_mean


def compare_classifiers(X: np.ndarray, y: np.ndarray,
                        config: Optional[ClassifierConfig] = None
                        ) -> Dict[str, float]:
    """
    Compara todos os classificadores retornando acurácias.
    
    Args:
        X: Features
        y: Labels
        config: Configuração
        
    Returns:
        Dicionário de acurácias por classificador
    """
    validator = CrossValidator(config)
    results = validator.validate_all(X, y)
    
    return {name: result.accuracy_mean for name, result in results.items()}


# =============================================================================
# Funções de compatibilidade com código antigo
# =============================================================================

def mlp_classify(x_train, y_train, x_test, y_test):
    """Compatibilidade com função antiga."""
    config = ClassifierConfig()
    validator = CrossValidator(config)
    result = validator.validate(x_train, y_train, ClassifierType.MLP)
    
    return result.accuracy_mean, result.sensitivity_mean, result.precision_mean


def rf_classify(x_train, y_train, x_test, y_test):
    """Compatibilidade com função antiga."""
    config = ClassifierConfig()
    validator = CrossValidator(config)
    result = validator.validate(x_train, y_train, ClassifierType.RF)
    
    return result.accuracy_mean, result.sensitivity_mean, result.precision_mean


def linearsvm_classify(x_train, y_train, x_test, y_test):
    """Compatibilidade com função antiga."""
    config = ClassifierConfig()
    validator = CrossValidator(config)
    result = validator.validate(x_train, y_train, ClassifierType.SVM)
    
    return result.accuracy_mean, result.sensitivity_mean, result.precision_mean


def adaboost_classify(x_train, y_train, x_test, y_test):
    """Compatibilidade com função antiga."""
    config = ClassifierConfig()
    validator = CrossValidator(config)
    result = validator.validate(x_train, y_train, ClassifierType.ADABOOST)
    
    return result.accuracy_mean, result.sensitivity_mean, result.precision_mean

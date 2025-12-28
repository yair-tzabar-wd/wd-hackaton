"""
Training Pipeline for Hiring Success Prediction

A generic, modular training pipeline supporting:
- Linear Regression / Logistic Regression
- Gradient Boosting (sklearn's GradientBoosting + HistGradientBoosting)

Supports both classification (predict success) and regression (predict applicants/time) tasks.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class TaskType(Enum):
    """Type of ML task."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ModelType(Enum):
    """Supported model types."""
    LINEAR = "linear"
    GRADIENT_BOOSTING = "gradient_boosting"
    HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""
    
    # Task configuration
    task_type: TaskType = TaskType.CLASSIFICATION
    target_column: str = "target_is_successful"
    
    # Data split
    test_size: float = 0.2
    random_state: int = 42
    
    # Cross-validation
    cv_folds: int = 5
    
    # Feature configuration - columns to exclude from features
    # Note: target column is automatically excluded, so don't need to list it here
    drop_columns: List[str] = field(default_factory=lambda: [
        "req_id", "target_date", "applications_detail", "target_status"
    ])
    
    # Additional target-related columns to drop (avoids data leakage)
    leakage_columns: List[str] = field(default_factory=lambda: [
        "target_is_successful", "target_total_applicants"
    ])
    
    # Model parameters (will be overridden per model)
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Output
    output_dir: str = "./models"


# =============================================================================
# Feature Engineering
# =============================================================================

class FeatureEngineer:
    """Handles feature preprocessing and engineering."""
    
    # Column type detection patterns
    NUMERIC_PATTERNS = [
        "days_since_open", "total_active", "_all", "_good", "_moderate",
        "_N", "_D", "_C", "_B", "_A", "avg_skill_score", "target_total_applicants"
    ]
    
    CATEGORICAL_COLUMNS = ["req_is_internal", "req_seniority", "req_top_profession"]
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.preprocessor: Optional[ColumnTransformer] = None
        self._fitted = False
    
    def identify_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify numeric and categorical columns from dataframe."""
        # Build complete exclude set: drop columns + target + leakage columns
        exclude = set(self.config.drop_columns + [self.config.target_column])
        exclude.update(self.config.leakage_columns)
        
        numeric_cols = []
        categorical_cols = []
        
        for col in df.columns:
            if col in exclude:
                continue
            
            # Check if it's a known categorical column
            if col in self.CATEGORICAL_COLUMNS:
                categorical_cols.append(col)
                continue
            
            # Check if it matches numeric patterns
            if any(pattern in col for pattern in self.NUMERIC_PATTERNS):
                numeric_cols.append(col)
                continue
            
            # Infer from dtype
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_cols.append(col)
            elif df[col].dtype == 'object' or df[col].dtype == 'bool':
                categorical_cols.append(col)
        
        self.numeric_columns = numeric_cols
        self.categorical_columns = categorical_cols
        
        logger.info(f"Identified {len(numeric_cols)} numeric columns: {numeric_cols[:5]}...")
        logger.info(f"Identified {len(categorical_cols)} categorical columns: {categorical_cols}")
        
        return numeric_cols, categorical_cols
    
    def create_preprocessor(self) -> ColumnTransformer:
        """Create sklearn ColumnTransformer for preprocessing."""
        transformers = []
        
        if self.numeric_columns:
            transformers.append((
                'numeric',
                StandardScaler(),
                self.numeric_columns
            ))
        
        if self.categorical_columns:
            transformers.append((
                'categorical',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                self.categorical_columns
            ))
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop any columns not specified
        )
        
        return self.preprocessor
    
    def prepare_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target from dataframe.
        
        Returns:
            X: Feature dataframe
            y: Target series
        """
        # Build list of columns to drop (excluding the target column itself)
        drop_cols = [c for c in self.config.drop_columns if c in df.columns]
        
        # Add leakage columns (other target-related cols) but not the actual target
        leakage_cols = [
            c for c in self.config.leakage_columns 
            if c in df.columns and c != self.config.target_column
        ]
        drop_cols.extend(leakage_cols)
        
        df_clean = df.drop(columns=drop_cols, errors='ignore')
        
        # Separate features and target
        if self.config.target_column not in df_clean.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in data")
        
        y = df_clean[self.config.target_column].copy()
        X = df_clean.drop(columns=[self.config.target_column])
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Convert boolean target for classification
        if self.config.task_type == TaskType.CLASSIFICATION:
            y = y.astype(int)
        
        return X, y
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        X = X.copy()
        
        # Numeric columns: fill with median
        for col in self.numeric_columns:
            if col in X.columns and X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        
        # Categorical columns: fill with 'Unknown'
        for col in self.categorical_columns:
            if col in X.columns and X[col].isna().any():
                X[col] = X[col].fillna('Unknown')
        
        return X


# =============================================================================
# Model Factory
# =============================================================================

class ModelFactory:
    """Factory for creating sklearn models."""
    
    @staticmethod
    def create_model(
        model_type: ModelType,
        task_type: TaskType,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Create a model based on type and task.
        
        Args:
            model_type: Type of model (linear, gradient_boosting, etc.)
            task_type: Classification or regression
            params: Optional model parameters
            
        Returns:
            Sklearn estimator
        """
        params = params or {}
        
        if model_type == ModelType.LINEAR:
            if task_type == TaskType.CLASSIFICATION:
                default_params = {
                    'max_iter': 1000,
                    'random_state': 42,
                    'solver': 'lbfgs',
                }
                default_params.update(params)
                return LogisticRegression(**default_params)
            else:
                # Use Ridge for regularization by default
                default_params = {'alpha': 1.0}
                default_params.update(params)
                return Ridge(**default_params)
        
        elif model_type == ModelType.GRADIENT_BOOSTING:
            if task_type == TaskType.CLASSIFICATION:
                default_params = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': 42,
                }
                default_params.update(params)
                return GradientBoostingClassifier(**default_params)
            else:
                default_params = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': 42,
                }
                default_params.update(params)
                return GradientBoostingRegressor(**default_params)
        
        elif model_type == ModelType.HIST_GRADIENT_BOOSTING:
            # HistGradientBoosting handles missing values natively
            if task_type == TaskType.CLASSIFICATION:
                default_params = {
                    'max_iter': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': 42,
                }
                default_params.update(params)
                return HistGradientBoostingClassifier(**default_params)
            else:
                default_params = {
                    'max_iter': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': 42,
                }
                default_params.update(params)
                return HistGradientBoostingRegressor(**default_params)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Metrics Calculator
# =============================================================================

class MetricsCalculator:
    """Calculate and report model metrics."""
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        if y_prob is not None:
            try:
                # For binary classification
                if len(y_prob.shape) == 1 or y_prob.shape[1] == 2:
                    prob = y_prob if len(y_prob.shape) == 1 else y_prob[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_true, prob)
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except ValueError:
                # ROC AUC undefined for single class
                metrics['roc_auc'] = None
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
        """Pretty print metrics."""
        print(f"\n{'='*50}")
        print(f" {title}")
        print(f"{'='*50}")
        for name, value in metrics.items():
            if value is not None:
                print(f"  {name:15s}: {value:.4f}")
        print(f"{'='*50}\n")


# =============================================================================
# Training Pipeline
# =============================================================================

class TrainingPipeline:
    """
    Main training pipeline orchestrator.
    
    Example usage:
        config = TrainingConfig(
            task_type=TaskType.CLASSIFICATION,
            target_column="target_is_successful"
        )
        pipeline = TrainingPipeline(config)
        results = pipeline.train(df, ModelType.GRADIENT_BOOSTING)
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.metrics_calculator = MetricsCalculator()
        
        # Will be set during training
        self.model: Optional[Pipeline] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
    
    def load_data(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load dataset from file."""
        path = Path(path)
        
        if path.suffix == '.csv':
            df = pd.read_csv(path)
        elif path.suffix == '.parquet':
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training.
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Identify column types
        self.feature_engineer.identify_columns(df)
        
        # Prepare features and target
        X, y = self.feature_engineer.prepare_data(df)
        
        # Split data - only stratify if we have multiple classes
        stratify = None
        if self.config.task_type == TaskType.CLASSIFICATION:
            n_classes = len(y.unique())
            if n_classes > 1:
                stratify = y
            else:
                logger.warning("Only one class in data - cannot stratify split")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def build_pipeline(self, model_type: ModelType) -> Pipeline:
        """
        Build the full sklearn pipeline (preprocessing + model).
        
        Args:
            model_type: Type of model to use
            
        Returns:
            Sklearn Pipeline
        """
        # Create preprocessor
        preprocessor = self.feature_engineer.create_preprocessor()
        
        # Create model
        model = ModelFactory.create_model(
            model_type=model_type,
            task_type=self.config.task_type,
            params=self.config.model_params
        )
        
        # Combine into pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        return pipeline
    
    def train(
        self,
        df: pd.DataFrame,
        model_type: ModelType,
        run_cv: bool = True
    ) -> Dict[str, Any]:
        """
        Train a model on the dataset.
        
        Args:
            df: Training dataframe
            model_type: Type of model to train
            run_cv: Whether to run cross-validation
            
        Returns:
            Dict with training results including metrics and model
        """
        logger.info(f"Starting training with {model_type.value} model")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Build pipeline
        pipeline = self.build_pipeline(model_type)
        
        # Run cross-validation if requested
        cv_scores = None
        if run_cv:
            # Check if we have enough samples and classes for CV
            n_classes = len(y_train.unique()) if self.config.task_type == TaskType.CLASSIFICATION else None
            n_samples = len(y_train)
            cv_folds = min(self.config.cv_folds, n_samples)
            
            if n_classes is not None and n_classes < 2:
                logger.warning("Only one class in training data - skipping cross-validation")
            elif cv_folds < 2:
                logger.warning("Not enough samples for cross-validation")
            else:
                try:
                    scoring = 'accuracy' if self.config.task_type == TaskType.CLASSIFICATION else 'r2'
                    cv_scores = cross_val_score(
                        pipeline, X_train, y_train,
                        cv=cv_folds,
                        scoring=scoring
                    )
                    logger.info(f"CV {scoring}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                except ValueError as e:
                    logger.warning(f"Cross-validation failed: {e}")
        
        # Train on full training set
        try:
            pipeline.fit(X_train, y_train)
            self.model = pipeline
        except ValueError as e:
            if "only one class" in str(e).lower():
                logger.error("Cannot train classification model with only one class in data")
                raise ValueError(
                    "Training data contains only one class. "
                    "For classification, you need examples of both success and failure cases."
                ) from e
            raise
        
        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        
        if self.config.task_type == TaskType.CLASSIFICATION:
            # Get probabilities for ROC AUC
            y_prob = None
            if hasattr(pipeline.named_steps['model'], 'predict_proba'):
                y_prob = pipeline.named_steps['model'].predict_proba(
                    pipeline.named_steps['preprocessor'].transform(X_test)
                )
            
            metrics = self.metrics_calculator.calculate_classification_metrics(
                y_test.values, y_pred, y_prob
            )
            self.metrics_calculator.print_metrics(
                metrics, f"{model_type.value} - Classification Results"
            )
            
            # Print detailed classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
        else:
            metrics = self.metrics_calculator.calculate_regression_metrics(
                y_test.values, y_pred
            )
            self.metrics_calculator.print_metrics(
                metrics, f"{model_type.value} - Regression Results"
            )
        
        return {
            'model': pipeline,
            'model_type': model_type,
            'metrics': metrics,
            'cv_scores': cv_scores,
            'y_test': y_test,
            'y_pred': y_pred,
        }
    
    def train_all_models(self, df: pd.DataFrame) -> Dict[ModelType, Dict[str, Any]]:
        """
        Train all supported models and compare.
        
        Args:
            df: Training dataframe
            
        Returns:
            Dict mapping model types to their results
        """
        results = {}
        
        for model_type in [ModelType.LINEAR, ModelType.GRADIENT_BOOSTING]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_type.value}")
            logger.info(f"{'='*60}")
            
            # Reset feature engineer for fresh column detection
            self.feature_engineer = FeatureEngineer(self.config)
            
            results[model_type] = self.train(df, model_type)
        
        # Print comparison
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict[ModelType, Dict[str, Any]]):
        """Print comparison of model results."""
        print("\n" + "="*70)
        print(" MODEL COMPARISON")
        print("="*70)
        
        if self.config.task_type == TaskType.CLASSIFICATION:
            print(f"{'Model':<25} {'Accuracy':<12} {'F1':<12} {'ROC AUC':<12}")
            print("-"*70)
            for model_type, result in results.items():
                m = result['metrics']
                roc = m.get('roc_auc', 'N/A')
                roc_str = f"{roc:.4f}" if isinstance(roc, float) else roc
                print(f"{model_type.value:<25} {m['accuracy']:<12.4f} {m['f1']:<12.4f} {roc_str:<12}")
        else:
            print(f"{'Model':<25} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
            print("-"*70)
            for model_type, result in results.items():
                m = result['metrics']
                print(f"{model_type.value:<25} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<12.4f}")
        
        print("="*70 + "\n")
    
    def save_model(self, path: Union[str, Path], model: Optional[Pipeline] = None):
        """Save trained model to disk."""
        model = model or self.model
        if model is None:
            raise ValueError("No model to save. Train a model first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]) -> Pipeline:
        """Load model from disk."""
        path = Path(path)
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model."""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importances (for tree-based models).
        
        Returns:
            DataFrame with feature names and importances, or None
        """
        if self.model is None:
            return None
        
        model = self.model.named_steps['model']
        
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not support feature importances")
            return None
        
        # Get feature names from preprocessor
        preprocessor = self.model.named_steps['preprocessor']
        feature_names = []
        
        for name, trans, cols in preprocessor.transformers_:
            if name == 'numeric':
                feature_names.extend(cols)
            elif name == 'categorical':
                # Get encoded feature names
                if hasattr(trans, 'get_feature_names_out'):
                    encoded_names = trans.get_feature_names_out(cols)
                    feature_names.extend(encoded_names)
                else:
                    feature_names.extend(cols)
        
        importances = model.feature_importances_
        
        # Handle length mismatch
        if len(feature_names) != len(importances):
            logger.warning(f"Feature names ({len(feature_names)}) don't match importances ({len(importances)})")
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df_importance


# =============================================================================
# Convenience Functions
# =============================================================================

def train_classification_model(
    data_path: str,
    model_type: ModelType = ModelType.GRADIENT_BOOSTING,
    target_column: str = "target_is_successful",
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to train a classification model.
    
    Args:
        data_path: Path to CSV/parquet file
        model_type: Type of model to train
        target_column: Name of target column
        **kwargs: Additional config parameters
        
    Returns:
        Training results dict
    """
    config = TrainingConfig(
        task_type=TaskType.CLASSIFICATION,
        target_column=target_column,
        **kwargs
    )
    
    pipeline = TrainingPipeline(config)
    df = pipeline.load_data(data_path)
    return pipeline.train(df, model_type)


def train_regression_model(
    data_path: str,
    model_type: ModelType = ModelType.GRADIENT_BOOSTING,
    target_column: str = "target_total_applicants",
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to train a regression model.
    
    Args:
        data_path: Path to CSV/parquet file
        model_type: Type of model to train
        target_column: Name of target column
        **kwargs: Additional config parameters
        
    Returns:
        Training results dict
    """
    config = TrainingConfig(
        task_type=TaskType.REGRESSION,
        target_column=target_column,
        **kwargs
    )
    
    pipeline = TrainingPipeline(config)
    df = pipeline.load_data(data_path)
    return pipeline.train(df, model_type)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train hiring prediction models")
    parser.add_argument("data_path", help="Path to training data (CSV or Parquet)")
    parser.add_argument(
        "--task", choices=["classification", "regression"],
        default="classification", help="Task type"
    )
    parser.add_argument(
        "--target", default="target_is_successful",
        help="Target column name"
    )
    parser.add_argument(
        "--model", choices=["linear", "gradient_boosting", "all"],
        default="all", help="Model type to train"
    )
    parser.add_argument(
        "--output", default="./models",
        help="Output directory for models"
    )
    
    args = parser.parse_args()
    
    # Create config
    task_type = TaskType.CLASSIFICATION if args.task == "classification" else TaskType.REGRESSION
    
    config = TrainingConfig(
        task_type=task_type,
        target_column=args.target,
        output_dir=args.output
    )
    
    # Initialize pipeline
    pipeline = TrainingPipeline(config)
    
    # Load data
    df = pipeline.load_data(args.data_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Target column: {args.target}")
    print(f"Task type: {task_type.value}\n")
    
    # Train models
    if args.model == "all":
        results = pipeline.train_all_models(df)
        
        # Save best model
        best_model_type = max(
            results.keys(),
            key=lambda mt: results[mt]['metrics'].get('f1' if task_type == TaskType.CLASSIFICATION else 'r2', 0)
        )
        best_result = results[best_model_type]
        
        output_path = Path(args.output) / f"best_model_{task_type.value}.joblib"
        pipeline.save_model(output_path, best_result['model'])
        print(f"\nBest model ({best_model_type.value}) saved to {output_path}")
        
        # Print feature importances for best model if available
        pipeline.model = best_result['model']
        importance = pipeline.get_feature_importance()
        if importance is not None:
            print("\nTop 10 Feature Importances:")
            print(importance.head(10).to_string(index=False))
    else:
        model_type = ModelType.LINEAR if args.model == "linear" else ModelType.GRADIENT_BOOSTING
        result = pipeline.train(df, model_type)
        
        output_path = Path(args.output) / f"{model_type.value}_{task_type.value}.joblib"
        pipeline.save_model(output_path)
        
        # Print feature importances if available
        importance = pipeline.get_feature_importance()
        if importance is not None:
            print("\nTop 10 Feature Importances:")
            print(importance.head(10).to_string(index=False))


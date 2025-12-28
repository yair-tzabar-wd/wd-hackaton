"""
Model Trainer for Hiring Pipeline Predictor

This module trains an XGBoost model to predict the "Gap" - how many more
candidates a hiring manager needs to fill a position.

The model learns patterns like:
- "If Seniority=Senior and Current_Count=5, the Gap is usually 0"
- "If Profession=Engineering and is_internal=0, typically need 80+ candidates"

Developer 2 - The Model Builder
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass, field
import json
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from hs_gimme.logging_service.logging_service import LoggingService

logger = LoggingService(logging.getLogger(__name__))


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for the XGBoost model."""
    
    # Target column
    target_col: str = "gap"
    
    # Feature columns (can be auto-detected)
    feature_cols: Optional[List[str]] = None
    
    # Columns to exclude from features
    exclude_cols: List[str] = field(default_factory=lambda: [
        "req_id", "candidate_id", "date", "gap", "magic_number",
        "_id", "_extraction_error"
    ])
    
    # Categorical columns (for encoding)
    categorical_cols: List[str] = field(default_factory=lambda: [
        "seniority_level", "profession_group", "dominant_seniority",
        "dominant_profession", "job_title", "country", "job_band"
    ])
    
    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    early_stopping_rounds: int = 10
    
    # Regularization
    reg_alpha: float = 0.0  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    mae: float
    rmse: float
    r2: float
    mape: Optional[float] = None
    cv_scores: Optional[List[float]] = None
    feature_importance: Optional[Dict[str, float]] = None


# =============================================================================
# FEATURE PREPROCESSOR
# =============================================================================

class FeaturePreprocessor:
    """Handles feature preprocessing for the model."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_cols: List[str] = []
        self._fitted = False
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Tuple of (X features array, y target array)
        """
        # Identify feature columns
        self.feature_cols = self._identify_feature_cols(df)
        logger.info(f"Using features: {self.feature_cols}")
        
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Encode categorical columns
        for col in self.config.categorical_cols:
            if col in df_processed.columns:
                df_processed = self._encode_categorical(df_processed, col, fit=True)
        
        # Extract features and target
        X = df_processed[self.feature_cols].values
        y = df_processed[self.config.target_col].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        self._fitted = True
        return X, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Feature array
        """
        if not self._fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        df_processed = df.copy()
        
        # Encode categorical columns
        for col in self.config.categorical_cols:
            if col in df_processed.columns:
                df_processed = self._encode_categorical(df_processed, col, fit=False)
        
        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        X = df_processed[self.feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        
        return X
    
    def _identify_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Identify which columns to use as features."""
        if self.config.feature_cols:
            return [c for c in self.config.feature_cols if c in df.columns]
        
        # Auto-detect: all columns except excluded ones
        feature_cols = []
        for col in df.columns:
            if col in self.config.exclude_cols:
                continue
            if col in self.config.categorical_cols:
                # Will be encoded, include it
                feature_cols.append(col)
            elif df[col].dtype in [np.int64, np.float64, np.int32, np.float32, int, float]:
                feature_cols.append(col)
        
        return feature_cols
    
    def _encode_categorical(
        self,
        df: pd.DataFrame,
        col: str,
        fit: bool
    ) -> pd.DataFrame:
        """Encode a categorical column."""
        if col not in df.columns:
            return df
        
        # Fill NaN with 'Unknown'
        df[col] = df[col].fillna("Unknown").astype(str)
        
        if fit:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            self.label_encoders[col] = encoder
        else:
            encoder = self.label_encoders.get(col)
            if encoder:
                # Handle unseen labels
                known_labels = set(encoder.classes_)
                df[col] = df[col].apply(
                    lambda x: x if x in known_labels else "Unknown"
                )
                if "Unknown" not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, "Unknown")
                df[col] = encoder.transform(df[col])
        
        return df


# =============================================================================
# MODEL TRAINER
# =============================================================================

class GapModelTrainer:
    """
    Trains and manages the XGBoost model for gap prediction.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Model configuration (uses defaults if None)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is required. Install with: pip install xgboost"
            )
        
        self.config = config or ModelConfig()
        self.preprocessor = FeaturePreprocessor(self.config)
        self.model: Optional[xgb.XGBRegressor] = None
        self.metrics: Optional[ModelMetrics] = None
    
    def train(
        self,
        training_df: pd.DataFrame,
        validation_df: Optional[pd.DataFrame] = None,
        verbose: bool = True
    ) -> ModelMetrics:
        """
        Train the XGBoost model.
        
        Args:
            training_df: Training data with features and gap target
            validation_df: Optional separate validation data
            verbose: Whether to print training progress
            
        Returns:
            ModelMetrics with evaluation results
        """
        logger.info("Starting model training")
        
        # Preprocess data
        X, y = self.preprocessor.fit_transform(training_df)
        
        # Split if no validation set provided
        if validation_df is not None:
            X_train, y_train = X, y
            X_val = self.preprocessor.transform(validation_df)
            y_val = validation_df[self.config.target_col].values
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
        
        logger.info(
            "Data split complete",
            train_size=len(X_train),
            val_size=len(X_val)
        )
        
        # Initialize model
        self.model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbosity=1 if verbose else 0
        )
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose
        )
        
        # Evaluate
        self.metrics = self._evaluate(X_val, y_val)
        
        # Get feature importance
        self.metrics.feature_importance = self._get_feature_importance()
        
        logger.info(
            "Training complete",
            mae=self.metrics.mae,
            rmse=self.metrics.rmse,
            r2=self.metrics.r2
        )
        
        return self.metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of predicted gaps
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.preprocessor.transform(df)
        predictions = self.model.predict(X)
        
        # Gap can't be negative
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_with_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions and return with context.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with original data plus predictions
        """
        predictions = self.predict(df)
        result = df.copy()
        result["predicted_gap"] = predictions
        result["predicted_gap_rounded"] = np.round(predictions).astype(int)
        
        return result
    
    def save_model(self, path: Union[str, Path]) -> Path:
        """
        Save the trained model and preprocessor.
        
        Args:
            path: Directory to save model files
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_path = path / "gap_model.json"
        self.model.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            "config": {
                "target_col": self.config.target_col,
                "feature_cols": self.preprocessor.feature_cols,
                "categorical_cols": self.config.categorical_cols,
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
                "learning_rate": self.config.learning_rate
            },
            "metrics": {
                "mae": self.metrics.mae if self.metrics else None,
                "rmse": self.metrics.rmse if self.metrics else None,
                "r2": self.metrics.r2 if self.metrics else None
            },
            "feature_importance": self.metrics.feature_importance if self.metrics else None,
            "saved_at": datetime.now().isoformat()
        }
        
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save label encoders
        encoder_data = {}
        for col, encoder in self.preprocessor.label_encoders.items():
            encoder_data[col] = encoder.classes_.tolist()
        
        encoder_path = path / "encoders.json"
        with open(encoder_path, "w") as f:
            json.dump(encoder_data, f, indent=2)
        
        logger.info(f"Model saved to {path}")
        return path
    
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load a saved model.
        
        Args:
            path: Directory containing saved model files
        """
        path = Path(path)
        
        # Load XGBoost model
        model_path = path / "gap_model.json"
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(model_path))
        
        # Load metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        self.preprocessor.feature_cols = metadata["config"]["feature_cols"]
        
        # Load label encoders
        encoder_path = path / "encoders.json"
        with open(encoder_path) as f:
            encoder_data = json.load(f)
        
        for col, classes in encoder_data.items():
            encoder = LabelEncoder()
            encoder.classes_ = np.array(classes)
            self.preprocessor.label_encoders[col] = encoder
        
        self.preprocessor._fitted = True
        logger.info(f"Model loaded from {path}")
    
    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Evaluate model on validation data."""
        predictions = self.model.predict(X)
        predictions = np.maximum(predictions, 0)  # Gap can't be negative
        
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        
        # MAPE only if no zeros in y
        mape = None
        if not np.any(y == 0):
            mape = mean_absolute_percentage_error(y, predictions)
        
        return ModelMetrics(
            mae=float(mae),
            rmse=float(rmse),
            r2=float(r2),
            mape=float(mape) if mape is not None else None
        )
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        feature_names = self.preprocessor.feature_cols
        
        return dict(sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        ))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def train_gap_model(
    training_df: pd.DataFrame,
    config: Optional[ModelConfig] = None,
    save_path: Optional[Union[str, Path]] = None
) -> Tuple[GapModelTrainer, ModelMetrics]:
    """
    Convenience function to train a gap prediction model.
    
    Args:
        training_df: Training data with features and gap target
        config: Optional model configuration
        save_path: Optional path to save model
        
    Returns:
        Tuple of (trainer, metrics)
    """
    trainer = GapModelTrainer(config)
    metrics = trainer.train(training_df)
    
    if save_path:
        trainer.save_model(save_path)
    
    return trainer, metrics


# =============================================================================
# MAIN ENTRY POINT (for testing)
# =============================================================================

if __name__ == "__main__":
    print("Gap Model Trainer Module")
    print("=" * 50)
    
    if not XGBOOST_AVAILABLE:
        print("\nWARNING: XGBoost not installed!")
        print("Install with: pip install xgboost")
    else:
        print("\nXGBoost is available.")
        
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        synthetic_data = pd.DataFrame({
            "current_count": np.random.poisson(20, n_samples),
            "is_internal": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            "seniority_numeric": np.random.choice([1, 2, 3, 4, 5], n_samples),
            "avg_candidate_quality": np.random.uniform(0.4, 0.9, n_samples),
            "internal_ratio": np.random.uniform(0, 0.3, n_samples),
        })
        
        # Create target: gap depends on features
        synthetic_data["gap"] = (
            50  # Base gap
            - synthetic_data["current_count"] * 0.8
            + synthetic_data["seniority_numeric"] * 5
            - synthetic_data["is_internal"] * 10
            + np.random.normal(0, 5, n_samples)
        ).clip(lower=0)
        
        print(f"\nSynthetic training data shape: {synthetic_data.shape}")
        print("\nTraining model...")
        
        trainer, metrics = train_gap_model(synthetic_data)
        
        print(f"\nModel Metrics:")
        print(f"  MAE:  {metrics.mae:.2f}")
        print(f"  RMSE: {metrics.rmse:.2f}")
        print(f"  R²:   {metrics.r2:.4f}")
        
        print("\nTop Feature Importance:")
        for feature, importance in list(metrics.feature_importance.items())[:5]:
            print(f"  {feature}: {importance:.4f}")


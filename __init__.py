"""
Hiring Pipeline Predictor - The "Gap" Model

This package provides tools to predict how many more candidates a 
Hiring Manager needs to fill a position.

Uses hs_gimme constants (Req, Application, SamuraiJson, etc.) for proper
field access following HiredScore conventions.

Developer 2 Components:
- feature_engine: Static feature extraction from job applications
- success_calculator: Calculate "Magic Numbers" for successful requisitions
- gap_calculator: Merge timeline data and calculate the Gap (target variable)
- model_trainer: Train XGBoost model to predict the Gap
- db_utils: MongoDB connection utilities using gimme

Usage:
    from wd_hackaton import HiringPipelinePredictor, PipelineConfig
    
    config = PipelineConfig(
        environment='production',
        account_id='my_account'
    )
    
    pipeline = HiringPipelinePredictor(config)
    pipeline.run()
"""

from .feature_engine import (
    extract_static_features,
    extract_features_batch,
    aggregate_features_by_req,
)

from .success_calculator import (
    SuccessCalculator,
)

from .gap_calculator import (
    GapCalculator,
    create_gap_calculator,
    calculate_gaps_from_timeline,
    generate_synthetic_timeline,
)

from .model_trainer import (
    GapModelTrainer,
    ModelConfig,
    ModelMetrics,
    train_gap_model,
)

from .db_utils import (
    HiringDataClient,
    create_data_client,
)

from .main import (
    HiringPipelinePredictor,
    PipelineConfig,
)

__version__ = "0.1.0"
__author__ = "Developer 2 - Context Builder"

__all__ = [
    # Feature Engine
    "extract_static_features",
    "extract_features_batch",
    "aggregate_features_by_req",
    # Success Calculator
    "SuccessCalculator",
    # Gap Calculator
    "GapCalculator",
    "create_gap_calculator",
    "calculate_gaps_from_timeline",
    "generate_synthetic_timeline",
    # Model Trainer
    "GapModelTrainer",
    "ModelConfig",
    "ModelMetrics",
    "train_gap_model",
    # DB Utils
    "HiringDataClient",
    "create_data_client",
    # Main Pipeline
    "HiringPipelinePredictor",
    "PipelineConfig",
]


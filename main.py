#!/usr/bin/env python3
"""
Hiring Pipeline Predictor - Main Entry Point

This is the main orchestrator for the "Gap" Model pipeline.
It coordinates all components to train a model that predicts:
"How many more candidates does a Hiring Manager need?"

Developer 2 - Pipeline Orchestrator
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

from hs_gimme.logging_service.logging_service import LoggingService
from hs_gimme.env_utils.env_utils import get_current_environment

# HiredScore constants - using proper gimme package classes
from hs_gimme.constants.objects.req import Req
from hs_gimme.constants.objects.application import Application

# Local imports
from db_utils import create_data_client, HiringDataClient
from feature_engine import extract_static_features, extract_features_batch, aggregate_features_by_req
from success_calculator import SuccessCalculator
from gap_calculator import GapCalculator, generate_synthetic_timeline
from model_trainer import GapModelTrainer, ModelConfig, train_gap_model

logger = LoggingService(logging.getLogger(__name__))


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

class PipelineConfig:
    """Configuration for the training pipeline."""
    
    def __init__(
        self,
        environment: str,
        account_id: str,
        output_dir: str = "./output",
        min_applications: int = 5,
        max_applications: int = 500,
        use_synthetic_timeline: bool = True,
        synthetic_days_per_req: int = 30
    ):
        self.environment = environment
        self.account_id = account_id
        self.output_dir = Path(output_dir)
        self.min_applications = min_applications
        self.max_applications = max_applications
        self.use_synthetic_timeline = use_synthetic_timeline
        self.synthetic_days_per_req = synthetic_days_per_req
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.data_dir = self.output_dir / "data"
        self.model_dir = self.output_dir / "models"
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class HiringPipelinePredictor:
    """
    Main pipeline for training the Hiring Gap Prediction model.
    Uses hs_gimme constants (Req, Application) for field access.
    
    This orchestrates:
    1. Data extraction from MongoDB (using gimme)
    2. Static feature extraction
    3. Success calculation (Magic Numbers)
    4. Gap calculation (merging with Developer 1's timeline data)
    5. Model training (XGBoost)
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.data_client: Optional[HiringDataClient] = None
        self.success_calculator: Optional[SuccessCalculator] = None
        self.gap_calculator: Optional[GapCalculator] = None
        self.trainer: Optional[GapModelTrainer] = None
        
        # Data containers
        self.magic_numbers_df: Optional[pd.DataFrame] = None
        self.training_df: Optional[pd.DataFrame] = None
        
        logger.info(
            "Pipeline initialized",
            environment=config.environment,
            account_id=config.account_id,
            output_dir=str(config.output_dir)
        )
    
    def run(self, timeline_df: Optional[pd.DataFrame] = None) -> None:
        """
        Run the complete pipeline.
        
        Args:
            timeline_df: Optional DataFrame from Developer 1's Replayer
                        If None and use_synthetic_timeline=True, generates synthetic data
        """
        logger.info("=" * 60)
        logger.info("Starting Hiring Pipeline Predictor")
        logger.info("=" * 60)
        
        try:
            # Step 1: Initialize connections
            self._initialize_components()
            
            # Step 2: Calculate Magic Numbers
            self._calculate_magic_numbers()
            
            # Step 3: Prepare timeline data
            timeline_df = self._prepare_timeline_data(timeline_df)
            
            # Step 4: Calculate gaps and create training data
            self._create_training_data(timeline_df)
            
            # Step 5: Train the model
            self._train_model()
            
            # Step 6: Save outputs
            self._save_outputs()
            
            logger.info("=" * 60)
            logger.info("Pipeline completed successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Step 1: Initializing components...")
        
        self.data_client = create_data_client(
            self.config.environment,
            self.config.account_id
        )
        
        self.success_calculator = SuccessCalculator(self.data_client)
        self.gap_calculator = GapCalculator(self.data_client, self.success_calculator)
        
        logger.info("Components initialized successfully")
    
    def _calculate_magic_numbers(self) -> None:
        """Calculate Magic Numbers for all successful requisitions."""
        logger.info("Step 2: Calculating Magic Numbers...")
        
        magic_numbers = self.gap_calculator.prepare_magic_numbers(
            min_applications=self.config.min_applications,
            max_applications=self.config.max_applications
        )
        
        self.magic_numbers_df = self.success_calculator.to_dataframe()
        
        logger.info(
            f"Magic Numbers calculated for {len(magic_numbers)} requisitions"
        )
    
    def _prepare_timeline_data(
        self,
        timeline_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Prepare timeline data.
        
        If no timeline data is provided and synthetic is enabled,
        generates synthetic timeline data for testing.
        """
        logger.info("Step 3: Preparing timeline data...")
        
        if timeline_df is not None:
            logger.info(f"Using provided timeline data: {len(timeline_df)} rows")
            return timeline_df
        
        if self.config.use_synthetic_timeline:
            logger.info("Generating synthetic timeline data...")
            
            # Use requisition IDs from magic numbers
            req_ids = list(self.gap_calculator._magic_numbers.keys())
            
            # Limit for testing
            req_ids = req_ids[:100] if len(req_ids) > 100 else req_ids
            
            timeline_df = generate_synthetic_timeline(
                req_ids=req_ids,
                days_per_req=self.config.synthetic_days_per_req
            )
            
            logger.info(f"Generated synthetic timeline: {len(timeline_df)} rows")
            return timeline_df
        
        raise ValueError(
            "No timeline data provided and use_synthetic_timeline=False. "
            "Please provide timeline_df from Developer 1's Replayer."
        )
    
    def _create_training_data(self, timeline_df: pd.DataFrame) -> None:
        """Create the complete training dataset."""
        logger.info("Step 4: Creating training dataset...")
        
        self.training_df = self.gap_calculator.create_training_dataset(
            timeline_df=timeline_df,
            include_features=True
        )
        
        logger.info(
            f"Training dataset created: {len(self.training_df)} rows, "
            f"{len(self.training_df.columns)} columns"
        )
        
        # Log some statistics
        if "gap" in self.training_df.columns:
            logger.info(f"  Gap statistics:")
            logger.info(f"    Mean: {self.training_df['gap'].mean():.2f}")
            logger.info(f"    Median: {self.training_df['gap'].median():.2f}")
            logger.info(f"    Std: {self.training_df['gap'].std():.2f}")
            logger.info(f"    Min: {self.training_df['gap'].min()}")
            logger.info(f"    Max: {self.training_df['gap'].max()}")
    
    def _train_model(self) -> None:
        """Train the XGBoost model."""
        logger.info("Step 5: Training XGBoost model...")
        
        if self.training_df is None or len(self.training_df) == 0:
            raise ValueError("No training data available")
        
        # Configure model
        model_config = ModelConfig(
            target_col="gap",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            test_size=0.2
        )
        
        self.trainer = GapModelTrainer(model_config)
        metrics = self.trainer.train(self.training_df, verbose=True)
        
        logger.info("Model training complete!")
        logger.info(f"  MAE:  {metrics.mae:.2f}")
        logger.info(f"  RMSE: {metrics.rmse:.2f}")
        logger.info(f"  R²:   {metrics.r2:.4f}")
        
        if metrics.feature_importance:
            logger.info("  Top features:")
            for feature, importance in list(metrics.feature_importance.items())[:5]:
                logger.info(f"    {feature}: {importance:.4f}")
    
    def _save_outputs(self) -> None:
        """Save all outputs to disk."""
        logger.info("Step 6: Saving outputs...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save magic numbers
        if self.magic_numbers_df is not None:
            magic_path = self.config.data_dir / f"magic_numbers_{timestamp}.parquet"
            self.magic_numbers_df.to_parquet(magic_path)
            logger.info(f"  Saved magic numbers to: {magic_path}")
        
        # Save training data
        if self.training_df is not None:
            training_path = self.config.data_dir / f"training_data_{timestamp}.parquet"
            self.training_df.to_parquet(training_path)
            logger.info(f"  Saved training data to: {training_path}")
        
        # Save model
        if self.trainer is not None:
            model_path = self.config.model_dir / f"gap_model_{timestamp}"
            self.trainer.save_model(model_path)
            logger.info(f"  Saved model to: {model_path}")
    
    def predict(
        self,
        current_pipeline_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Make predictions for current pipeline states.
        
        Args:
            current_pipeline_df: DataFrame with current pipeline state
                Must include columns matching training features
                
        Returns:
            DataFrame with predictions added
        """
        if self.trainer is None:
            raise ValueError("Model not trained. Run pipeline first.")
        
        return self.trainer.predict_with_context(current_pipeline_df)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hiring Pipeline Predictor - Train the Gap Model"
    )
    
    parser.add_argument(
        "--environment", "-e",
        type=str,
        default=None,
        help="Environment (production, staging, local). Auto-detected if not specified."
    )
    
    parser.add_argument(
        "--account-id", "-a",
        type=str,
        required=True,
        help="Account/client ID"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output",
        help="Output directory for models and data"
    )
    
    parser.add_argument(
        "--min-applications",
        type=int,
        default=5,
        help="Minimum applications for successful requisitions"
    )
    
    parser.add_argument(
        "--max-applications",
        type=int,
        default=500,
        help="Maximum applications for successful requisitions"
    )
    
    parser.add_argument(
        "--timeline-file",
        type=str,
        default=None,
        help="Path to Developer 1's timeline data (parquet/csv)"
    )
    
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic timeline data for testing"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Auto-detect environment if not specified
    environment = args.environment or get_current_environment()
    
    # Load timeline data if provided
    timeline_df = None
    if args.timeline_file:
        timeline_path = Path(args.timeline_file)
        if timeline_path.suffix == ".parquet":
            timeline_df = pd.read_parquet(timeline_path)
        elif timeline_path.suffix == ".csv":
            timeline_df = pd.read_csv(timeline_path)
        else:
            raise ValueError(f"Unsupported timeline file format: {timeline_path.suffix}")
        logger.info(f"Loaded timeline data from {timeline_path}")
    
    # Create configuration
    config = PipelineConfig(
        environment=environment,
        account_id=args.account_id,
        output_dir=args.output_dir,
        min_applications=args.min_applications,
        max_applications=args.max_applications,
        use_synthetic_timeline=args.synthetic or (timeline_df is None)
    )
    
    # Run pipeline
    pipeline = HiringPipelinePredictor(config)
    pipeline.run(timeline_df=timeline_df)
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print(f"Outputs saved to: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()


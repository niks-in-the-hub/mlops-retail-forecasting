"""
Training functions for retail sales forecasting with AutoGluon Chronos.
Includes MLflow experiment tracking.
"""

import pandas as pd
import numpy as np
import logging
import mlflow
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from src.utils import (
    get_mlflow_tracking_uri, 
    create_output_dir,
    log_dict_as_params,
    log_metrics_dict,
    get_current_timestamp
)


# ============================================================================
# MLFLOW SETUP
# ============================================================================

def setup_mlflow(experiment_name="rossmann-forecasting"):
    """
    Initialize MLflow tracking.
    
    Args:
        experiment_name: Name of the MLflow experiment
    
    Returns:
        Experiment ID
    """
    logger = logging.getLogger(__name__)
    
    # Set tracking URI to local directory
    tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    # Create or get experiment
    experiment = mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name} (ID: {experiment.experiment_id})")
    
    return experiment.experiment_id


# ============================================================================
# DATA CONVERSION FOR AUTOGLUON
# ============================================================================

def convert_to_timeseries_dataframe(df, freq='D'):
    """
    Convert pandas DataFrame to AutoGluon TimeSeriesDataFrame.
    Handles missing dates by filling them.
    
    Args:
        df: DataFrame with timestamp, target, item_id columns
        freq: Frequency of the time series ('D' for daily, 'W' for weekly, etc.)
    
    Returns:
        TimeSeriesDataFrame for AutoGluon
    """
    logger = logging.getLogger(__name__)
    logger.info("Converting to TimeSeriesDataFrame...")
    
    # AutoGluon expects specific column names
    # timestamp -> index, item_id stays, target stays
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column='item_id',
        timestamp_column='timestamp'
    )
    
    logger.info(f"Created TimeSeriesDataFrame with {len(ts_df.item_ids)} items")
    
    # Convert to regular frequency (fill missing dates)
    # This is CRITICAL for AutoGluon to work properly
    logger.info(f"Converting to regular frequency: {freq}")
    ts_df = ts_df.convert_frequency(freq=freq)
    logger.info("Frequency conversion complete!")
    
    return ts_df


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(train_df, prediction_length=7, time_limit=600, model_path=None, freq='D'):
    """
    Train AutoGluon TimeSeriesPredictor.
    
    Args:
        train_df: Training data (pandas DataFrame or TimeSeriesDataFrame)
        prediction_length: Number of days to forecast (default: 7)
        time_limit: Training time limit in seconds (default: 600 = 10 mins)
        model_path: Path to save the model (optional)
        freq: Frequency of time series ('D' = daily, 'W' = weekly, etc.)
    
    Returns:
        Trained TimeSeriesPredictor
    """
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Starting model training")
    logger.info("="*60)
    
    # Convert to TimeSeriesDataFrame if needed
    if not isinstance(train_df, TimeSeriesDataFrame):
        train_df = convert_to_timeseries_dataframe(train_df, freq=freq)
    
    # Create model save path if not provided
    if model_path is None:
        model_dir = create_output_dir("models")
        timestamp = get_current_timestamp()
        model_path = f"{model_dir}/model_{timestamp}"
    
    logger.info(f"Model will be saved to: {model_path}")
    logger.info(f"Prediction length: {prediction_length} days")
    logger.info(f"Time limit: {time_limit} seconds")
    logger.info(f"Frequency: {freq}")
    
    # Initialize the predictor with frequency specified
    predictor = TimeSeriesPredictor(
        path=model_path,
        target='target',  # Column we're forecasting
        prediction_length=prediction_length,
        eval_metric='MASE',  # Mean Absolute Scaled Error (good for retail)
        freq=freq,  # CRITICAL: Specify frequency to avoid errors
        verbosity=2  # Show training progress
    )
    
    # Train the model
    # AutoGluon will automatically try multiple models and ensemble them
    logger.info("Training started...")
    predictor.fit(
        train_data=train_df,
        time_limit=time_limit,
        presets='medium_quality',  # Options: fast_training, medium_quality, high_quality, best_quality
        skip_model_selection=False  # Try multiple models
    )
    
    logger.info("Training complete!")
    logger.info(f"Model saved to: {model_path}")
    
    return predictor


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(predictor, val_df, freq='D'):
    """
    Evaluate the trained model on validation data.
    
    Args:
        predictor: Trained TimeSeriesPredictor
        val_df: Validation data (pandas DataFrame or TimeSeriesDataFrame)
        freq: Frequency of time series
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Evaluating model on validation set")
    logger.info("="*60)
    
    # Convert to TimeSeriesDataFrame if needed
    if not isinstance(val_df, TimeSeriesDataFrame):
        val_df = convert_to_timeseries_dataframe(val_df, freq=freq)
    
    # Get predictions on validation set
    logger.info("Generating predictions...")
    predictions = predictor.predict(val_df)
    
    # Calculate metrics using AutoGluon's built-in evaluation
    try:
        # Try to get leaderboard with validation scores
        leaderboard = predictor.leaderboard(val_df, silent=True)
        best_model = leaderboard.iloc[0]
        
        metrics = {
            'val_MASE': float(best_model['score_val']),
            'best_model': str(best_model['model']),
            'num_models_trained': len(leaderboard)
        }
    except Exception as e:
        logger.warning(f"Could not get leaderboard: {e}")
        metrics = {
            'best_model': 'unknown',
            'num_models_trained': 0
        }
    
    # Calculate additional metrics manually
    # Get actual values and predictions
    try:
        # Align predictions with validation data
        actual_list = []
        pred_list = []
        
        for item_id in val_df.item_ids:
            val_item = val_df.loc[item_id]
            pred_item = predictions.loc[item_id]
            
            # Get overlapping timestamps
            common_timestamps = val_item.index.intersection(pred_item.index)
            
            if len(common_timestamps) > 0:
                actual_list.extend(val_item.loc[common_timestamps, 'target'].values)
                pred_list.extend(pred_item.loc[common_timestamps, 'mean'].values)
        
        if len(actual_list) > 0:
            actuals = np.array(actual_list)
            preds = np.array(pred_list)
            
            # Calculate metrics
            mae = np.mean(np.abs(actuals - preds))
            rmse = np.sqrt(np.mean((actuals - preds) ** 2))
            
            # MAPE (avoid division by zero)
            mape = np.mean(np.abs((actuals - preds) / np.maximum(actuals, 1))) * 100
            
            metrics['val_MAE'] = float(mae)
            metrics['val_RMSE'] = float(rmse)
            metrics['val_MAPE'] = float(mape)
    
    except Exception as e:
        logger.warning(f"Could not calculate additional metrics: {e}")
    
    logger.info("Evaluation metrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")
    
    return metrics


# ============================================================================
# MLFLOW LOGGING
# ============================================================================

def log_training_to_mlflow(predictor, metrics, config):
    """
    Log training parameters, metrics, and model to MLflow.
    
    Args:
        predictor: Trained predictor
        metrics: Dictionary of evaluation metrics
        config: Dictionary of training configuration
    """
    logger = logging.getLogger(__name__)
    logger.info("Logging to MLflow...")
    
    # Separate metrics into numeric (for mlflow.log_metric) and string (for mlflow.log_param)
    numeric_metrics = {}
    string_metrics = {}
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            numeric_metrics[key] = value
        else:
            string_metrics[key] = value
    
    # Log parameters (config + string metrics)
    log_dict_as_params(config, prefix="config_")
    log_dict_as_params(string_metrics, prefix="")
    
    # Log only numeric metrics
    log_metrics_dict(numeric_metrics)
    
    # Log model leaderboard as artifact
    try:
        leaderboard = predictor.leaderboard(silent=True)
        leaderboard_path = f"{predictor.path}/leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        mlflow.log_artifact(leaderboard_path, artifact_path="model_info")
    except Exception as e:
        logger.warning(f"Could not log leaderboard: {e}")
    
    # Log the model path
    mlflow.log_param("model_path", predictor.path)
    
    logger.info("MLflow logging complete!")


# ============================================================================
# FULL TRAINING PIPELINE
# ============================================================================

def train_pipeline(train_df, val_df, config=None):
    """
    Full training pipeline with MLflow tracking.
    This is the main function to call.
    
    Args:
        train_df: Training data
        val_df: Validation data
        config: Training configuration dict (optional)
    
    Returns:
        Tuple of (predictor, metrics, model_path)
    """
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("="*60)
    
    # Default configuration
    if config is None:
        config = {
            'prediction_length': 7,
            'time_limit': 600,  # 10 minutes
            'presets': 'medium_quality',
            'eval_metric': 'MASE',
            'freq': 'D'  # Daily frequency
        }
    
    # Setup MLflow
    experiment_id = setup_mlflow()
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"training_{get_current_timestamp()}") as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")
        
        # Train model
        predictor = train_model(
            train_df=train_df,
            prediction_length=config['prediction_length'],
            time_limit=config['time_limit'],
            freq=config.get('freq', 'D')
        )
        
        # Evaluate model
        metrics = evaluate_model(predictor, val_df, freq=config.get('freq', 'D'))
        
        # Log to MLflow
        log_training_to_mlflow(predictor, metrics, config)
        
        model_path = predictor.path
        
        logger.info("="*60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"MLflow run ID: {run.info.run_id}")
        logger.info("="*60)
    
    return predictor, metrics, model_path


# ============================================================================
# HELPER: QUICK TRAINING FOR TESTING
# ============================================================================

def quick_train(train_df, val_df, time_limit=300):
    """
    Quick training function for fast experimentation.
    Uses shorter time limit and fast_training preset.
    
    Args:
        train_df: Training data
        val_df: Validation data
        time_limit: Time limit in seconds (default: 300 = 5 mins)
    
    Returns:
        Tuple of (predictor, metrics, model_path)
    """
    logger = logging.getLogger(__name__)
    logger.info("Running QUICK TRAINING (fast mode)...")
    
    config = {
        'prediction_length': 7,
        'time_limit': time_limit,
        'presets': 'fast_training',  # Faster but less accurate
        'eval_metric': 'MASE',
        'freq': 'D'  # Daily frequency
    }
    
    return train_pipeline(train_df, val_df, config)
"""
Prediction/Inference functions for retail sales forecasting.
Load trained models and generate forecasts.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from src.utils import create_output_dir, get_current_timestamp


# LOAD MODEL

def load_model(model_path):
    """
    Load a trained TimeSeriesPredictor model.
    
    Args:
        model_path: Path to the saved model directory
    
    Returns:
        Loaded TimeSeriesPredictor
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from: {model_path}")
    
    # Check if path exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    # Load the predictor
    predictor = TimeSeriesPredictor.load(model_path)
    logger.info("Model loaded successfully!")
    logger.info(f"Prediction length: {predictor.prediction_length}")
    logger.info(f"Target column: {predictor.target}")
    
    return predictor


# MAKE PREDICTIONS

def make_predictions(predictor, data):
    """
    Generate forecasts using the trained model.
    
    Args:
        predictor: Trained TimeSeriesPredictor
        data: Input data (pandas DataFrame or TimeSeriesDataFrame)
    
    Returns:
        DataFrame with predictions
    """
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Generating predictions")
    logger.info("="*60)
    
    # Convert to TimeSeriesDataFrame if needed
    if not isinstance(data, TimeSeriesDataFrame):
        from src.train import convert_to_timeseries_dataframe
        data = convert_to_timeseries_dataframe(data, freq=predictor.freq)
    
    logger.info(f"Predicting for {len(data.item_ids)} stores/items")
    
    # Make predictions
    # Quantiles are already configured in the trained model
    predictions = predictor.predict(data)
    
    logger.info("Predictions generated successfully!")
    logger.info(f"Prediction shape: {predictions.shape}")
    
    return predictions


# FORMAT PREDICTIONS

def format_predictions_for_export(predictions):
    """
    Convert predictions to a clean pandas DataFrame for export.
    
    Args:
        predictions: TimeSeriesDataFrame with predictions
    
    Returns:
        Clean pandas DataFrame with columns: store_id, date, predicted_sales, lower_bound, upper_bound
    """
    logger = logging.getLogger(__name__)
    logger.info("Formatting predictions for export...")
    
    # Convert to regular pandas DataFrame
    pred_list = []
    
    for item_id in predictions.item_ids:
        item_preds = predictions.loc[item_id]
        
        # Create a dataframe for this store
        store_df = pd.DataFrame({
            'store_id': item_id,
            'date': item_preds.index,
            'predicted_sales': item_preds['mean'].values if 'mean' in item_preds.columns else item_preds.iloc[:, 0].values
        })
        
        # Add quantiles if available
        if '0.1' in item_preds.columns:
            store_df['lower_bound'] = item_preds['0.1'].values
        if '0.9' in item_preds.columns:
            store_df['upper_bound'] = item_preds['0.9'].values
        
        pred_list.append(store_df)
    
    # Combine all stores
    final_df = pd.concat(pred_list, ignore_index=True)
    
    # Sort by store and date
    final_df = final_df.sort_values(['store_id', 'date']).reset_index(drop=True)
    
    logger.info(f"Formatted predictions: {final_df.shape}")
    
    return final_df


# SAVE PREDICTIONS

def save_predictions(predictions_df, filename=None, output_dir="outputs"):
    """
    Save predictions to CSV file.
    
    Args:
        predictions_df: DataFrame with predictions
        filename: Name of the output file (optional, auto-generated if None)
        output_dir: Output directory name
    
    Returns:
        Path to saved file
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    out_dir = create_output_dir(output_dir)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = get_current_timestamp()
        filename = f"predictions_{timestamp}.csv"
    
    # Full path
    filepath = Path(out_dir) / filename
    
    # Save to CSV
    predictions_df.to_csv(filepath, index=False)
    logger.info(f"Predictions saved to: {filepath}")
    
    return str(filepath)


# PREDICTION SUMMARY

def generate_prediction_summary(predictions_df):
    """
    Generate summary statistics for predictions.
    
    Args:
        predictions_df: DataFrame with predictions
    
    Returns:
        Dictionary with summary statistics
    """
    logger = logging.getLogger(__name__)
    
    summary = {
        'num_stores': predictions_df['store_id'].nunique(),
        'num_predictions': len(predictions_df),
        'date_range': {
            'start': str(predictions_df['date'].min()),
            'end': str(predictions_df['date'].max())
        },
        'predicted_sales': {
            'total': float(predictions_df['predicted_sales'].sum()),
            'mean': float(predictions_df['predicted_sales'].mean()),
            'median': float(predictions_df['predicted_sales'].median()),
            'min': float(predictions_df['predicted_sales'].min()),
            'max': float(predictions_df['predicted_sales'].max())
        }
    }
    
    logger.info("\n" + "="*60)
    logger.info("Prediction Summary")
    logger.info("="*60)
    logger.info(f"Number of stores: {summary['num_stores']}")
    logger.info(f"Total predictions: {summary['num_predictions']}")
    logger.info(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    logger.info(f"Total predicted sales: ${summary['predicted_sales']['total']:,.2f}")
    logger.info(f"Average predicted sales per day: ${summary['predicted_sales']['mean']:,.2f}")
    logger.info("="*60)
    
    return summary


# FULL PREDICTION PIPELINE

def predict_pipeline(model_path, data, save_output=True):
    """
    Full prediction pipeline.
    Load model, make predictions, format, and save.
    
    Args:
        model_path: Path to trained model
        data: Input data for prediction
        save_output: Whether to save predictions to CSV
    
    Returns:
        Tuple of (predictions_df, summary, save_path)
    """
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("STARTING PREDICTION PIPELINE")
    logger.info("="*60)
    
    # Step 1: Load model
    predictor = load_model(model_path)
    
    # Step 2: Make predictions
    predictions = make_predictions(predictor, data)
    
    # Step 3: Format predictions
    predictions_df = format_predictions_for_export(predictions)
    
    # Step 4: Generate summary
    summary = generate_prediction_summary(predictions_df)
    
    # Step 5: Save predictions
    save_path = None
    if save_output:
        save_path = save_predictions(predictions_df)
    
    logger.info("="*60)
    logger.info("PREDICTION PIPELINE COMPLETE")
    logger.info("="*60)
    
    return predictions_df, summary, save_path


# HELPER: PREDICT FUTURE (NO HISTORICAL DATA)

def predict_future(model_path, train_data, num_steps=7):
    """
    Predict future values beyond the training data.
    Useful for forecasting into the future when you don't have recent data.
    
    Args:
        model_path: Path to trained model
        train_data: Historical training data
        num_steps: Number of steps to forecast into future
    
    Returns:
        DataFrame with future predictions
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Forecasting {num_steps} steps into the future...")
    
    # Load model
    predictor = load_model(model_path)
    
    # Convert to TimeSeriesDataFrame
    if not isinstance(train_data, TimeSeriesDataFrame):
        from src.train import convert_to_timeseries_dataframe
        train_data = convert_to_timeseries_dataframe(train_data, freq=predictor.freq)
    
    # Make predictions (AutoGluon will automatically forecast future)
    predictions = predictor.predict(train_data)
    
    # Format and return
    predictions_df = format_predictions_for_export(predictions)
    
    logger.info(f"Future forecast complete! {len(predictions_df)} predictions generated.")
    
    return predictions_df


# HELPER: BATCH PREDICTION

def batch_predict_by_store(model_path, data, store_ids):
    """
    Make predictions for specific stores only.
    Useful for testing or selective forecasting.
    
    Args:
        model_path: Path to trained model
        data: Input data containing all stores
        store_ids: List of store IDs to predict for
    
    Returns:
        DataFrame with predictions for selected stores
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Making predictions for {len(store_ids)} specific stores...")
    
    # Filter data to selected stores
    store_ids = [str(s) for s in store_ids]  # Convert to strings
    filtered_data = data[data['item_id'].isin(store_ids)].copy()
    
    logger.info(f"Filtered data to {len(filtered_data)} rows")
    
    # Run prediction pipeline
    predictions_df, summary, save_path = predict_pipeline(
        model_path=model_path,
        data=filtered_data,
        save_output=True
    )
    
    return predictions_df
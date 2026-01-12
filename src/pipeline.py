import logging
from utils import setup_logging
from preprocess import preprocess_pipeline
from train import train_pipeline
from predict import predict_pipeline


# MAIN PIPELINE

def run_pipeline(num_stores=None, val_days=7, time_limit=600, presets='medium_quality'):
    """
    Run the complete forecasting pipeline.
    
    Args:
        num_stores: Number of stores (None = all stores)
        val_days: Validation days
        time_limit: Training time in seconds
        presets: 'fast_training', 'medium_quality', or 'high_quality'
    
    Returns:
        Dictionary with results
    """
    logger = setup_logging()
    
    logger.info("="*70)
    logger.info("STARTING RETAIL FORECASTING PIPELINE")
    logger.info("="*70)
    
    try:
        # Step 1: Preprocess
        logger.info("Step 1/3: Preprocessing data...")
        train_df, val_df = preprocess_pipeline(num_stores=num_stores, val_days=val_days)
        logger.info(" Preprocessing complete")
        
        # Step 2: Train
        logger.info("Step 2/3: Training model...")
        config = {
            'prediction_length': 7,
            'time_limit': time_limit,
            'presets': presets,
            'eval_metric': 'MASE',
            'freq': 'D'
        }
        predictor, metrics, model_path = train_pipeline(train_df, val_df, config)
        logger.info(" Training complete")
        
        # Step 3: Predict
        logger.info("Step 3/3: Generating predictions...")
        predictions_df, summary, save_path = predict_pipeline(model_path, val_df, save_output=True)
        logger.info(" Predictions complete")
        
        # Results
        results = {
            'model_path': model_path,
            'metrics': metrics,
            'predictions_path': save_path,
            'summary': summary
        }
        
        logger.info("="*70)
        logger.info("PIPELINE COMPLETE!")
        logger.info(f"Model: {model_path}")
        logger.info(f"Predictions: {save_path}")
        logger.info(f"MASE: {metrics.get('val_MASE', 'N/A')}")
        logger.info("="*70)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


# QUICK PIPELINE (FOR TESTING)

def quick_pipeline(num_stores=5, time_limit=300):
    """
    Quick test pipeline - 5 stores, 5 minutes.
    """
    logger = logging.getLogger(__name__)
    logger.info("Running QUICK PIPELINE...")
    
    return run_pipeline(
        num_stores=num_stores,
        val_days=7,
        time_limit=time_limit,
        presets='fast_training'
    )


# PRODUCTION PIPELINE

def production_pipeline(num_stores=None):
    """
    Production pipeline - all stores, high quality, 30 minutes.
    """
    logger = logging.getLogger(__name__)
    logger.info("Running PRODUCTION PIPELINE...")
    
    return run_pipeline(
        num_stores=num_stores,
        val_days=14,
        time_limit=1800,
        presets='high_quality'
    )


if __name__ == '__main__':
    # Default: run quick pipeline
    results = quick_pipeline(num_stores=3, time_limit=120)
    print("\n Pipeline finished successfully!")
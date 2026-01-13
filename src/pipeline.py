"""
Luigi orchestration for retail forecasting pipeline.
Simple task-based workflow orchestration with config support.
"""

import luigi
import logging
import pickle
from pathlib import Path
from utils import setup_logging, create_output_dir, load_config
from preprocess import preprocess_pipeline
from train import train_pipeline
from predict import predict_pipeline


# ============================================================================
# LUIGI TASKS
# ============================================================================

class PreprocessTask(luigi.Task):
    """
    Task 1: Preprocess the data.
    """
    
    def output(self):
        """Define output files for this task."""
        config = load_config()
        output_dir = Path(create_output_dir(config['output']['luigi_dir']))
        return {
            'train': luigi.LocalTarget(str(output_dir / 'train_data.pkl')),
            'val': luigi.LocalTarget(str(output_dir / 'val_data.pkl'))
        }
    
    def run(self):
        """Run preprocessing."""
        logger = setup_logging()
        config = load_config()
        
        logger.info("="*70)
        logger.info("LUIGI TASK: Preprocessing")
        logger.info("="*70)
        
        # Run preprocessing with config
        train_df, val_df = preprocess_pipeline(config=config)
        
        # Save outputs (BINARY mode for pickle)
        with open(self.output()['train'].path, 'wb') as f:
            pickle.dump(train_df, f)
        
        with open(self.output()['val'].path, 'wb') as f:
            pickle.dump(val_df, f)
        
        logger.info("✓ Preprocessing complete")


class TrainTask(luigi.Task):
    """
    Task 2: Train the model (SKIPPED if zero_shot=yes).
    Depends on PreprocessTask.
    """
    
    def requires(self):
        """This task requires PreprocessTask to complete first."""
        return PreprocessTask()
    
    def output(self):
        """Define output files for this task."""
        config = load_config()
        output_dir = Path(create_output_dir(config['output']['luigi_dir']))
        return {
            'model_path': luigi.LocalTarget(str(output_dir / 'model_path.txt')),
            'metrics': luigi.LocalTarget(str(output_dir / 'metrics.pkl'))
        }
    
    def run(self):
        """Run training."""
        logger = setup_logging()
        config = load_config()
        
        logger.info("="*70)
        logger.info("LUIGI TASK: Training")
        logger.info("="*70)
        
        # Load preprocessed data (BINARY mode for pickle)
        with open(self.input()['train'].path, 'rb') as f:
            train_df = pickle.load(f)
        
        with open(self.input()['val'].path, 'rb') as f:
            val_df = pickle.load(f)
        
        # Build training config from YAML
        train_config = {
            'prediction_length': config['forecast']['horizon'],
            'time_limit': config['training']['time_limit'],
            'presets': config['training']['presets'],
            'eval_metric': config['model']['eval_metric'],
            'freq': config['forecast']['frequency'],
            'zero_shot': config['model']['zero_shot'],
            'zero_shot_model': config['model'].get('zero_shot_model', 'chronos-t5-base')
        }
        
        # Train model
        predictor, metrics, model_path = train_pipeline(
            train_df=train_df,
            val_df=val_df,
            config=train_config
        )
        
        # Save outputs
        with self.output()['model_path'].open('w') as f:
            f.write(model_path)
        
        with open(self.output()['metrics'].path, 'wb') as f:
            pickle.dump(metrics, f)
        
        logger.info("✓ Training complete")


class PredictTask(luigi.Task):
    """
    Task 3: Generate predictions.
    Depends on TrainTask OR PreprocessTask (if zero-shot).
    """
    
    def requires(self):
        """This task requires TrainTask OR PreprocessTask (if zero-shot)."""
        config = load_config()
        zero_shot = config['model']['zero_shot']
        
        if zero_shot:
            # Zero-shot: skip training, only need preprocessing
            return PreprocessTask()
        else:
            # Training mode: need trained model
            return TrainTask()
    
    def output(self):
        """Define output files for this task."""
        config = load_config()
        output_dir = Path(create_output_dir(config['output']['luigi_dir']))
        return luigi.LocalTarget(str(output_dir / 'predictions_path.txt'))
    
    def run(self):
        """Run prediction."""
        logger = setup_logging()
        config = load_config()
    
        logger.info("="*70)
        logger.info("LUIGI TASK: Prediction")
        logger.info("="*70)
    
        zero_shot = config['model']['zero_shot']
    
        if zero_shot:
            # Zero-shot path: load data and use pre-trained model
            preprocess_task = PreprocessTask()
            with open(preprocess_task.output()['val'].path, 'rb') as f:
                val_df = pickle.load(f)
        
            # Use zero-shot model (model_path will be placeholder)
            model_path = f"zero_shot_{config['model']['zero_shot_model']}"
        
        else:
            # Training path: load model path
            with self.input()['model_path'].open('r') as f:
                model_path = f.read().strip()
        
            # Load validation data
            preprocess_task = PreprocessTask()
            with open(preprocess_task.output()['val'].path, 'rb') as f:
                val_df = pickle.load(f)
    
        # Generate predictions (PASS zero_shot FLAG)
        predictions_df, summary, save_path = predict_pipeline(
            model_path=model_path,
            data=val_df,
            save_output=True,
            zero_shot=zero_shot  # ADD THIS
        )
    
        # Save output path
        with self.output().open('w') as f:
            f.write(save_path)
    
        mode_str = "ZERO-SHOT" if zero_shot else "TRAINED"
        logger.info("✓ Predictions complete")
        logger.info("="*70)
        logger.info(f"PIPELINE COMPLETE ({mode_str} MODE)!")
        logger.info(f"Predictions saved to: {save_path}")
        logger.info("="*70)


# ============================================================================
# WRAPPER TASK (MAIN PIPELINE)
# ============================================================================

class ForecastingPipeline(luigi.Task):
    """
    Main pipeline task that runs all steps based on config.
    """
    
    def requires(self):
        """Require the final task (which will trigger all dependencies)."""
        return PredictTask()
    
    def output(self):
        """Pipeline completion marker."""
        config = load_config()
        output_dir = Path(create_output_dir(config['output']['luigi_dir']))
        return luigi.LocalTarget(str(output_dir / 'pipeline_complete.txt'))
    
    def run(self):
        """Mark pipeline as complete."""
        logger = setup_logging()
        config = load_config()
        
        # Load results
        with self.input().open('r') as f:
            predictions_path = f.read().strip()
        
        # Write completion marker
        mode_str = "ZERO-SHOT" if config['model']['zero_shot'] else "TRAINED"
        with self.output().open('w') as f:
            f.write(f"Pipeline completed successfully! ({mode_str} mode)\n")
            f.write(f"Predictions: {predictions_path}\n")
        
        logger.info("\n" + "="*70)
        logger.info(f"✓✓✓ ENTIRE PIPELINE COMPLETED SUCCESSFULLY ({mode_str}) ✓✓✓")
        logger.info(f"Predictions: {predictions_path}")
        logger.info("="*70 + "\n")
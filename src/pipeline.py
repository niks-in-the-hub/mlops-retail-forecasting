import luigi
import logging
import pickle
from pathlib import Path
from utils import setup_logging, create_output_dir, get_current_timestamp
from preprocess import preprocess_pipeline
from train import train_pipeline
from predict import predict_pipeline


# LUIGI TASKS

class PreprocessTask(luigi.Task):
    """
    Task 1: Preprocess the data.
    """
    num_stores = luigi.IntParameter(default=10)
    val_days = luigi.IntParameter(default=7)
    
    def output(self):
        """Define output files for this task."""
        output_dir = Path(create_output_dir("luigi_outputs"))
        return {
            'train': luigi.LocalTarget(str(output_dir / 'train_data.pkl')),
            'val': luigi.LocalTarget(str(output_dir / 'val_data.pkl'))
        }
    
    def run(self):
        """Run preprocessing."""
        logger = setup_logging()
        logger.info("="*70)
        logger.info("LUIGI TASK: Preprocessing")
        logger.info("="*70)
        
        # Run preprocessing
        train_df, val_df = preprocess_pipeline(
            num_stores=self.num_stores,
            val_days=self.val_days
        )
        
        # Save outputs
        with self.output()['train'].open('w') as f:
            pickle.dump(train_df, f)
        
        with self.output()['val'].open('w') as f:
            pickle.dump(val_df, f)
        
        logger.info("✓ Preprocessing complete")


class TrainTask(luigi.Task):
    """
    Task 2: Train the model.
    Depends on PreprocessTask.
    """
    num_stores = luigi.IntParameter(default=10)
    val_days = luigi.IntParameter(default=7)
    time_limit = luigi.IntParameter(default=600)
    presets = luigi.Parameter(default='medium_quality')
    
    def requires(self):
        """This task requires PreprocessTask to complete first."""
        return PreprocessTask(num_stores=self.num_stores, val_days=self.val_days)
    
    def output(self):
        """Define output files for this task."""
        output_dir = Path(create_output_dir("luigi_outputs"))
        return {
            'model_path': luigi.LocalTarget(str(output_dir / 'model_path.txt')),
            'metrics': luigi.LocalTarget(str(output_dir / 'metrics.pkl'))
        }
    
    def run(self):
        """Run training."""
        logger = setup_logging()
        logger.info("="*70)
        logger.info("LUIGI TASK: Training")
        logger.info("="*70)
        
        # Load preprocessed data
        with self.input()['train'].open('r') as f:
            train_df = pickle.load(f)
        
        with self.input()['val'].open('r') as f:
            val_df = pickle.load(f)
        
        # Train model
        config = {
            'prediction_length': 7,
            'time_limit': self.time_limit,
            'presets': self.presets,
            'eval_metric': 'MASE',
            'freq': 'D'
        }
        
        predictor, metrics, model_path = train_pipeline(
            train_df=train_df,
            val_df=val_df,
            config=config
        )
        
        # Save outputs
        with self.output()['model_path'].open('w') as f:
            f.write(model_path)
        
        with self.output()['metrics'].open('w') as f:
            pickle.dump(metrics, f)
        
        logger.info("✓ Training complete")


class PredictTask(luigi.Task):
    """
    Task 3: Generate predictions.
    Depends on TrainTask.
    """
    num_stores = luigi.IntParameter(default=10)
    val_days = luigi.IntParameter(default=7)
    time_limit = luigi.IntParameter(default=600)
    presets = luigi.Parameter(default='medium_quality')
    
    def requires(self):
        """This task requires TrainTask to complete first."""
        return TrainTask(
            num_stores=self.num_stores,
            val_days=self.val_days,
            time_limit=self.time_limit,
            presets=self.presets
        )
    
    def output(self):
        """Define output files for this task."""
        output_dir = Path(create_output_dir("luigi_outputs"))
        return luigi.LocalTarget(str(output_dir / 'predictions_path.txt'))
    
    def run(self):
        """Run prediction."""
        logger = setup_logging()
        logger.info("="*70)
        logger.info("LUIGI TASK: Prediction")
        logger.info("="*70)
        
        # Load model path and validation data
        with self.input()['model_path'].open('r') as f:
            model_path = f.read().strip()
        
        # Load validation data from PreprocessTask
        preprocess_task = PreprocessTask(num_stores=self.num_stores, val_days=self.val_days)
        with preprocess_task.output()['val'].open('r') as f:
            val_df = pickle.load(f)
        
        # Generate predictions
        predictions_df, summary, save_path = predict_pipeline(
            model_path=model_path,
            data=val_df,
            save_output=True
        )
        
        # Save output path
        with self.output().open('w') as f:
            f.write(save_path)
        
        logger.info("✓ Predictions complete")
        logger.info("="*70)
        logger.info("PIPELINE COMPLETE!")
        logger.info(f"Predictions saved to: {save_path}")
        logger.info("="*70)


# MAIN PIPELINE

class ForecastingPipeline(luigi.Task):
    """
    Main pipeline task that runs all steps.
    This is what you run to execute the entire pipeline.
    """
    num_stores = luigi.IntParameter(default=10)
    val_days = luigi.IntParameter(default=7)
    time_limit = luigi.IntParameter(default=600)
    presets = luigi.Parameter(default='medium_quality')
    
    def requires(self):
        """Require the final task (which will trigger all dependencies)."""
        return PredictTask(
            num_stores=self.num_stores,
            val_days=self.val_days,
            time_limit=self.time_limit,
            presets=self.presets
        )
    
    def output(self):
        """Pipeline completion marker."""
        output_dir = Path(create_output_dir("luigi_outputs"))
        return luigi.LocalTarget(str(output_dir / 'pipeline_complete.txt'))
    
    def run(self):
        """Mark pipeline as complete."""
        logger = setup_logging()
        
        # Load results
        with self.input().open('r') as f:
            predictions_path = f.read().strip()
        
        # Write completion marker
        with self.output().open('w') as f:
            f.write(f"Pipeline completed successfully!\nPredictions: {predictions_path}\n")
        
        logger.info("\n" + "="*70)
        logger.info("✓✓✓ ENTIRE PIPELINE COMPLETED SUCCESSFULLY ✓✓✓")
        logger.info(f"Predictions: {predictions_path}")
        logger.info("="*70 + "\n")


# CONVENIENCE TASKS

class QuickPipeline(luigi.Task):
    """Quick test pipeline - 3 stores, 2 minutes."""
    
    def requires(self):
        return ForecastingPipeline(
            num_stores=3,
            val_days=7,
            time_limit=120,
            presets='fast_training'
        )
    
    def output(self):
        output_dir = Path(create_output_dir("luigi_outputs"))
        return luigi.LocalTarget(str(output_dir / 'quick_pipeline_complete.txt'))
    
    def run(self):
        with self.output().open('w') as f:
            f.write("Quick pipeline completed!\n")


class ProductionPipeline(luigi.Task):
    """Production pipeline - all stores, 30 minutes."""
    
    num_stores = luigi.IntParameter(default=None)
    
    def requires(self):
        return ForecastingPipeline(
            num_stores=self.num_stores if self.num_stores else 1000,
            val_days=14,
            time_limit=1800,
            presets='high_quality'
        )
    
    def output(self):
        output_dir = Path(create_output_dir("luigi_outputs"))
        return luigi.LocalTarget(str(output_dir / 'production_pipeline_complete.txt'))
    
    def run(self):
        with self.output().open('w') as f:
            f.write("Production pipeline completed!\n")
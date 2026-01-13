"""
Main entry point for the retail forecasting pipeline with Luigi orchestration.
"""

import sys
import luigi
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline import QuickPipeline, ProductionPipeline, ForecastingPipeline


def print_usage():
    """Print usage instructions."""
    print("\n" + "="*70)
    print("RETAIL FORECASTING PIPELINE - LUIGI ORCHESTRATION")
    print("="*70)
    print("\nAvailable modes:")
    print("  python run.py quick        - Quick test (3 stores, 2 mins)")
    print("  python run.py medium       - Medium run (10 stores, 10 mins)")
    print("  python run.py production   - Production (all stores, 30 mins)")
    print("\nWhat Luigi does:")
    print("  - Tracks task dependencies")
    print("  - Skips completed tasks (resume capability)")
    print("  - Logs execution flow")
    print("  - Creates task dependency graph")
    print("="*70 + "\n")


def main():
    """Run the pipeline using Luigi."""
    
    # Check for arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print_usage()
        print("No mode specified. Running QUICK mode by default...\n")
        mode = 'quick'
    
    if mode not in ['quick', 'medium', 'production']:
        print(f"\n Error: Unknown mode '{mode}'")
        print_usage()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("RETAIL FORECASTING PIPELINE WITH LUIGI")
    print("="*70)
    
    # Run the appropriate Luigi task
    if mode == 'quick':
        print("Mode: QUICK TEST")
        print("  - Stores: 3")
        print("  - Time limit: 2 minutes")
        print("  - Preset: fast_training")
        print("="*70 + "\n")
        
        success = luigi.build([QuickPipeline()], local_scheduler=True)
    
    elif mode == 'medium':
        print("Mode: MEDIUM RUN")
        print("  - Stores: 10")
        print("  - Time limit: 10 minutes")
        print("  - Preset: medium_quality")
        print("="*70 + "\n")
        
        success = luigi.build(
            [ForecastingPipeline(num_stores=10, time_limit=600, presets='medium_quality')],
            local_scheduler=True
        )
    
    elif mode == 'production':
        print("Mode: PRODUCTION")
        print("  - Stores: ALL")
        print("  - Time limit: 30 minutes")
        print("  - Preset: high_quality")
        print("="*70 + "\n")
        
        success = luigi.build([ProductionPipeline()], local_scheduler=True)
    
    # Check results
    if success:
        print("\n" + "="*70)
        print(" PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nCheck outputs:")
        print("  - Predictions: outputs/predictions_*.csv")
        print("  - Models: models/")
        print("  - Luigi outputs: luigi_outputs/")
        print("\nView MLflow experiments:")
        print("  mlflow ui")
        print("="*70 + "\n")
        return 0
    else:
        print("\n" + "="*70)
        print(" PIPELINE FAILED")
        print("="*70)
        print("\nCheck the logs above for details.")
        print("="*70 + "\n")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
"""
Main entry point for the retail forecasting pipeline.
Simple command-line interface to run different pipeline modes.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline import quick_pipeline, production_pipeline, run_pipeline
from utils import setup_logging


def print_usage():
    """Print usage instructions."""
    print("\n" + "="*70)
    print("RETAIL FORECASTING PIPELINE - USAGE")
    print("="*70)
    print("\nAvailable modes:")
    print("  python run.py quick        - Quick test (3 stores, 2 mins)")
    print("  python run.py medium       - Medium run (10 stores, 10 mins)")
    print("  python run.py production   - Production (all stores, 30 mins)")
    print("\nExamples:")
    print("  python run.py quick")
    print("  python run.py medium")
    print("="*70 + "\n")


def main():
    """Run the pipeline based on command line arguments."""
    
    logger = setup_logging()
    
    # Check for arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print_usage()
        print("No mode specified. Running QUICK mode by default...\n")
        mode = 'quick'
    
    # Invalid mode check
    if mode not in ['quick', 'medium', 'production']:
        print(f"\n Error: Unknown mode '{mode}'")
        print_usage()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("RETAIL FORECASTING PIPELINE")
    print("="*70)
    
    try:
        if mode == 'quick':
            print("Mode: QUICK TEST")
            print("  - Stores: 3")
            print("  - Time limit: 2 minutes")
            print("  - Preset: fast_training")
            print("="*70 + "\n")
            results = quick_pipeline(num_stores=3, time_limit=120)
        
        elif mode == 'medium':
            print("Mode: MEDIUM RUN")
            print("  - Stores: 10")
            print("  - Time limit: 10 minutes")
            print("  - Preset: medium_quality")
            print("="*70 + "\n")
            results = run_pipeline(num_stores=10, time_limit=600, presets='medium_quality')
        
        elif mode == 'production':
            print("Mode: PRODUCTION")
            print("  - Stores: ALL")
            print("  - Time limit: 30 minutes")
            print("  - Preset: high_quality")
            print("="*70 + "\n")
            results = production_pipeline()
        
        # Print final results
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n Model saved to:")
        print(f"   {results['model_path']}")
        print(f"\n Predictions saved to:")
        print(f"   {results['predictions_path']}")
        
        print(f"\n Performance Metrics:")
        for key, value in results['metrics'].items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("\n1. View predictions:")
        print(f"   open {results['predictions_path']}")
        print("\n2. View MLflow experiments:")
        print("   mlflow ui")
        print("   Then open: http://localhost:5000")
        print("\n3. Check output files:")
        print("   ls -la outputs/")
        print("   ls -la models/")
        print("="*70 + "\n")
        
        return 0
    
    except Exception as e:
        print("\n" + "="*70)
        print("PIPELINE FAILED")
        print("="*70)
        print(f"\nError: {str(e)}")
        print("\nPlease check the logs above for details.")
        print("="*70 + "\n")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
"""
Test script for the Metaflow retail forecasting pipeline.
Run this to verify everything works correctly.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging
import logging


def test_imports():
    """
    Test that all required modules can be imported.
    """
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*70)
    print("TESTING IMPORTS")
    print("="*70)
    
    modules = {
        'metaflow': 'Metaflow',
        'mlflow': 'MLflow',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'autogluon.timeseries': 'AutoGluon TimeSeries',
        'src.utils': 'Utils Module',
        'src.preprocess': 'Preprocess Module',
        'src.train': 'Train Module',
        'src.predict': 'Predict Module',
        'src.pipeline': 'Pipeline Module'
    }
    
    all_passed = True
    
    for module_name, display_name in modules.items():
        try:
            __import__(module_name)
            print(f"✓ {display_name:.<50} OK")
        except Exception as e:
            print(f"✗ {display_name:.<50} FAILED: {str(e)}")
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("✓ All imports successful!\n")
    else:
        print("✗ Some imports failed. Please check your installation.\n")
    
    return all_passed


def test_data_availability():
    """
    Test that required data files exist.
    """
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*70)
    print("TESTING DATA AVAILABILITY")
    print("="*70)
    
    from src.utils import get_data_path
    
    data_files = ['train.csv', 'store.csv']
    all_exist = True
    
    for filename in data_files:
        filepath = get_data_path(filename)
        exists = Path(filepath).exists()
        status = "✓ EXISTS" if exists else "✗ MISSING"
        print(f"{filename:.<50} {status}")
        
        if exists:
            # Show file size
            size_mb = Path(filepath).stat().st_size / (1024 * 1024)
            print(f"  Size: {size_mb:.2f} MB")
        else:
            all_exist = False
    
    print("="*70)
    
    if all_exist:
        print("✓ All data files present!\n")
    else:
        print("✗ Some data files missing. Please check your data/ folder.\n")
    
    return all_exist


def test_directory_structure():
    """
    Test that all required directories exist.
    """
    print("\n" + "="*70)
    print("TESTING DIRECTORY STRUCTURE")
    print("="*70)
    
    required_dirs = {
        'data': 'Data directory',
        'src': 'Source code directory',
        'models': 'Models directory (will be created)',
        'outputs': 'Outputs directory (will be created)',
        'mlruns': 'MLflow runs directory (will be created)'
    }
    
    project_root = Path(__file__).parent
    
    for dir_name, description in required_dirs.items():
        dir_path = project_root / dir_name
        exists = dir_path.exists()
        
        if dir_name in ['models', 'outputs', 'mlruns']:
            # These are created automatically
            status = "✓ EXISTS" if exists else "○ WILL BE CREATED"
        else:
            # These should exist
            status = "✓ EXISTS" if exists else "✗ MISSING"
        
        print(f"{description:.<50} {status}")
    
    print("="*70 + "\n")


def test_metaflow_setup():
    """
    Test that Metaflow can see the flows.
    """
    print("\n" + "="*70)
    print("TESTING METAFLOW SETUP")
    print("="*70)
    
    try:
        from src.pipeline import QuickFlow, ForecastingFlow, ProductionFlow
        print("✓ QuickFlow imported successfully")
        print("✓ ForecastingFlow imported successfully")
        print("✓ ProductionFlow imported successfully")
        print("="*70 + "\n")
        return True
    except Exception as e:
        print(f"✗ Failed to import flows: {str(e)}")
        print("="*70 + "\n")
        return False


def run_all_tests():
    """
    Run all tests and provide a summary.
    """
    logger = setup_logging()
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " RETAIL FORECASTING PIPELINE - TEST SUITE ".center(68) + "║")
    print("╚" + "="*68 + "╝")
    
    # Run tests
    test_directory_structure()
    imports_ok = test_imports()
    data_ok = test_data_availability()
    metaflow_ok = test_metaflow_setup()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Imports:        {'✓ PASSED' if imports_ok else '✗ FAILED'}")
    print(f"Data Files:     {'✓ PASSED' if data_ok else '✗ FAILED'}")
    print(f"Metaflow Setup: {'✓ PASSED' if metaflow_ok else '✗ FAILED'}")
    print("="*70)
    
    if imports_ok and data_ok and metaflow_ok:
        print("\n✓ All tests passed! Your pipeline is ready to run.")
        print("\n" + "="*70)
        print("NEXT STEPS - RUN THE PIPELINE")
        print("="*70)
        print("\n1. QUICK TEST (3 stores, 2 minutes):")
        print("   python src/pipeline.py QuickFlow run --num_stores 3 --time_limit 120")
        
        print("\n2. MEDIUM TEST (5 stores, 5 minutes):")
        print("   python src/pipeline.py QuickFlow run --num_stores 5 --time_limit 300")
        
        print("\n3. STANDARD RUN (10 stores, 10 minutes):")
        print("   python src/pipeline.py run --num_stores 10")
        
        print("\n4. PRODUCTION RUN (all stores, 30 minutes):")
        print("   python src/pipeline.py ProductionFlow run")
        
        print("\n" + "="*70)
        print("AFTER RUNNING - VIEW RESULTS")
        print("="*70)
        print("\n1. View MLflow experiment tracking:")
        print("   mlflow ui")
        print("   Then open: http://localhost:5000")
        
        print("\n2. Check predictions:")
        print("   ls -la outputs/")
        
        print("\n3. Check trained models:")
        print("   ls -la models/")
        
        print("\n4. View Metaflow run history:")
        print("   cd src && python pipeline.py QuickFlow show")
        
        print("\n" + "="*70)
        print("TIP: Start with the QUICK TEST to verify everything works!")
        print("="*70 + "\n")
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
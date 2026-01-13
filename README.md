# MLOps Retail Forecasting

A production-ready MLOps pipeline for retail demand forecasting using AutoGluon TimeSeries with MLflow tracking and Luigi orchestration.

## Features

- **Zero-Shot Forecasting**: Instant predictions using pre-trained Chronos models (no training required)
- **Automated ML Pipeline**: End-to-end forecasting with AutoGluon TimeSeries
- **MLOps Integration**: Experiment tracking with MLflow, workflow orchestration with Luigi
- **Production Ready**: Configurable preprocessing, model training, and prediction pipelines
- **Multiple Modes**: Quick prototyping, medium-scale, and production deployment options
- **YAML Configuration**: Centralized configuration management with mode-specific overrides
- **Comprehensive Logging**: Structured logging for monitoring and debugging

## Prerequisites

- Python 3.10+
- Virtual environment (recommended)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/niks-in-the-hub/mlops-retail-forecasting.git
   cd mlops-retail-forecasting
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

Run the pipeline with default settings:
```bash
python run.py quick
```

### Available Modes

Run with different predefined configurations:
```bash
python run.py quick      # Fast test (3 stores, 2 mins)
python run.py medium     # Medium run (10 stores, 10 mins)  
python run.py production # Full production (all stores, 30 mins)
```

### Zero-Shot Mode

Enable instant forecasting without training by editing `config.yaml`:
```yaml
model:
  zero_shot: yes
  zero_shot_model: chronos-t5-base  # Options: tiny, small, base, large
```

### Configuration

All settings are managed through `config.yaml`:
- **Pipeline modes**: Switch between quick/medium/production
- **Zero-shot settings**: Enable pre-trained models
- **Data parameters**: Configure target variables and forecast horizon
- **Training settings**: Adjust time limits and model quality presets
- **Output paths**: Customize where results are saved

## Project Structure

```
mlops-retail-forecasting/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── run.py                   # Main entry point
├── config.yaml              # Pipeline configuration
utilities
├── .gitignore              # Git ignore rules
│
├── src/                    # Source code
│   ├── pipeline.py         # Luigi task orchestration
│   ├── preprocess.py       # Data preprocessing
│   ├── train.py           # Model training and zero-shot
│   ├── predict.py         # Prediction generation
│   └── utils.py           # Utility functions
│
├── notebooks/              # Jupyter notebooks
│   └── zero_shot_comparison.ipynb  # Zero-shot analysis
│
├── data/                   # Data directory
│   ├── train.csv          # Training data
│   └── store.csv          # Store metadata
│
├── models/                # Trained models
├── AutogluonModels/       # AutoGluon model artifacts
├── outputs/               # Pipeline outputs
├── luigi_outputs/         # Luigi task outputs
├── mlruns/               # MLflow experiment tracking
└── venv/                 # Virtual environment
```

## Configuration

The pipeline supports various configuration options:

- **Data paths**: Configure input/output data locations
- **Model parameters**: Adjust AutoGluon training settings
- **MLflow tracking**: Set experiment names and tracking URIs
- **Logging levels**: Control verbosity of pipeline logs

## MLflow Tracking

View experiment results and model metrics:
```bash
mlflow ui
```

Navigate to `http://localhost:5000` to access the MLflow UI.


## Model Performance

The pipeline uses AutoGluon TimeSeries which automatically:
- Selects optimal forecasting models
- Performs hyperparameter tuning
- Provides ensemble predictions
- Generates prediction intervals


## Pipeline Workflow

1. **Data Preprocessing** - Clean and prepare time series data
2. **Model Training** - Train AutoGluon forecasting models
3. **Model Evaluation** - Validate model performance
4. **Prediction Generation** - Generate forecasts
5. **Results Logging** - Track experiments with MLflow
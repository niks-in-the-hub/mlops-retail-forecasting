# MLOps Retail Forecasting

A production-ready MLOps pipeline for retail demand forecasting using AutoGluon TimeSeries with MLflow tracking and Luigi orchestration.

## Features

- **Automated ML Pipeline**: End-to-end forecasting with AutoGluon TimeSeries
- **MLOps Integration**: Experiment tracking with MLflow, workflow orchestration with Luigi
- **Production Ready**: Configurable preprocessing, model training, and prediction pipelines
- **Multiple Modes**: Quick prototyping and production deployment options
- **Comprehensive Logging**: Structured logging for monitoring and debugging

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
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

### Production Pipeline

Run the full production pipeline:
```bash
python run.py production
```

### Custom Pipeline

Run with specific configuration:
```bash
python run.py pipeline --config config.yaml
```

### Available Commands

- `quick` - Fast pipeline for prototyping
- `production` - Full production pipeline with all features
- `pipeline` - Custom pipeline with configuration file

## Project Structure

```
mlops-retail-forecasting/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── run.py                   # Main entry point
├── checking.py              # Validation and testing utilities
├── .gitignore              # Git ignore rules
│
├── src/                    # Source code
│   ├── pipeline.py         # Pipeline orchestration
│   ├── preprocess.py       # Data preprocessing
│   ├── train.py           # Model training
│   ├── predict.py         # Prediction generation
│   └── utils.py           # Utility functions
│
├── data/                   # Data directory
│   ├── raw/               # Raw input data
│   └── processed/         # Processed data
│
├── models/                # Trained models
├── AutogluonModels/       # AutoGluon model artifacts
├── outputs/               # Pipeline outputs
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

## Testing

Run validation checks:
```bash
python checking.py
```

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
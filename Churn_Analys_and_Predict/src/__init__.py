"""
Package: src
Version: 1.0.0

Modules:
- utils: Utility functions (Logger, IO, Metrics, Config)
- preprocessing: Data preprocessing pipeline
- modeling: Model training and optimization
- visualization: Data and results visualization
- pipeline: Main pipeline orchestrator
"""

__version__ = "1.0.0"
__author__ = "Vi"

from .Utils import (
    ConfigLoader,
    Logger,
    IOHandler,
    MetricsCalculator,
    set_random_seed,   # <--- Thêm cái này để main.py dùng
    get_timestamp      # <--- Thêm cái này để main.py dùng
)
from .Preprocessing import DataPreprocessor
from .Modeling import ModelTrainer
from .Visualization import DataVisualizer
from .RunPipeline import Pipeline  # Đã sửa 'RunPipeline' thành 'pipeline' cho chuẩn

# --- EXPORT TO OUTSIDE ---
__all__ = [
    'ConfigLoader',
    'Logger',
    'IOHandler',
    'MetricsCalculator',
    'set_random_seed',
    'get_timestamp',
    'DataPreprocessor',
    'ModelTrainer',
    'DataVisualizer',
    'Pipeline',
]
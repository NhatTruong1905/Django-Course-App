"""
Utility functions cho logging, I/O operations, và metrics
"""
import os, sys, yaml, json, logging, joblib, pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# ===== 1. ConfigLoader =====
class ConfigLoader:
    """
    Class để load và parse file YAML configuration
    Sử dụng:
        config = ConfigLoader.load_config('config/config.yaml')
        print(config['data']['raw_path'])
    """

    @staticmethod
    def load_config(config_path: str = "config/config.yaml") -> Dict:
        """
        Load configuration từ YAML file

        Args:
            config_path (str): Đường dẫn đến file config YAML
        Returns:
            Dict: Dictionary chứa tất cả configurations
        Raises:
            FileNotFoundError: Nếu file config không tồn tại
            yaml.YAMLError: Nếu file YAML không đúng format
        """
        try:
            # Kiểm tra file có tồn tại không
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file không tìm thấy: {config_path}")

            # Đọc và parse YAML file
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Kiểm tra config có hợp lệ không
            if config is None:
                raise ValueError(f"Config file rỗng hoặc không hợp lệ: {config_path}")
            return config
        except FileNotFoundError as e:
            print(f"Lỗi: {e}")
            raise
        except yaml.YAMLError as e:
            print(f" khi parse YAML file: {e}")
            raise
        except Exception as e:
            print(f"Lỗi không xác định: {e}")
            raise


# ===== 2. Logger =====
class Logger:
    """
    Class quản lý logging cho toàn bộ project

    Features:
    - Tạo logger với cả console handler và file handler
    - Lưu logs vào file theo ngày
    - Singleton pattern (mỗi name chỉ có 1 logger instance)

    Sử dụng:
        logger = Logger.get_logger('preprocessing')
        logger.info("Bắt đầu preprocessing...")
        logger.error("This is an error message")
    """

    # Dictionary lưu trữ tất cả logger instances (Singleton pattern)
    _loggers = {}

    @classmethod
    def get_logger(cls, name: str, log_dir: str = "artifacts/logs",
                   level: str = "INFO") -> logging.Logger:
        """
        Tạo hoặc lấy logger instance đã tồn tại

        Args:
            name (str): Tên của logger (vd: 'preprocessing', 'modeling')
            log_dir (str): Thư mục lưu log files
            level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        Returns:
            logging.Logger: Logger instance
        """

        # Nếu logger với name này đã tồn tại, trả về luôn
        if name in cls._loggers:
            return cls._loggers[name]

        # Tạo thư mục logs nếu chưa có
        os.makedirs(log_dir, exist_ok=True)

        # Tạo logger instance mới
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        # Tránh duplicate handlers nếu logger đã có handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # Định nghĩa format cho log messages
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # === Console Handler (hiển thị trên terminal) ===
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)  # Console chỉ show INFO trở lên
        logger.addHandler(console_handler)

        # === File Handler (lưu vào file) ===
        # Tạo log file với tên: {name}_{date}.log
        log_file = os.path.join(
            log_dir,
            f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  # File lưu tất cả levels
        logger.addHandler(file_handler)

        # Lưu logger vào dictionary để tái sử dụng
        cls._loggers[name] = logger

        return logger


# ===== 3. IOHandler =====
class IOHandler:
    """
    Class xử lý tất cả các thao tác đọc/ghi file

    Hỗ trợ:
    - Đọc/ghi DataFrame (CSV, Excel, JSON, Parquet)
    - Lưu/load models (Joblib, Pickle)
    - Đọc/ghi JSON files
    """

    @staticmethod
    def read_data(file_path: str, **kwargs) -> pd.DataFrame:
        """
        Đọc dữ liệu từ nhiều định dạng file khác nhau

        Args:
            file_path (str): Đường dẫn đến file
            **kwargs: Tham số bổ sung cho pandas read functions
                     Ví dụ: sheet_name='E Comm' cho Excel
        Returns:
            pd.DataFrame: DataFrame chứa dữ liệu

        Raises:
            ValueError: Nếu định dạng file không được hỗ trợ
            IOError: Nếu có lỗi khi đọc file

        Example:
            # Đọc CSV:  df = IOHandler.read_data('data.csv')
            # Đọc Excel với sheet cụ thể:  df = IOHandler.read_data('data.xlsx', sheet_name='Sheet1'
            # Đọc JSON:  df = IOHandler.read_data('data.json')
        """

        # Lấy extension của file
        file_ext = Path(file_path).suffix.lower()

        try:
            # Đọc theo từng loại file
            if file_ext == '.csv':
                return pd.read_csv(file_path, **kwargs)

            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path, **kwargs)

            elif file_ext == '.json':
                return pd.read_json(file_path, **kwargs)

            elif file_ext == '.parquet':
                return pd.read_parquet(file_path, **kwargs)

            else:
                raise ValueError(
                    f"Định dạng file không phù hợp : {file_ext}\n"
                    f"Hỗ trợ: .csv, .xlsx, .xls, .json, .parquet"
                )

        except Exception as e:
            raise IOError(f"Lỗi khi đọc file {file_path}: {str(e)}")

    @staticmethod
    def save_data(df: pd.DataFrame, file_path: str, **kwargs) -> None:
        """
        Lưu DataFrame ra file

        Args:
            df (pd.DataFrame): DataFrame cần lưu
            file_path (str): Đường dẫn file output
            **kwargs: Tham số bổ sung cho pandas write functions

        Raises:
            ValueError: Nếu định dạng file không được hỗ trợ
            IOError: Nếu có lỗi khi ghi file
        Example:
            df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            IOHandler.save_data(df, 'output.csv')
            IOHandler.save_data(df, 'output.xlsx')
        """
        # Tạo thư mục cha nếu chưa tồn tại
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Lấy extension của file
        file_ext = Path(file_path).suffix.lower()

        try:
            # Lưu theo từng loại file
            if file_ext == '.csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif file_ext in ['.xlsx', '.xls']:
                df.to_excel(file_path, index=False, **kwargs)
            elif file_ext == '.json':
                df.to_json(file_path, **kwargs)
            elif file_ext == '.parquet':
                df.to_parquet(file_path, index=False, **kwargs)
            else:
                raise ValueError(
                    f"Định dạng file không hỗ trợ: {file_ext}\n"
                    f"Hỗ trợ: .csv, .xlsx, .xls, .json, .parquet"
                )

        except Exception as e:
            raise IOError(f"Lỗi khi lưu file {file_path}: {str(e)}")

    @staticmethod
    def save_model(model: Any, file_path: str, method: str = "joblib") -> None:
        """
        Lưu machine learning model ra file
        Args:
            model: Model object cần lưu
            file_path (str): Đường dẫn file output
            method (str): Phương thức lưu ('joblib' hoặc 'pickle')
        Raises:
            ValueError: Nếu method không hợp lệ
            IOError: Nếu có lỗi khi lưu model
        Example:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()
            IOHandler.save_model(model, 'models/rf_model.joblib')
        """
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            if method == "joblib":
                # Joblib: nhanh hơn, tối ưu cho numpy arrays
                joblib.dump(model, file_path)

            elif method == "pickle":
                # Pickle: standard Python serialization
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)

            else:
                raise ValueError(
                    f"Phương thức '{method}' không được hỗ trợ\n"
                    f"Chỉ hỗ trợ: 'joblib' hoặc 'pickle'"
                )

        except Exception as e:
            raise IOError(f"Lỗi khi lưu model: {str(e)}")

    @staticmethod
    def load_model(file_path: str, method: str = "joblib") -> Any:
        """
        Load machine learning model từ file

        Args:
            file_path (str): Đường dẫn đến file model
            method (str): Phương thức load ('joblib' hoặc 'pickle')

        Returns:
            Model object đã được load

        Raises:
            FileNotFoundError: Nếu file không tồn tại
            ValueError: Nếu method không hợp lệ
            IOError: Nếu có lỗi khi load model

        Example:
            model = IOHandler.load_model('models/rf_model.joblib')
            predictions = model.predict(X_test)
        """
        # Kiểm tra file có tồn tại không
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file không tìm thấy: {file_path}")

        try:
            if method == "joblib":
                return joblib.load(file_path)

            elif method == "pickle":
                with open(file_path, 'rb') as f:
                    return pickle.load(f)

            else:
                raise ValueError(
                    f"Phương thức '{method}' không được hỗ trợ\n"
                    f"Chỉ hỗ trợ: 'joblib' hoặc 'pickle'"
                )

        except Exception as e:
            raise IOError(f"Lỗi khi load model: {str(e)}")

    @staticmethod
    def save_json(data: Dict, file_path: str, indent: int = 4) -> None:
        """
        Lưu dictionary ra JSON file

        Args:
            data (Dict): Dictionary cần lưu
            file_path (str): Đường dẫn file output
            indent (int): Số spaces để indent (cho đẹp)

        Example:
            metrics = {'accuracy': 0.95, 'f1': 0.93}
            IOHandler.save_json(metrics, 'results/metrics.json')
        """
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    @staticmethod
    def load_json(file_path: str) -> Dict:
        """
        Load JSON file thành dictionary

        Args:
            file_path (str): Đường dẫn đến JSON file

        Returns:
            Dict: Dictionary chứa dữ liệu từ JSON

        Raises:
            FileNotFoundError: Nếu file không tồn tại

        Example:
            metrics = IOHandler.load_json('results/metrics.json')
            print(metrics['accuracy'])
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file không tìm thấy: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

# ===== 4. MetricsCalculator =====
class MetricsCalculator:
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred, y_pred_proba):
        """Tính accuracy, precision, recall, f1, roc_auc"""

    @staticmethod
    def print_metrics(metrics: Dict, logger):
        """In metrics ra console + log"""

# ===== 5. DirectoryManager =====
class DirectoryManager:
    @staticmethod
    def create_project_structure(base_dir: str = "."):
        """Tạo cấu trúc thư mục project"""
        dirs = [
            "config",
            "data/raw",
            "data/processed",
            "notebooks",
            "src",
            "artifacts/models",
            "artifacts/figures",
            "artifacts/results",
            "artifacts/logs"
        ]
        for dir_path in dirs:
            os.makedirs(os.path.join(base_dir, dir_path), exist_ok=True)

# ===== 6. Helper Functions =====
def set_random_seed(seed: int = 42):
    """Set random seed cho numpy, random, torch"""
    np.random.seed(seed)
    # torch if available

def get_timestamp() -> str:
    """Lấy timestamp: YYYYMMDD_HHMMSS"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')
"""
Model Training, Optimization và Evaluation Pipeline
Hỗ trợ multiple models và hyperparameter tuning
"""
import pandas as pd
import os
from typing import Dict, Tuple, Any, Optional
from datetime import datetime

# Scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_curve, roc_auc_score, accuracy_score,
                            precision_score, recall_score, f1_score)

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Class quản lý toàn bộ quy trình training, optimization và evaluation
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """
        Khởi tạo ModelTrainer
        
        Args:
            config: Dictionary chứa cấu hình
            logger: Logger object
        """
        self.config = config
        self.logger = logger
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        if self.logger:
            self.logger.info("ModelTrainer initialized")
    
    def __repr__(self) -> str:
        return f"ModelTrainer(models={len(self.models)}, best={self.best_model_name})"
    
    def load_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Load dữ liệu vào trainer
        
        Args:
            X: Features
            y: Target
        """
        self.X = X
        self.y = y

        if self.logger:
            self.logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    def split_data(self, test_size: float = None, random_state: int = None) -> Tuple:
        """
        Chia dữ liệu thành train/test
        
        Args:
            test_size: Tỷ lệ test set
            random_state: Random seed
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or self.config.get('data', {}).get('test_size', 0.2)
        random_state = random_state or self.config.get('data', {}).get('random_state', 42)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.y
        )
        
        if self.logger:
            self.logger.info(f"Data split: Train={len(self.X_train)}, Test={len(self.X_test)}")
            self.logger.info(f"Train class distribution: {dict(self.y_train.value_counts())}")
            self.logger.info(f"Test class distribution: {dict(self.y_test.value_counts())}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _get_model_instance(self, model_name: str, params: Dict = None) -> Any:
        """
        Tạo instance của model
        
        Args:
            model_name: Tên model
            params: Parameters cho model
            
        Returns:
            Model instance
        """
        params = params or {}
        random_state = self.config.get('data', {}).get('random_state', 42)
        
        if model_name == 'random_forest':
            return RandomForestClassifier(random_state=random_state, **params)
        
        elif model_name == 'logistic_regression':
            return LogisticRegression(random_state=random_state, max_iter=1000, **params)
        
        elif model_name == 'svm':
            return SVC(random_state=random_state, probability=True, **params)
        
        elif model_name == 'decision_tree':
            return DecisionTreeClassifier(random_state=random_state, **params)
        
        elif model_name == 'adaboost':
            return AdaBoostClassifier(random_state=random_state, **params)
        
        elif model_name == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed")
            return XGBClassifier(random_state=random_state, n_jobs = -1, **params)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_model(self, model_name: str, params: Dict = None) -> Any:
        """
        Train một model
        
        Args:
            model_name: Tên model
            params: Parameters cho model
            
        Returns:
            Trained model
        """
        if self.logger:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Training {model_name.upper()}")
            self.logger.info(f"{'='*50}")
        
        model = self._get_model_instance(model_name, params)
        
        # Train
        start_time = datetime.now()
        model.fit(self.X_train, self.y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)
        
        if self.logger:
            self.logger.info(f"Training time: {training_time:.2f}s")
            self.logger.info(f"Train accuracy: {train_score:.4f}")
            self.logger.info(f"Test accuracy: {test_score:.4f}")
        
        # Store model
        self.models[model_name] = model
        
        return model
    
    def optimize_params(self, model_name: str) -> Tuple[Any, Dict]:
        """
        Tối ưu hyperparameters cho model
        
        Args:
            model_name: Tên model
            
        Returns:
            (Best model, Best parameters)
        """
        if self.logger:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"OPTIMIZING {model_name.upper()}")
            self.logger.info(f"{'='*50}")
        
        # Get param grid
        param_grid = self.config.get('models', {}).get(model_name, {})
        if not param_grid:
            if self.logger:
                self.logger.warning(f"No param grid found for {model_name}, using default params")
            return self.train_model(model_name), {}
        
        # Get tuning config
        tuning_config = self.config.get('tuning', {})
        method = tuning_config.get('method', 'randomized')
        cv_folds = tuning_config.get('cv_folds', 5)
        cv_strategy = tuning_config.get('cv_strategy', 'stratified')
        scoring = tuning_config.get('scoring', 'roc_auc')
        n_jobs = tuning_config.get('n_jobs', -1)
        
        # Create base model
        base_model = self._get_model_instance(model_name)
        
        # Setup cross-validation
        if cv_strategy == 'stratified':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv = cv_folds
        
        # Choose search method
        if method == 'grid':
            search = GridSearchCV(
                base_model, param_grid, cv=cv, 
                scoring=scoring, n_jobs=n_jobs, verbose=1
            )
        else:  # randomized
            n_iter = tuning_config.get('n_iter', 30)
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=n_iter, cv=cv,
                scoring=scoring, n_jobs=n_jobs, verbose=1, random_state=42
            )
        
        # Perform search
        if self.logger:
            self.logger.info(f"Method: {method.upper()}")
            self.logger.info(f"CV strategy: {cv_strategy}")
            self.logger.info(f"CV folds: {cv_folds}")
            self.logger.info(f"Scoring: {scoring}")
        
        start_time = datetime.now()
        search.fit(self.X_train, self.y_train)
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Get best model
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        
        if self.logger:
            self.logger.info(f"Optimization time: {optimization_time:.2f}s")
            self.logger.info(f"Best CV {scoring.upper()}: {best_score:.4f}")
            self.logger.info(f"Best parameters: {best_params}")
        
        # Store model
        self.models[model_name] = best_model
        
        return best_model, best_params

    def evaluate(self, model_name: str = None, model: Any = None) -> Dict:
        """
        Đánh giá model trên test set

        Args:
            model_name: Tên model (nếu đã train)
            model: Model instance (nếu truyền trực tiếp)

        Returns:
            Dictionary chứa các metrics
        """
        if model is None:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} chưa được train")
            model = self.models[model_name]
            name = model_name
        else:
            name = model_name or 'custom_model'

        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate metrics
        metrics = {
            'model_name': name,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0)
        }

        roc_curve_data = None
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)

            # Tính toán FPR, TPR để Visualizer chỉ việc lấy ra vẽ
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_curve_data = (fpr, tpr, metrics['roc_auc'])

        # Store results
        self.results[name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(self.y_test, y_pred),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'roc_curve_data': roc_curve_data
        }

        if self.logger:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"EVALUATION: {name.upper()}")
            self.logger.info(f"{'='*50}")
            for metric, value in metrics.items():
                if metric != 'model_name' and value is not None:
                    self.logger.info(f"{metric.upper()}: {value:.4f}")
            self.logger.info(f"{'='*50}")

        return metrics
    
    def train_all_models(self, optimize: bool = True) -> Dict[str, Dict]:
        """
        Train tất cả models được định nghĩa trong config
        
        Args:
            optimize: True nếu cần optimize hyperparameters
            
        Returns:
            Dictionary chứa kết quả của tất cả models
        """
        model_names = list(self.config.get('models', {}).keys())
        
        if not model_names:
            raise ValueError("No models defined in config")
        
        if self.logger:
            self.logger.info(f"\n{'#'*50}")
            self.logger.info(f"TRAINING {len(model_names)} MODELS")
            self.logger.info(f"{'#'*50}")
        
        all_metrics = {}
        
        for model_name in model_names:
            try:
                # Train or optimize
                if optimize:
                    model, best_params = self.optimize_params(model_name)
                else:
                    model = self.train_model(model_name)
                
                # Evaluate
                metrics = self.evaluate(model_name, model)
                all_metrics[model_name] = metrics
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Find best model
        self._select_best_model(all_metrics)
        
        return all_metrics
    
    def _select_best_model(self, all_metrics: Dict[str, Dict]) -> None:
        """
        Chọn model tốt nhất dựa trên scoring metric
        
        Args:
            all_metrics: Dictionary chứa metrics của tất cả models
        """
        scoring_metric = self.config.get('tuning', {}).get('scoring', 'f1')
        
        best_score = -1
        best_name = None
        
        for model_name, metrics in all_metrics.items():
            score = metrics.get(scoring_metric, 0)
            if score > best_score:
                best_score = score
                best_name = model_name
        
        if best_name:
            self.best_model_name = best_name
            self.best_model = self.models[best_name]
            
            if self.logger:
                self.logger.info(f"\n{'*'*50}")
                self.logger.info(f"BEST MODEL: {best_name.upper()}")
                self.logger.info(f"Best {scoring_metric.upper()}: {best_score:.4f}")
                self.logger.info(f"{'*'*50}\n")
    
    def save_model(self, model_name: str = None, file_path: str = None, 
                   method: str = 'joblib') -> str:
        """
        Lưu model ra file
        
        Args:
            model_name: Tên model (mặc định: best model)
            file_path: Đường dẫn file output
            method: Phương thức lưu (joblib/pickle)
            
        Returns:
            Đường dẫn file đã lưu
        """
        from Utils import IOHandler, get_timestamp
        
        # Determine which model to save
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        # Generate file path
        if file_path is None:
            models_dir = self.config.get('artifacts', {}).get('models_dir', 'artifacts/models')
            timestamp = get_timestamp()
            file_path = os.path.join(models_dir, f"{model_name}_{timestamp}.{method}")
        
        # Save
        IOHandler.save_model(model, file_path, method)
        
        if self.logger:
            self.logger.info(f"Model saved: {file_path}")
        
        return file_path
    
    def load_model(self, file_path: str, method: str = 'joblib') -> Any:
        """
        Load model từ file
        
        Args:
            file_path: Đường dẫn file
            method: Phương thức load
            
        Returns:
            Model object
        """
        from Utils import IOHandler
        
        model = IOHandler.load_model(file_path, method)
        
        if self.logger:
            self.logger.info(f"Model loaded: {file_path}")
        
        return model
    
    def save_results(self, file_path: str = None) -> str:
        """
        Lưu kết quả evaluation ra file JSON
        
        Args:
            file_path: Đường dẫn file output
            
        Returns:
            Đường dẫn file đã lưu
        """
        from Utils import IOHandler, get_timestamp
        
        if file_path is None:
            results_dir = self.config.get('artifacts', {}).get('results_dir', 'artifacts/results')
            timestamp = get_timestamp()
            file_path = os.path.join(results_dir, f"results_{timestamp}.json")
        
        # Prepare results for JSON
        results_json = {}
        for model_name, result in self.results.items():
            results_json[model_name] = {
                'metrics': result['metrics'],
                'confusion_matrix': result['confusion_matrix'].tolist()
            }
        
        IOHandler.save_json(results_json, file_path)
        
        if self.logger:
            self.logger.info(f"Results saved: {file_path}")
        
        return file_path
    
    def get_feature_importance(self, model_name: str = None, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Lấy feature importance từ model
        
        Args:
            model_name: Tên model (mặc định: best model)
            top_n: Số features quan trọng nhất
            
        Returns:
            DataFrame chứa feature importance
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models.get(model_name)
        
        if not hasattr(model, 'feature_importances_'):
            if self.logger:
                self.logger.warning(f"Model {model_name} không có feature_importances_")
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df


if __name__ == "__main__":
    print("Testing ModelTrainer...")
    
    # Example usage
    test_config = {
        'data': {'test_size': 0.2, 'random_state': 42},
        'models': {
            'random_forest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
            'logistic_regression': {'C': [0.1, 1, 10]}
        },
        'tuning': {'method': 'randomized', 'cv_folds': 3, 'n_iter': 5, 'scoring': 'f1'}
    }
    
    trainer = ModelTrainer(test_config)
    print(f"✓ {trainer}")
    print("✓ ModelTrainer is ready!")
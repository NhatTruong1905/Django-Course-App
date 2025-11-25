import sys
import logging
from . import DataPreprocessor, ModelTrainer, DataVisualizer, IOHandler

class Pipeline:
    """
    Class quản lý toàn bộ quy trình E-commerce Churn Prediction.
    Đóng gói logic xử lý để main.py gọi gọn gàng.
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.preprocessor = None
        self.trainer = None

    def run_eda(self):
        """Chạy phân tích dữ liệu khám phá (EDA)"""
        self.logger.info("\n" + "="*70)
        self.logger.info("EXPLORATORY DATA ANALYSIS")
        self.logger.info("="*70)

        # Load raw data
        data_path = self.config['data']['raw_path']
        self.logger.info(f"Loading data for EDA from: {data_path}")
        df = IOHandler.read_data(data_path)

        # Visualize
        visualizer = DataVisualizer(self.config, self.logger)
        
        self.logger.info("Plotting missing values...")
        visualizer.plot_missing_values(df)
        
        self.logger.info("Plotting target distribution...")
        visualizer.plot_target_distribution(df['Churn'])
        
        self.logger.info("Plotting correlation matrix...")
        visualizer.plot_correlation_matrix(df)
        
        self.logger.info("EDA completed!")

    def run_preprocessing(self):
        """Chạy quy trình tiền xử lý dữ liệu"""
        self.logger.info("\n" + "="*70)
        self.logger.info("STAGE 1: DATA PREPROCESSING")
        self.logger.info("="*70)
        
        # Initialize
        self.preprocessor = DataPreprocessor(self.config['preprocessing'], self.logger)
        
        # Load data
        data_path = self.config['data']['raw_path']
        self.logger.info(f"Loading data from: {data_path}")
        
        # Handle Excel logic inside pipeline (cleaner main)
        if data_path.endswith(('.xlsx', '.xls')):
            sheet_name = self.config['data'].get('sheet_name', 0)
            df = self.preprocessor.load_data(data_path, sheet_name=sheet_name)
        else:
            df = self.preprocessor.load_data(data_path)
        
        # Transform
        X, y = self.preprocessor.fit_transform(df, target_col='Churn')
        
        # Save processed data
        processed_path = self.config['data']['processed_path']
        processed_df = X.copy()
        processed_df['Churn'] = y
        IOHandler.save_data(processed_df, processed_path)
        self.logger.info(f"Processed data saved to: {processed_path}")
        
        return X, y

    def run_training(self, X, y, optimize=True):
        """Chạy quy trình huấn luyện"""
        self.logger.info("STAGE 2: MODEL TRAINING & OPTIMIZATION")
        
        self.trainer = ModelTrainer(self.config, self.logger)
        self.trainer.load_data(X, y)
        self.trainer.split_data()
        
        # Train
        metrics = self.trainer.train_all_models(optimize=optimize)
        
        # Save artifacts
        model_path = self.trainer.save_model()
        self.logger.info(f"\nBest model saved to: {model_path}")
        
        results_path = self.trainer.save_results()
        self.logger.info(f"Results saved to: {results_path}")
        
        return self.trainer, metrics

    def run_visualization(self, trainer, metrics):
        """Chạy quy trình vẽ biểu đồ báo cáo (Đã fix lỗi tham số)"""
        self.logger.info("STAGE 3: VISUALIZATION & ANALYSIS")

        visualizer = DataVisualizer(self.config, self.logger)

        # 1. Comparison
        # FIX: Thêm metric_names vào
        metric_names = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
        visualizer.plot_model_comparison(metrics, metric_names)

        # 2. Confusion Matrix
        for model_name, result in trainer.results.items():
            cm = result['confusion_matrix']
            # FIX: Thêm labels=[0, 1] và XÓA save_name (vì hàm không nhận tham số này)
            visualizer.plot_confusion_matrix(
                cm,
                labels=[0, 1],
                title=f'Confusion Matrix - {model_name.upper()}'
            )

        # 3. ROC Curve
        # FIX: Lấy dữ liệu roc_curve_data đã tính ở ModelTrainer ra
        roc_results = {}
        for model_name, result in trainer.results.items():
            # Kiểm tra xem ModelTrainer đã tính ROC chưa
            if result.get('roc_curve_data') is not None:
                roc_results[model_name] = result['roc_curve_data']

        if roc_results:
            visualizer.plot_roc_curve(roc_results)

        # 4. Feature Importance (Best Model)
        if self.config.get('evaluation', {}).get('feature_importance', True):
            importance_df = trainer.get_feature_importance()
            if importance_df is not None:
                # FIX: Thêm top_n=20 và XÓA title, save_name (hàm tự xử lý)
                visualizer.plot_feature_importance(
                    importance_df,
                    top_n=20
                )

        self.logger.info("Visualization completed!")

    def load_processed_data(self):
        """Hàm phụ trợ để load data đã xử lý (dùng cho mode train riêng lẻ)"""
        processed_path = self.config['data']['processed_path']
        self.logger.info(f"Loading processed data from: {processed_path}")
        df = IOHandler.read_data(processed_path)
        X = df.drop(columns=['Churn'])
        y = df['Churn']
        return X, y



if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import shutil
    import os
    # Vì chạy trực tiếp file này, ta cần config giả lập
    print("\n" + "=" * 50)
    print("🧪 TESTING PIPELINE FLOW (MOCK DATA)")
    print("=" * 50)

    # 1. Tạo Config Giả (Mock Config)
    # Lưu tạm vào thư mục test_output để không ảnh hưởng folder thật
    test_config = {
        'data': {
            'raw_path': 'test_raw_dummy.csv',
            'processed_path': 'test_output/processed_dummy.csv',
            'target_col': 'Churn',
            'test_size': 0.2,
            'random_state': 42
        },
        'preprocessing': {
            'missing_strategy': {'numerical': 'median', 'categorical': 'mode'},
            'feature_selection': False,
            'scaler_type': 'standard',
            'categorical_encoding': 'label'
        },
        'models': {
            # Test với model nhẹ nhất để chạy nhanh
            'random_forest': {'n_estimators': 5, 'max_depth': 3}
        },
        'tuning': {
            'cv_folds': 2,  # Fold ít thôi cho nhanh
            'scoring': 'accuracy'
        },
        'evaluation': {
            'feature_importance': True
        },
        'artifacts': {
            'logs_dir': 'test_output/logs',
            'models_dir': 'test_output/models',
            'results_dir': 'test_output/results',
            'figures_dir': 'test_output/figures'
        }
    }

    # 2. Tạo Dữ liệu Giả (Mock Data)
    print("1. Generating dummy data...")
    df_dummy = pd.DataFrame({
        'Tenure': np.random.randint(1, 20, 50),
        'CityTier': np.random.choice([1, 2, 3], 50),
        'WarehouseToHome': np.random.randint(5, 35, 50),
        'Gender': np.random.choice(['Male', 'Female'], 50),
        'Churn': np.random.choice([0, 1], 50)  # Target
    })
    # Lưu file giả để pipeline đọc vào (giả vờ như file thật)
    df_dummy.to_csv('test_raw_dummy.csv', index=False)

    # 3. Khởi tạo Pipeline
    # Logger tạm
    logger = logging.getLogger("TEST_PIPE")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    pipeline = Pipeline(test_config, logger)

    try:
        # 4. Chạy thử các bước
        print("\n2. Testing Preprocessing...")
        X, y = pipeline.run_preprocessing()
        print(f"   -> OK. Shape: {X.shape}")

        print("\n3. Testing Training (No Optimize)...")
        trainer, metrics = pipeline.run_training(X, y, optimize=False)
        print(f"   -> OK. Metrics: {metrics}")

        print("\n4. Testing Visualization...")
        pipeline.run_visualization(trainer, metrics)
        print("   -> OK. Plots saved.")

        print("\n✅ PIPELINE FLOW TEST PASSED!")

    except Exception as e:
        print(f"\TEST FAILED: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 5. Dọn dẹp rác (Clean up)
        print("\n5. Cleaning up test files...")
        if os.path.exists('test_raw_dummy.csv'):
            os.remove('test_raw_dummy.csv')
        if os.path.exists('test_output'):
            shutil.rmtree('test_output')  # Xóa thư mục tạm
        print("   -> Cleaned.")
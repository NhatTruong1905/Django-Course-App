"""
Main Pipeline Script - E-commerce Customer Churn Prediction
Đồ án cuối kỳ Python cho Khoa học Dữ liệu - K23

Usage:
    python main.py --mode full
    python main.py --mode preprocess
    python main.py --mode train --optimize
    python main.py --mode visualize
"""
import argparse
import sys
# Add src to path (Giữ lại theo ý bạn cho chắc chắn)
sys.path.append('src')

# Import Class Pipeline và các Utils cần thiết
from src import ConfigLoader, Logger, set_random_seed, get_timestamp
from src import Pipeline  

def parse_arguments():
    parser = argparse.ArgumentParser(description='E-commerce Customer Churn Pipeline')
    
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['full', 'preprocess', 'train', 'visualize', 'eda'],
                        help='Pipeline mode to run')
    
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--data', type=str, default=None, help='Override data path')
    parser.add_argument('--optimize', action='store_true', help='Enable hyperparameter optimization')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization in full mode')
    
    return parser.parse_args()

def main():
    """Main Entry Point"""
    args = parse_arguments()
    
    # 1. Load Config
    print(f"\nLoading configuration from: {args.config}")
    config = ConfigLoader.load_config(args.config)
    
    # Override data path
    if args.data:
        config['data']['raw_path'] = args.data
    
    # 2. Setup Reproducibility
    if config.get('mlops', {}).get('reproducibility', {}).get('set_seed', True):
        seed = config['mlops']['reproducibility'].get('seed_value', 42)
        set_random_seed(seed)
    
    # 3. Initialize Logger
    logger = Logger.get_logger(
        'main_controller',
        log_dir=config.get('artifacts', {}).get('logs_dir', 'artifacts/logs'),
        level=config.get('logging', {}).get('level', 'INFO')
    )
    
    logger.info("="*70)
    logger.info(f"PIPELINE STARTED | MODE: {args.mode.upper()}")
    logger.info(f"Timestamp: {get_timestamp()}")
    logger.info("="*70)
    
    # 4. KHỞI TẠO PIPELINE (Chỉ 1 dòng duy nhất)
    pipeline = Pipeline(config, logger)
    
    try:
        # 5. Điều khiển luồng chạy (Flow Control)
        
        if args.mode == 'full':
            # Chạy từ A-Z
            X, y = pipeline.run_preprocessing()
            trainer, metrics = pipeline.run_training(X, y, optimize=args.optimize)
            
            if not args.no_viz:
                pipeline.run_visualization(trainer, metrics)
                
            # Log kết quả cuối cùng
            logger.info("\n" + "="*70)
            logger.info(f"BEST MODEL: {trainer.best_model_name.upper()}")
            logger.info(f"METRICS: {metrics[trainer.best_model_name]}")
            logger.info("="*70)

        elif args.mode == 'preprocess':
            pipeline.run_preprocessing()
            
        elif args.mode == 'train':
            # Nếu chỉ chạy train, phải load data đã xử lý
            X, y = pipeline.load_processed_data()
            pipeline.run_training(X, y, optimize=args.optimize)
            
        elif args.mode == 'visualize':
            # Mode này cần logic hơi phức tạp để load lại model cũ
            # Ở đây ta demo chạy EDA thay thế nếu user muốn xem data
            pipeline.run_eda()
            
        elif args.mode == 'eda':
            pipeline.run_eda()
            
        logger.info("\n✓ Pipeline execution completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
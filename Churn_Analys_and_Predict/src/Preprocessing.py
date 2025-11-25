"""
Data Preprocessing Pipeline
Xử lý dữ liệu E-commerce Customer Churn
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Class xử lý tiền xử lý dữ liệu
    Bao gồm: đọc dữ liệu, xử lý missing values, outliers, encoding, scaling, feature engineering
    """

    def __init__(self, config: Dict, logger=None):
        """
        Khởi tạo DataPreprocessor

        Args:
            config: Dictionary chứa cấu hình preprocessing
            logger: Logger object
        """
        self.config = config
        self.logger = logger
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.numerical_cols = []
        self.categorical_cols = []

        if self.logger:
            self.logger.info("DataPreprocessor initialized")

    def __repr__(self) -> str:
        return f"DataPreprocessor(scaler={type(self.scaler).__name__ if self.scaler else None})"

    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Đọc dữ liệu từ file (hỗ trợ Excel với sheet name)

        Args:
            file_path: Đường dẫn file
            **kwargs: Tham số bổ sung cho pandas

        Returns:
            DataFrame
        """
        try:
            from Utils import IOHandler

            # Nếu là Excel và có sheet_name trong kwargs
            if file_path.endswith(('.xlsx', '.xls')):
                df = IOHandler.read_data(file_path, **kwargs)
            else:
                if 'sheet_name' in kwargs:
                    if self.logger:
                        self.logger.warning(f"File CSV không dùng sheet -> loại bỏ sheet .")
                    kwargs.pop('sheet_name')
                df = IOHandler.read_data(file_path, **kwargs)

            if self.logger:
                self.logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

            return df
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading data: {str(e)}")
            raise

    def inspect_data(self, df: pd.DataFrame) -> Dict:
        """
        Phân tích sơ bộ dữ liệu

        Args:
            df: DataFrame cần phân tích

        Returns:
            Dictionary chứa thông tin về dữ liệu
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }

        if self.logger:
            self.logger.info(f"Data inspection completed")
            self.logger.info(f"Shape: {info['shape']}")
            self.logger.info(f"Missing values: {sum(info['missing_values'].values())}")
            self.logger.info(f"Duplicates: {info['duplicates']}")

        return info

    def _identify_column_types(self, df: pd.DataFrame, target_col: str = 'Churn') -> None:
        """Tự động phát hiện loại cột (numerical/categorical)"""
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Loại bỏ target column
        if target_col in self.numerical_cols:
            self.numerical_cols.remove(target_col)
        if target_col in self.categorical_cols:
            self.categorical_cols.remove(target_col)

        if self.logger:
            self.logger.info(f"Numerical columns: {len(self.numerical_cols)}")
            self.logger.info(f"Categorical columns: {len(self.categorical_cols)}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu cơ bản
        - Chuẩn hóa tên cột
        - Xử lý duplicates
        - Chuẩn hóa giá trị categorical

        Args:
            df: DataFrame gốc

        Returns:
            DataFrame đã được làm sạch
        """
        df = df.copy()

        # Chuẩn hóa tên cột
        df.columns = df.columns.str.strip()

        # Xóa duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        if self.logger and initial_rows > len(df):
            self.logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

        # Chuẩn hóa categorical values (dựa trên dataset)
        if 'PreferredLoginDevice' in df.columns:
            df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace({
                'Phone': 'Mobile Phone'
            })

        if 'PreferredPaymentMode' in df.columns:
            df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({
                'CC': 'Credit Card',
                'Cash on Delivery': 'COD'
            })

        if self.logger:
            self.logger.info("Data cleaning completed")

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xử lý missing values

        Args:
            df: DataFrame

        Returns:
            DataFrame đã xử lý missing values
        """
        df = df.copy()

        missing_config = self.config.get('missing_strategy', {})
        num_strategy = missing_config.get('numerical', 'median')
        cat_strategy = missing_config.get('categorical', 'mode')

        # Xử lý numerical columns
        for col in self.numerical_cols:
            if df[col].isnull().sum() > 0:
                if num_strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif num_strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif num_strategy == 'mode':
                    df[col].fillna(df[col].mode()[0], inplace=True)

                if self.logger:
                    self.logger.info(f"Filled missing values in {col} with {num_strategy}")

        # Xử lý categorical columns
        for col in self.categorical_cols:
            if df[col].isnull().sum() > 0:
                if cat_strategy == 'mode':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif cat_strategy == 'unknown':
                    df[col].fillna('Unknown', inplace=True)

                if self.logger:
                    self.logger.info(f"Filled missing values in {col} with {cat_strategy}")

        return df

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xử lý outliers bằng IQR method hoặc Z-score

        Args:
            df: DataFrame

        Returns:
            DataFrame đã xử lý outliers
        """
        df = df.copy()

        method = self.config.get('outlier_method', 'iqr')
        threshold = self.config.get('outlier_threshold', 1.5)

        if method == 'iqr':
            for col in self.numerical_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                # Clip values
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        elif method == 'zscore':
            from scipy import stats
            for col in self.numerical_cols:
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < threshold]

        elif method == 'isolation_forest':
            iso_forest = IsolationForest(contamination=0.01, random_state=42)
            outliers = iso_forest.fit_predict(df[self.numerical_cols])
            df = df[outliers == 1]

        if self.logger:
            self.logger.info(f"Outliers handled using {method} method")

        return df

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Mã hóa categorical variables

        Args:
            df: DataFrame
            fit: True nếu fit encoder, False nếu chỉ transform

        Returns:
            DataFrame đã encode
        """
        df = df.copy()

        encoding_method = self.config.get('categorical_encoding', 'label')

        if encoding_method == 'label':
            for col in self.categorical_cols:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Xử lý unseen labels
                        le = self.label_encoders[col]
                        df[col] = df[col].apply(
                            lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                        )

        if self.logger:
            self.logger.info(f"Categorical encoding completed using {encoding_method}")

        return df

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Chuẩn hóa dữ liệu

        Args:
            df: DataFrame
            fit: True nếu fit scaler, False nếu chỉ transform

        Returns:
            DataFrame đã scale
        """
        df = df.copy()

        scaler_type = self.config.get('scaler_type', 'standard')

        if fit:
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                self.scaler = RobustScaler()

            df[self.numerical_cols] = self.scaler.fit_transform(df[self.numerical_cols])
        else:
            if self.scaler:
                df[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols])

        if self.logger:
            self.logger.info(f"Features scaled using {scaler_type} scaler")

        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo features mới - Advanced Feature Engineering
        Đặc biệt cho E-commerce Churn dataset

        Args:
            df: DataFrame

        Returns:
            DataFrame với features mới
        """
        df = df.copy()

        if not self.config.get('create_features', False):
            return df

        if self.logger:
            self.logger.info("Creating advanced features...")

        # 1. Behavioral Features
        # Average cashback per order
        df['avg_cashbk_per_order'] = df['CashbackAmount'] / np.maximum(df['OrderCount'], 1)

        # Order frequency per month
        df['order_frequency_mpm'] = df['OrderCount'] / np.maximum(df['Tenure'], 1)

        # Coupon usage rate
        df['coupon_rate'] = df['CouponUsed'] / np.maximum(df['OrderCount'], 1)

        # Engagement score (time spent * devices)
        df['engagement_score'] = df['HourSpendOnApp'] * np.log1p(df['NumberOfDeviceRegistered'])

        # 2. Log Transform Features (for skewed distributions)
        df['log_recency'] = np.log1p(df['DaySinceLastOrder'])
        df['log_distance'] = np.log1p(df['WarehouseToHome'])
        df['log_OrderCount'] = np.log1p(df['OrderCount'])
        df['log_CashbackAmount'] = np.log1p(df['CashbackAmount'])
        df['log_WarehouseToHome'] = np.log1p(df['WarehouseToHome'])
        df['log_DaySinceLastOrder'] = np.log1p(df['DaySinceLastOrder'])

        # 3. Interaction Features
        df['growth_x_freq'] = df['OrderAmountHikeFromlastYear'] * df['order_frequency_mpm']
        df['satis_x_complain'] = df['SatisfactionScore'] * (1 - df['Complain'].astype(int))

        # 4. Binary Indicators
        df['city_is_tier1'] = (df['CityTier'] == '1').astype(int) if df['CityTier'].dtype == 'object' else (df['CityTier'] == 1).astype(int)
        df['multi_address'] = (df['NumberOfAddress'] > 1).astype(int)

        # Update numerical columns list
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'Churn' in self.numerical_cols:
            self.numerical_cols.remove('Churn')

        if self.logger:
            n_new_features = len([col for col in df.columns if col not in self.feature_names])
            self.logger.info(f"Created {n_new_features} new features")
            self.logger.info(f"New features include: behavioral, log-transforms, interactions, binary indicators")

        return df

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                        n_features: int = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Feature selection sử dụng statistical tests

        Args:
            X: Features
            y: Target
            n_features: Số features cần chọn

        Returns:
            (Selected features DataFrame, Selected feature names)
        """
        if not self.config.get('feature_selection', False):
            return X, list(X.columns)

        n_features = n_features or self.config.get('n_top_features', 15)
        n_features = min(n_features, X.shape[1])

        # Mặc định dùng f_classif (nhanh hơn), nếu config chọn 'mutual_info' thì đổi
        method = self.config.get('feature_selection_method', 'f_classif')

        if method == 'mutual_info':
            if self.logger: self.logger.info("Using Mutual Information for feature selection")
            # random_state để kết quả ổn định
            score_func = lambda X, y: mutual_info_classif(X, y, random_state=42)
        else:
            if self.logger: self.logger.info("Using ANOVA F-value for feature selection")
            score_func = f_classif

        selector = SelectKBest(score_func=score_func, k=n_features)
        X_selected = selector.fit_transform(X, y)
        mask = selector.get_support()

        # noinspection PyTypeChecker
        #Lệnh comment trên để tắt báo warning mask từ pycharm mặc dù mask vẫn ổn tồn tại trong list)
        selected_features = [
            feature for feature, is_selected in zip(X.columns, list(mask))
            if is_selected
        ]

        # Fix 2: Giữ nguyên index của X gốc
        X_selected_df = pd.DataFrame(
            X_selected,
            columns=selected_features,
            index=X.index  # ← QUAN TRỌNG: giữ nguyên index
        )

        if self.logger:
            self.logger.info(f"Selected top {n_features} features: {selected_features}")

        return X_selected_df, selected_features

    def fit_transform(self, df: pd.DataFrame, target_col: str = 'Churn') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Pipeline hoàn chỉnh: fit và transform data

        Args:
            df: DataFrame gốc
            target_col: Tên cột target

        Returns:
            (X_processed, y)
        """
        if self.logger:
            self.logger.info("="*50)
            self.logger.info("Starting data preprocessing pipeline (FIT mode)")
            self.logger.info("="*50)

        # 1. Inspect
        self.inspect_data(df)

        # 2. Clean
        df = self.clean_data(df)

        # 3. Identify column types
        self._identify_column_types(df, target_col)
        self.feature_names = [col for col in df.columns if col != target_col]

        # 4. Separate X and y
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 5. Handle missing values
        X = self.handle_missing_values(X)

        # 6. Feature engineering (BEFORE outlier handling)
        X = self.feature_engineering(X)

        # 7. Handle outliers
        X = self.handle_outliers(X)

        # 8. Encode categorical
        X = self.encode_categorical(X, fit=True)

        # 9. Apply SMOTE balancing if configured
        if self.config.get('use_smote', False):
            if self.logger:
                self.logger.info("Applying SMOTETomek for data balancing...")
            X, y = self._apply_smote(X, y)

        # 10. Scale features (AFTER SMOTE)
        X = self.scale_features(X, fit=True)

        # 11. Feature selection (optional, turned off by default)
        if self.config.get('feature_selection', False):
            X, selected_features = self.select_features(X, y)
            self.feature_names = selected_features

        if self.logger:
            self.logger.info("="*50)
            self.logger.info("Preprocessing completed!")
            self.logger.info(f"Final shape: {X.shape}")
            self.logger.info(f"Class distribution: {dict(pd.Series(y).value_counts())}")
            self.logger.info("="*50)

        return X, y

    def _apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTETomek for handling imbalanced data

        Args:
            X: Features
            y: Target

        Returns:
            (X_balanced, y_balanced)
        """
        try:
            from imblearn.combine import SMOTETomek

            if self.logger:
                self.logger.info(f"Before SMOTE: {X.shape}, Class distribution: {dict(y.value_counts())}")

            smt = SMOTETomek(random_state=42)
            X_balanced, y_balanced = smt.fit_resample(X, y)

            # Convert back to DataFrame/Series
            X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
            y_balanced = pd.Series(y_balanced, name=y.name)

            if self.logger:
                self.logger.info(f"After SMOTE: {X_balanced.shape}, Class distribution: {dict(y_balanced.value_counts())}")

            return X_balanced, y_balanced

        except ImportError:
            if self.logger:
                self.logger.warning("imbalanced-learn not installed. Skipping SMOTE.")
            return X, y

    def transform(self, df: pd.DataFrame, target_col: str = 'Churn') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform data sử dụng fitted preprocessor

        Args:
            df: DataFrame mới
            target_col: Tên cột target

        Returns:
            (X_processed, y)
        """
        if self.logger:
            self.logger.info("Transforming new data (TRANSFORM mode)")

        df = self.clean_data(df)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X = self.handle_missing_values(X)
        X = self.feature_engineering(X)
        X = self.encode_categorical(X, fit=False)
        X = self.scale_features(X, fit=False)

        # Select same features
        if self.feature_names:
            X = X[self.feature_names]

        return X, y


if __name__ == "__main__":
    print("Testing DataPreprocessor...")

    # Example config
    test_config = {
        'missing_strategy': {'numerical': 'median', 'categorical': 'mode'},
        'outlier_method': 'iqr',
        'outlier_threshold': 1.5,
        'scaler_type': 'standard',
        'categorical_encoding': 'label',
        'create_features': True,
        'feature_selection': True,
        'n_top_features': 15
    }

    preprocessor = DataPreprocessor(test_config)
    print(f"✓ {preprocessor}")
    print("✓ DataPreprocessor is ready!")
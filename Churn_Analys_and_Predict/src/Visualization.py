import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

class DataVisualizer:
    def __init__(self, config: Dict, logger=None):
        self.figures_dir = config['artifacts']['figures_dir']
        self.config = config
        self.logger = logger

    def plot_missing_values(self, df, save=True):
        # Bar chart missing values
        plt.figure(figsize=(10,6))
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        sns.barplot(x=missing.index, y=missing.values)
        plt.xticks(rotation=45)
        plt.title('Missing Values per Column')
        if save:
            plt.savefig(f"{self.figures_dir}/missing_values.png")
        plt.show()
        plt.close()

    def plot_target_distribution(self, y, save=True):
        # Count plot + pie chart
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        sns.countplot(x=y)
        plt.title('Target Variable Distribution')
        plt.subplot(1,2,2)
        y.value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Target Variable Proportion')
        if save:
            plt.savefig(f"{self.figures_dir}/target_distribution.png")
        plt.show()
        plt.close()


    def plot_numerical_distributions(self, df, columns, save=True):
        # Histograms
        for col in columns:
            plt.figure(figsize=(8,4))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            if save:
                plt.savefig(f"{self.figures_dir}/distribution_{col}.png")
            plt.show()
            plt.close()

    def plot_categorical_distributions(self, df, columns, target_col, save=True):
        # Bar charts với churn rate
        for col in columns:
            plt.figure(figsize=(10,5))
            sns.countplot(x=col, hue=target_col, data=df)
            plt.title(f'Distribution of {col} by {target_col}')
            if save:
                plt.savefig(f"{self.figures_dir}/cat_distribution_{col}.png")
            plt.show()
            plt.close()

    def plot_correlation_matrix(self, df, save=True):
        # Heatmap correlation
        plt.figure(figsize=(12,10))
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        if save:
            plt.savefig(f"{self.figures_dir}/correlation_matrix.png")
        plt.show()
        plt.close()

    def plot_confusion_matrix(self, cm, labels, title, save=True):
        # Confusion matrix heatmap
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        if save:
            plt.savefig(f"{self.figures_dir}/{title.replace(' ', '_').lower()}.png")
        plt.show()
        plt.close()

    def plot_roc_curve(self, results, save=True):
        # ROC curves cho multiple models
        plt.figure(figsize=(8,6))
        for model_name, (fpr, tpr, auc) in results.items():
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        if save:
            plt.savefig(f"{self.figures_dir}/roc_curve.png")
        plt.show()
        plt.close()

    def plot_feature_importance(self, importance_df, top_n, save=True):
        # Horizontal bar chart
        top_features = importance_df.nlargest(top_n, 'importance')
        plt.figure(figsize=(10,6))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top {top_n} Feature Importances')
        if save:
            plt.savefig(f"{self.figures_dir}/feature_importance_top_{top_n}.png")
        plt.show()
        plt.close()

    def plot_model_comparison(self, metrics_dict, metric_names, save=True):
        # Bar chart + heatmap
        metrics_df = pd.DataFrame(metrics_dict).T
        plt.figure(figsize=(10,6))
        sns.heatmap(metrics_df[metric_names], annot=True, cmap='YlGnBu')
        plt.title('Model Comparison')
        if save:
            plt.savefig(f"{self.figures_dir}/model_comparison.png")
        plt.show()
        plt.close()



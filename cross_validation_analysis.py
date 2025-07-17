# -*- coding: utf-8 -*-
"""
Cross-Validation Analysis for Parkinson Detection
Created on Sat Dec 28 15:30:00 2024

Author: Đặng Đình Sơn
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate, GridSearchCV, cross_val_predict
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import os
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
import warnings

# Turn off warnings for smoother code execution
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CrossValidationAnalyzer:
    """
    Class for performing Cross-Validation analysis for machine learning models
    """
    def __init__(self, X, y, cv_folds=5, random_state=42):
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        self.cv_results = {}
        
        # Remove problematic column if exists
        if 'Unnamed: 0' in self.X.columns:
            self.X = self.X.drop('Unnamed: 0', axis=1)
            print("✅ Removed 'Unnamed: 0' column")
        
        # Data normalization
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.X_scaled = pd.DataFrame(self.X_scaled, columns=self.X.columns)
        # Define metrics
        self.scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }
        # Define models with optimal parameters
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000, solver='lbfgs'),
            'Random Forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
            'SVM': SVC(random_state=random_state, kernel='rbf'),
            'Decision Tree': DecisionTreeClassifier(random_state=random_state, max_depth=10),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'MLP (Neural Net)': MLPClassifier(random_state=random_state, max_iter=1000, hidden_layer_sizes=(64, 32)),
        }
        # Add XGBoost if available
        if XGBClassifier is not None:
            self.models['XGBoost'] = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        # Add LightGBM if available (commented out due to column name issues)
        # if LGBMClassifier is not None:
        #     self.models['LightGBM'] = LGBMClassifier(random_state=random_state)

    def run_cross_validation(self):
        print("=== START CROSS-VALIDATION ANALYSIS ===")
        print(f"Dataset shape: {self.X.shape}")
        print(f"Class distribution:\n{self.y.value_counts()}")
        print(f"Cross-validation folds: {self.cv_folds}")
        print("🔧 Data preprocessing: StandardScaler applied")
        print("-" * 50)
        for name, model in self.models.items():
            print(f"\n🔍 Evaluating {name}...")
            scores = cross_validate(model, self.X_scaled, self.y, cv=self.cv, scoring=self.scoring, return_train_score=True, n_jobs=None)
            self.cv_results[name] = scores
            print(f"✅ {name}:")
            print(f"   Accuracy: {scores['test_accuracy'].mean():.4f} (±{scores['test_accuracy'].std() * 2:.4f})")
            print(f"   Precision: {scores['test_precision'].mean():.4f} (±{scores['test_precision'].std() * 2:.4f})")
            print(f"   Recall: {scores['test_recall'].mean():.4f} (±{scores['test_recall'].std() * 2:.4f})")
            print(f"   F1-Score: {scores['test_f1'].mean():.4f} (±{scores['test_f1'].std() * 2:.4f})")
        print("\n" + "=" * 50)
        return self.cv_results

    def plot_comparison(self):
        if not self.cv_results:
            print("❌ No cross-validation results found. Please run run_cross_validation() first.")
            return
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        model_names = list(self.cv_results.keys())
        for metric, metric_name in zip(metrics, metric_names):
            means = [self.cv_results[name][metric].mean() for name in model_names]
            stds = [self.cv_results[name][metric].std() for name in model_names]
            plt.figure(figsize=(9, 6))
            bars = plt.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
            plt.title(f'{metric_name} Comparison', fontweight='bold', fontsize=15)
            plt.ylabel(metric_name, fontsize=13)
            plt.xticks(rotation=30, fontsize=11)
            plt.grid(True, alpha=0.3)
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            plt.tight_layout()
            plt.show()

    def create_summary_table(self):
        if not self.cv_results:
            print("❌ No cross-validation results found. Please run run_cross_validation() first.")
            return None
        summary_data = []
        for model_name, scores in self.cv_results.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
                'Precision': f"{scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}",
                'Recall': f"{scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}",
                'F1-Score': f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}"
            }
            summary_data.append(row)
        summary_df = pd.DataFrame(summary_data)
        return summary_df

    def analyze_fold_performance(self, model_name='Random Forest'):
        if model_name not in self.cv_results:
            print(f"❌ Model '{model_name}' not found in results.")
            return
        scores = self.cv_results[model_name]
        print(f"=== DETAILED FOLD ANALYSIS FOR {model_name.upper()} ===")
        print(f"{'Fold':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 50)
        for i in range(self.cv_folds):
            print(f"{i+1:<6} {scores['test_accuracy'][i]:<10.4f} {scores['test_precision'][i]:<10.4f} "
                  f"{scores['test_recall'][i]:<10.4f} {scores['test_f1'][i]:<10.4f}")
        print("-" * 50)
        print(f"{'Mean':<6} {scores['test_accuracy'].mean():<10.4f} {scores['test_precision'].mean():<10.4f} "
              f"{scores['test_recall'].mean():<10.4f} {scores['test_f1'].mean():<10.4f}")
        print(f"{'Std':<6} {scores['test_accuracy'].std():<10.4f} {scores['test_precision'].std():<10.4f} "
              f"{scores['test_recall'].std():<10.4f} {scores['test_f1'].std():<10.4f}")

    def analyze_overfitting(self):
        if not self.cv_results:
            print("❌ No cross-validation results found. Please run run_cross_validation() first.")
            return
        models_list = list(self.cv_results.keys())
        for model_name in models_list:
            scores = self.cv_results[model_name]
            train_acc = scores['train_accuracy']
            test_acc = scores['test_accuracy']
            x = np.arange(self.cv_folds)
            width = 0.35
            plt.figure(figsize=(7, 5))
            plt.bar(x - width/2, train_acc, width, label='Train Accuracy', alpha=0.7, color='#F08080')
            plt.bar(x + width/2, test_acc, width, label='Test Accuracy', alpha=0.7, color='#BDB76B')
            plt.title(f'{model_name}', fontsize=15)
            plt.xlabel('Fold', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.xticks(x, [f'Fold {i+1}' for i in range(self.cv_folds)], rotation=0, fontsize=11)
            plt.tight_layout()
            plt.show()

    def get_best_model(self, metric='f1'):
        if not self.cv_results:
            print("❌ No cross-validation results found. Please run run_cross_validation() first.")
            return None
        metric_key = f'test_{metric}'
        if metric_key not in list(self.cv_results.values())[0].keys():
            print(f"❌ Metric '{metric}' is invalid. Available metrics: accuracy, precision, recall, f1")
            return None
        best_model = max(self.cv_results.keys(),
                        key=lambda x: self.cv_results[x][metric_key].mean())
        best_score = self.cv_results[best_model][metric_key].mean()
        return best_model, best_score

    def generate_report(self):
        if not self.cv_results:
            print("❌ No cross-validation results found. Please run run_cross_validation() first.")
            return
        print("=" * 60)
        print("📊 CROSS-VALIDATION SUMMARY REPORT")
        print("=" * 60)
        best_model_f1, best_f1 = self.get_best_model('f1')
        best_model_acc, best_acc = self.get_best_model('accuracy')
        print(f"\n🏆 BEST MODEL:")
        print(f"   - By F1-Score: {best_model_f1} (F1 = {best_f1:.4f})")
        print(f"   - By Accuracy: {best_model_acc} (Accuracy = {best_acc:.4f})")

    def plot_roc_auc_all_models(self):
        """
        Plot ROC Curve and calculate AUC for all classification models
        """
        if not hasattr(self, 'X_scaled'):
            print("❌ Data has not been normalized. Please run cross-validation first.")
            return
        plt.figure(figsize=(10, 8))
        for name, model in self.models.items():
            try:
                if hasattr(model, "predict_proba"):
                    y_score = cross_val_predict(model, self.X_scaled, self.y, cv=self.cv, method="predict_proba", n_jobs=None)[:, 1]
                elif hasattr(model, "decision_function"):
                    y_score = cross_val_predict(model, self.X_scaled, self.y, cv=self.cv, method="decision_function", n_jobs=None)
                else:
                    print(f"⚠️ Model {name} does not support ROC/AUC (no predict_proba/decision_function)")
                    continue
                fpr, tpr, _ = roc_curve(self.y, y_score)
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_score:.3f})")
            except Exception as e:
                print(f"❌ Error with model {name}: {e}")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for Models\nAuthor: Đặng Đình Sơn")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def analyze_clustering(self, methods=['KMeans', 'DBSCAN'], n_clusters=2, plot_3d=False, save_to_csv=False, csv_prefix='clustering_result'):
        """
        Analyze clustering (KMeans, DBSCAN) on normalized data.
        Plot PCA 2D or 3D, print clustering results, silhouette score, compare with true labels if available.
        If save_to_csv=True, save results to CSV file.
        """
        print("\n=== UNSUPERVISED CLUSTERING ANALYSIS ===")
        X = self.X_scaled.values if hasattr(self.X_scaled, 'values') else self.X_scaled
        # Reduce data to 2D or 3D for visualization
        n_components = 3 if plot_3d else 2
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        for method in methods:
            if method == 'KMeans':
                model = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                labels = model.fit_predict(X)
                sil = silhouette_score(X, labels)
                print(f"\nKMeans (n_clusters={n_clusters}): Silhouette Score = {sil:.3f}")
            elif method == 'DBSCAN':
                model = DBSCAN(eps=0.5, min_samples=5)
                labels = model.fit_predict(X)
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters_found > 1:
                    sil = silhouette_score(X, labels)
                    print(f"\nDBSCAN: Number of clusters found = {n_clusters_found}, Silhouette Score = {sil:.3f}")
                else:
                    print(f"\nDBSCAN: Number of clusters found = {n_clusters_found}, Silhouette Score = N/A (less than 2 clusters)")
            else:
                print(f"Clustering method not supported: {method}")
                continue
            # Plot clustering results
            if plot_3d and n_components == 3:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=(8, 7))
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='tab10', alpha=0.7)
                ax.set_title(f"{method} Clustering (PCA 3D)")
                ax.set_xlabel('PCA 1')
                ax.set_ylabel('PCA 2')
                ax.set_zlabel('PCA 3')
                fig.colorbar(scatter, label='Cluster Label')
                plt.tight_layout()
                plt.show()
            else:
                plt.figure(figsize=(7, 6))
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
                plt.title(f"{method} Clustering (PCA 2D)")
                plt.xlabel('PCA 1')
                plt.ylabel('PCA 2')
                plt.colorbar(scatter, label='Cluster Label')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            # Compare with true labels if available
            if hasattr(self, 'y') and self.y is not None:
                y_true = self.y.values if hasattr(self.y, 'values') else self.y
                # Only compare if number of unique labels > 1 and number of clusters > 1
                if len(set(y_true)) > 1 and len(set(labels)) > 1:
                    ari = adjusted_rand_score(y_true, labels)
                    print(f"Adjusted Rand Index (vs. true labels): {ari:.3f}")
                    # If number of clusters equals number of classes, try to compute accuracy (after label mapping)
                    def best_map(y_true, y_pred):
                        # Map cluster label to true label for best accuracy
                        labels_true = np.unique(y_true)
                        labels_pred = np.unique(y_pred)
                        cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
                        for i, t in enumerate(labels_true):
                            for j, p in enumerate(labels_pred):
                                cost_matrix[i, j] = np.sum((y_true == t) & (y_pred == p))
                        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
                        mapping = {labels_pred[j]: labels_true[i] for i, j in zip(row_ind, col_ind)}
                        y_pred_mapped = np.array([mapping.get(l, l) for l in y_pred])
                        return y_pred_mapped
                    if len(set(labels)) == len(set(y_true)):
                        y_pred_mapped = best_map(y_true, labels)
                        acc = accuracy_score(y_true, y_pred_mapped)
                        print(f"Clustering Accuracy (after label mapping): {acc:.3f}")
                        print("Confusion Matrix:")
                        print(confusion_matrix(y_true, y_pred_mapped))
                    # Analyze true label ratio in each cluster
                    print("\nTrue label ratio in each cluster:")
                    df = pd.DataFrame({'cluster': labels, 'label': y_true})
                    print(df.groupby('cluster')['label'].value_counts(normalize=True).unstack(fill_value=0).round(2))
            # Save results to file if needed
            if save_to_csv:
                df_out = pd.DataFrame(X)
                for i in range(X_pca.shape[1]):
                    df_out[f'PCA_{i+1}'] = X_pca[:, i]
                df_out['cluster'] = labels
                if hasattr(self, 'y') and self.y is not None:
                    df_out['label'] = y_true
                out_path = f"{csv_prefix}_{method}.csv"
                df_out.to_csv(out_path, index=False)
                print(f"Clustering results saved to file: {out_path}")
        print("\n=== END OF CLUSTERING ANALYSIS ===")

def run_cross_validation_analysis(X, y, cv_folds=5):
    analyzer = CrossValidationAnalyzer(X, y, cv_folds)
    analyzer.run_cross_validation()
    summary_table = analyzer.create_summary_table()
    print("\n=== CROSS-VALIDATION SUMMARY ===")
    print(summary_table.to_string(index=False))
    analyzer.plot_comparison()
    analyzer.analyze_overfitting()
    best_model, _ = analyzer.get_best_model()
    analyzer.analyze_fold_performance(best_model)
    analyzer.generate_report()
    analyzer.plot_roc_auc_all_models()
    analyzer.analyze_clustering()
    return analyzer

if __name__ == "__main__":
    print("Cross-validation analysis module loaded successfully!")
    print("Usage: run_cross_validation_analysis(X, y, cv_folds=5)")
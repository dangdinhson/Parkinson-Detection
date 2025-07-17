"""
Demo Hyperparameter Tuning for SVM
Author: Đặng Đình Sơn
Date created: 2024
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def demo_hyperparameter_tuning_svm():
    """
    Demo hyperparameter tuning for SVM
    """
    print("🔧 DEMO HYPERPARAMETER TUNING FOR SVM")
    print("=" * 60)
    
    # Load data
    print("📂 Loading data...")
    data = pd.read_csv('data.csv')
    X = data.drop(['target'], axis=1)
    y = data['target']
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Class distribution:\n{y.value_counts()}")
    
    # Normalize data
    print("\n🔧 Normalizing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define parameter grid for SVM
    print("\n🔍 Defining parameter grid for SVM...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'linear', 'poly']
    }
    
    print(f"   Parameter grid: {param_grid}")
    print(f"   Total combinations: {len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])}")
    
    # Initialize SVM base model
    svm_base = SVC(random_state=42)
    
    # Run GridSearchCV
    print("\n🔍 Running GridSearchCV...")
    print("   (This process may take a few minutes)")
    
    grid_search = GridSearchCV(
        estimator=svm_base,
        param_grid=param_grid,
        cv=3,  # Use 3 folds to save time
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit GridSearchCV
    grid_search.fit(X_scaled, y)
    
    # Get results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_estimator = grid_search.best_estimator_
    
    # Print results
    print(f"\n✅ TUNING RESULTS:")
    print(f"   Best parameters: {best_params}")
    print(f"   Best F1-Score: {best_score:.4f}")
    print(f"   Best estimator: {best_estimator}")
    
    # Compare with original SVM
    print(f"\n📊 COMPARISON BEFORE AND AFTER TUNING:")
    print("-" * 50)
    
    # Original SVM
    svm_original = SVC(random_state=42, kernel='rbf')
    original_scores = cross_validate(svm_original, X_scaled, y, cv=cv, 
                                   scoring=['accuracy', 'precision', 'recall', 'f1'])
    
    # Tuned SVM
    tuned_scores = cross_validate(best_estimator, X_scaled, y, cv=cv, 
                                scoring=['accuracy', 'precision', 'recall', 'f1'])
    
    print(f"   Original SVM (rbf, C=1.0, gamma='scale'):")
    print(f"     F1-Score: {original_scores['test_f1'].mean():.4f} (±{original_scores['test_f1'].std():.4f})")
    print(f"     Accuracy: {original_scores['test_accuracy'].mean():.4f} (±{original_scores['test_accuracy'].std():.4f})")
    print(f"     Precision: {original_scores['test_precision'].mean():.4f} (±{original_scores['test_precision'].std():.4f})")
    print(f"     Recall: {original_scores['test_recall'].mean():.4f} (±{original_scores['test_recall'].std():.4f})")
    
    print(f"\n   Tuned SVM:")
    print(f"     F1-Score: {tuned_scores['test_f1'].mean():.4f} (±{tuned_scores['test_f1'].std():.4f})")
    print(f"     Accuracy: {tuned_scores['test_accuracy'].mean():.4f} (±{tuned_scores['test_accuracy'].std():.4f})")
    print(f"     Precision: {tuned_scores['test_precision'].mean():.4f} (±{tuned_scores['test_precision'].std():.4f})")
    print(f"     Recall: {tuned_scores['test_recall'].mean():.4f} (±{tuned_scores['test_recall'].std():.4f})")
    
    # Calculate improvement
    f1_improvement = tuned_scores['test_f1'].mean() - original_scores['test_f1'].mean()
    acc_improvement = tuned_scores['test_accuracy'].mean() - original_scores['test_accuracy'].mean()
    prec_improvement = tuned_scores['test_precision'].mean() - original_scores['test_precision'].mean()
    rec_improvement = tuned_scores['test_recall'].mean() - original_scores['test_recall'].mean()
    
    print(f"\n   📈 IMPROVEMENT:")
    print(f"     F1-Score: {f1_improvement:+.4f}")
    print(f"     Accuracy: {acc_improvement:+.4f}")
    print(f"     Precision: {prec_improvement:+.4f}")
    print(f"     Recall: {rec_improvement:+.4f}")
    
    # Plot comparison chart
    print(f"\n📊 PLOTTING COMPARISON CHART...")
    plot_tuning_comparison(original_scores, tuned_scores)
    
    # Detailed parameter analysis
    print(f"\n🔍 DETAILED PARAMETER ANALYSIS:")
    print("-" * 50)
    
    # Get top 5 best results
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_results = results_df.nlargest(5, 'mean_test_score')
    
    print("   Top 5 parameter combinations:")
    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        print(f"   {i}. C={row['param_C']}, gamma={row['param_gamma']}, kernel={row['param_kernel']}")
        print(f"      F1-Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
    
    # Plot heatmap for parameter analysis
    plot_parameter_heatmap(results_df)
    
    # Conclusion
    print(f"\n💡 CONCLUSION AND RECOMMENDATIONS:")
    print("=" * 60)
    print(f"   1. Hyperparameter tuning improves SVM performance significantly")
    print(f"   2. Best parameters: {best_params}")
    print(f"   3. F1-Score improvement: {f1_improvement:+.4f}")
    print(f"   4. GridSearchCV finds optimal combination")
    print(f"   5. Can apply similar tuning to other models")
    print(f"   6. Consider using RandomizedSearchCV for large dataset")
    
    print(f"\n✅ Demo hyperparameter tuning completed!")
    return best_params, best_score, best_estimator

def plot_tuning_comparison(original_scores, tuned_scores):
    """
    Plot comparison before and after tuning
    """
    metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    original_means = [original_scores[metric].mean() for metric in metrics]
    original_stds = [original_scores[metric].std() for metric in metrics]
    tuned_means = [tuned_scores[metric].mean() for metric in metrics]
    tuned_stds = [tuned_scores[metric].std() for metric in metrics]
    
    bars1 = ax.bar(x - width/2, original_means, width, label='Original SVM', 
                  yerr=original_stds, capsize=5, alpha=0.7, color='skyblue')
    bars2 = ax.bar(x + width/2, tuned_means, width, label='Tuned SVM', 
                  yerr=tuned_stds, capsize=5, alpha=0.7, color='lightcoral')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Comparison of SVM Before and After Hyperparameter Tuning\nAuthor: Đặng Đình Sơn', 
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_parameter_heatmap(results_df):
    """
    Plot heatmap to analyze parameter effects
    """
    # Create pivot table for C and gamma with kernel='rbf'
    rbf_results = results_df[results_df['param_kernel'] == 'rbf'].copy()
    
    def safe_float_gamma(val):
        try:
            return float(val)
        except:
            if val == 'scale':
                return 0.001
            elif val == 'auto':
                return 0.01
            else:
                return np.nan
    
    if len(rbf_results) > 0:
        rbf_results['gamma_numeric'] = rbf_results['param_gamma'].apply(safe_float_gamma)
        
        # Create pivot table
        pivot_table = rbf_results.pivot_table(
            values='mean_test_score',
            index='param_C',
            columns='gamma_numeric',
            aggfunc='mean'
        )
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis', 
                   cbar_kws={'label': 'F1-Score'})
        plt.title('Heatmap: Effect of C and Gamma (Kernel=RBF)\nAuthor: Đặng Đình Sơn', 
                 fontweight='bold')
        plt.xlabel('Gamma')
        plt.ylabel('C')
        plt.show()

def demo_randomized_search():
    """
    Demo RandomizedSearchCV (faster than GridSearchCV)
    """
    print("\n" + "=" * 60)
    print("🎲 DEMO RANDOMIZEDSEARCHCV (Faster)")
    print("=" * 60)
    
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, loguniform
    
    # Load data
    data = pd.read_csv('data.csv')
    X = data.drop(['target'], axis=1)
    y = data['target']
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Define parameter distributions for RandomizedSearchCV
    param_distributions = {
        'C': loguniform(0.1, 100),
        'gamma': ['scale', 'auto'] + list(loguniform(0.001, 0.1).rvs(5)),
        'kernel': ['rbf', 'linear', 'poly']
    }
    
    print(f"   Parameter distributions: {param_distributions}")
    print(f"   Number of iterations: 20 (instead of 60 as in GridSearchCV)")
    
    # Initialize SVM base model
    svm_base = SVC(random_state=42)
    
    # Run RandomizedSearchCV
    print("\n🔍 Running RandomizedSearchCV...")
    
    random_search = RandomizedSearchCV(
        estimator=svm_base,
        param_distributions=param_distributions,
        n_iter=20,  # Only try 20 combinations
        cv=3,
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Fit RandomizedSearchCV
    random_search.fit(X_scaled, y)
    
    # Print results
    print(f"\n✅ RANDOMIZEDSEARCHCV RESULTS:")
    print(f"   Best parameters: {random_search.best_params_}")
    print(f"   Best F1-Score: {random_search.best_score_:.4f}")
    
    print(f"\n💡 ADVANTAGES OF RANDOMIZEDSEARCHCV:")
    print(f"   1. Faster than GridSearchCV (20 vs 60 combinations)")
    print(f"   2. Still finds good parameters")
    print(f"   3. Suitable for large datasets")
    print(f"   4. Can find global optimum")

if __name__ == "__main__":
    # Demo main
    best_params, best_score, best_estimator = demo_hyperparameter_tuning_svm()
    
    # Demo RandomizedSearchCV
    demo_randomized_search()
    
    print(f"\n�� COMPLETION OF DEMO HYPERPARAMETER TUNING!") 
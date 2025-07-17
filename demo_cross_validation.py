#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for Cross-Validation Analysis
Shows how to properly load data and run cross-validation analysis
"""

import pandas as pd
import numpy as np
from cross_validation_analysis import run_cross_validation_analysis

def load_sample_data():
    """
    Load sample data for demonstration
    If data.csv exists, load it. Otherwise create sample data
    """
    try:
        # Try to load existing data
        data = pd.read_csv('data.csv')
        print("✅ Loaded existing data from data.csv")
        print(f"Data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Assume the last column is the target variable
        X = data.iloc[:, :-1]  # All columns except the last
        y = data.iloc[:, -1]   # Last column is the target
        
        print(f"Features (X) shape: {X.shape}")
        print(f"Target (y) shape: {y.shape}")
        print(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
        
    except FileNotFoundError:
        print("⚠️ data.csv not found. Creating sample data for demonstration...")
        
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        # Generate sample features
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i+1}' for i in range(n_features)]
        )
        
        # Generate sample labels (0: control, 1: parkinson)
        y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]))
        
        print(f"✅ Created sample data:")
        print(f"   - Features: {X.shape}")
        print(f"   - Target: {y.shape}")
        print(f"   - Class distribution: {y.value_counts().to_dict()}")
        
        return X, y

def main():
    """
    Main function to demonstrate cross-validation analysis
    """
    print("=" * 60)
    print("🏥 PARKINSON DETECTION - CROSS-VALIDATION DEMO")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n📊 STEP 1: Loading data...")
    X, y = load_sample_data()
    
    # Step 2: Run cross-validation analysis
    print("\n🔬 STEP 2: Running cross-validation analysis...")
    print("This will take a few moments...")
    
    try:
        # Run the cross-validation analysis
        analyzer = run_cross_validation_analysis(X, y, cv_folds=5)
        
        print("\n✅ Cross-validation analysis completed successfully!")
        print("📈 Check the generated plots for detailed results.")
        
    except Exception as e:
        print(f"❌ Error during cross-validation analysis: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure all required libraries are installed:")
        print("   pip install -r requirements.txt")
        print("2. Check if your data is properly formatted")
        print("3. Ensure X and y have the same number of samples")
        print("4. Make sure y contains only 0s and 1s (binary classification)")

if __name__ == "__main__":
    main() 
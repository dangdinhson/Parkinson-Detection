#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix data column names for LightGBM compatibility
"""

import pandas as pd
import numpy as np
import re

def clean_column_names(df):
    """
    Clean column names to be compatible with LightGBM
    """
    # Remove 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        print("✅ Removed 'Unnamed: 0' column")
    
    # Clean column names
    new_columns = []
    for col in df.columns:
        # Replace special characters with underscore
        clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        # Remove multiple underscores
        clean_col = re.sub(r'_+', '_', clean_col)
        # Remove leading/trailing underscores
        clean_col = clean_col.strip('_')
        # Ensure column name is not empty
        if not clean_col:
            clean_col = 'feature_' + str(len(new_columns))
        new_columns.append(clean_col)
    
    df.columns = new_columns
    print("✅ Cleaned column names for LightGBM compatibility")
    return df

def main():
    print("🔧 Fixing data column names...")
    
    try:
        # Load data
        data = pd.read_csv('data.csv')
        print(f"📊 Original data shape: {data.shape}")
        print(f"📋 Original columns: {list(data.columns)}")
        
        # Clean column names
        data_clean = clean_column_names(data.copy())
        
        # Save cleaned data
        data_clean.to_csv('data_clean.csv', index=False)
        print(f"💾 Saved cleaned data to 'data_clean.csv'")
        print(f"📊 Cleaned data shape: {data_clean.shape}")
        print(f"📋 Cleaned columns: {list(data_clean.columns)}")
        
        # Test with cross-validation
        print("\n🧪 Testing with cross-validation...")
        X = data_clean.iloc[:, :-1]
        y = data_clean.iloc[:, -1]
        
        print(f"Features (X) shape: {X.shape}")
        print(f"Target (y) shape: {y.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Import and test
        from cross_validation_analysis import run_cross_validation_analysis
        analyzer = run_cross_validation_analysis(X, y, cv_folds=5)
        
        print("\n✅ Cross-validation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 
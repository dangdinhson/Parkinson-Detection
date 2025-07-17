import warnings
warnings.filterwarnings('ignore')

import os
# os.environ["JOBLIB_START_METHOD"] = "fork"  # Not used on Windows
# os.environ["LOKY_MAX_CPU_COUNT"] = "1"      # Not used on Windows

import logging
logging.getLogger('joblib').setLevel(logging.ERROR)
logging.getLogger('lightgbm').setLevel(logging.ERROR)

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from cross_validation_analysis import run_cross_validation_analysis
import re

# Read data
try:
    data = pd.read_csv('data.csv')
    print(f"✅ Successfully loaded data: {data.shape}")
except FileNotFoundError:
    print("❌ File data.csv not found. Please make sure it exists!")
    exit(1)

# Clean column names to avoid LightGBM errors
def clean_col(col):
    return re.sub(r'[^A-Za-z0-9_]', '_', col)
data.columns = [clean_col(col) for col in data.columns]

# Check target column
if 'target' not in data.columns:
    print("❌ File data.csv does not have 'target' column. Please check your data!")
    exit(1)

X = data.drop(['target'], axis=1)
y = data['target']

# Run cross-validation, plot ROC Curve and calculate AUC for models
analyzer = run_cross_validation_analysis(X, y, cv_folds=5) 
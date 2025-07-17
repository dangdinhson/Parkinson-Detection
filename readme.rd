1. Install Python Environment and Required Libraries
Make sure you have Python installed (Python 3.7+ recommended).
Install Required Libraries
  pip install -r requirements.txt
2. Understand Data Structures
The original data is located in the data/ and hw_dataset/ directories, divided into two groups: control (normal people) and parkinson (Parkinson's patients).
There may be image files in hw_drawings/ for hand-drawn tests.
3. Feature Extraction
Run the Feature_extraction.py or src/Feature_Extraction.py file to extract features from the raw data and create a synthetic data file (possibly data.csv).
 python Feature_extraction.py
4. Model training and analysis
Use files such as:
cross_validation_analysis.py: Cross-validation to evaluate the model.
  python cross_validation_analysis.py
demo_cross_validation.py: Demo of cross-validation process.
  python demo_cross_validation.py
hyperparameter_tuning_demo.py: Tuning hyperparameters for the model.
 python hyperparameter_tuning_demo.py
run_cv_roc.py: Run cross-validation and plot ROC curve
 python run_cv_roc.py
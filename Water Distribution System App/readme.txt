Anomaly Detection in Water Distribution Systems – Streamlit App
================================================================

This Streamlit app enables users to upload water system datasets, preprocess them, 
apply trained machine learning models (SVM, Random Forest, etc.), and visualize 
predictions to identify potential cyber-physical anomalies in water infrastructure.

Features
--------
- File Upload: Upload any `.csv` dataset with numeric features.
- Preprocessing Summary: Automatically handles missing values and scales data using a pre-trained `StandardScaler`.
- Model Selection: Choose from pre-trained ML models like Random Forest or SVM.
- Prediction & Detection: Get predictions for anomalies (e.g., cyberattacks) in the uploaded dataset.
- Visualisations: ROC curve, feature importance bar chart, and anomaly heatmap.
- Results Table: Displays predicted vs actual values with probabilities for each sample.

Requirements
------------
Install all dependencies using:

    pip install -r requirements.txt

Required Python packages include:
- streamlit
- pandas
- scikit-learn
- matplotlib
- joblib

How to Run the App
------------------
From your terminal or Anaconda prompt:

    streamlit run app.py

Then open the app in your browser at http://localhost:8501

Notes
-----
- Make sure your uploaded dataset has the same structure (number and order of features) as the dataset used to train the models.
- If using new data, retraining may be necessary using consistent preprocessing.

Background
----------
This app was developed as part of a research project on cyberattack detection 
in water distribution systems, using machine learning techniques on the BATADAL dataset.


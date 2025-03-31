Breast Cancer Detection Using Support Vector Machine (SVM)
Project Overview
This project aims to predict whether a tumor is malignant or benign based on a dataset of breast cancer measurements. By utilizing a Support Vector Machine (SVM) classifier, we are able to train a machine learning model to classify breast cancer tumor data with high accuracy. The dataset is a collection of various features (radius, texture, perimeter, etc.) extracted from digitized images of breast cancer biopsies.

Key Features:
High Accuracy: Achieved an accuracy of over 96% in predicting whether a tumor is malignant or benign.

Impactful: This tool can potentially be used by healthcare professionals to assist in early cancer detection and improve diagnosis accuracy.

Dataset
The dataset used in this project contains breast cancer features such as radius, texture, perimeter, and others. It provides essential measurements extracted from digital images of breast tissue, which helps determine whether a tumor is malignant (M) or benign (B).

File: The dataset is uploaded by users in CSV format.

Columns: Features include radius_mean, texture_mean, smoothness_mean, etc., which are important indicators for tumor classification.

How It Works
Data Preprocessing:

Handling missing values

Feature extraction and scaling

Model Training:

Using Support Vector Machine (SVM) for classification

Model Evaluation:

Performance is evaluated using accuracy, precision, recall, and confusion matrix

Achieved accuracy: 96.5%

How to Run the Project
Upload Your Dataset: Use any dataset with similar columns to the one used in this project. You will upload the dataset file in CSV format.

Run the Notebook: The code will preprocess the data, train the SVM model, and evaluate its performance.

Get Predictions: Based on the trained model, it will predict the likelihood of the tumor being benign or malignant.

Technologies Used
Python

Scikit-learn

Pandas

Matplotlib

Jupyter Notebooks

Result
The model successfully predicts breast cancer with an accuracy of 96.5%. This classification can help healthcare providers in making quicker and more accurate diagnoses.

Example Output:
Model Accuracy: 96.49%

Classification Report:

Precision, Recall, F1-Score for both classes (Malignant & Benign)

Confusion Matrix: Shows the true positives and false positives

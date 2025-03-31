# Step 1: Install and import necessary libraries
!pip install pandas scikit-learn imbalanced-learn matplotlib seaborn

from google.colab import files
import zipfile
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Upload and extract the ZIP file
uploaded = files.upload()  # Prompt to upload the file

# Extract the ZIP file
zip_file = list(uploaded.keys())[0]  # Get the uploaded file name
extract_dir = 'extracted_data'

# Check if the uploaded file is a zip file
if zip_file.endswith('.zip'):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)  # Extract to a folder named 'extracted_data'
    print(f'Extracted files: {os.listdir(extract_dir)}')
else:
    print('Please upload a ZIP file.')

# List the extracted files
extracted_files = os.listdir(extract_dir)

# Step 3: Load and preprocess the data
# Check for CSV or Excel files
data_file = None
for file in extracted_files:
    if file.endswith('.csv'):
        data_file = os.path.join(extract_dir, file)
        break
    elif file.endswith('.xlsx'):
        data_file = os.path.join(extract_dir, file)
        break

if data_file:
    print(f"Loading data from: {data_file}")
    
    # For CSV files
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    # For Excel files (if uploaded)
    elif data_file.endswith('.xlsx'):
        df = pd.read_excel(data_file)
    
    print(f"First 5 rows of the data:\n{df.head()}")
else:
    print('No valid data file found in the ZIP archive.')

# Step 4: Preprocess the data
# Drop columns that are not needed, like ID or any column with NaN values
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Encode categorical values (e.g., diagnosis column)
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

# Fill missing values
df = df.fillna(df.mean())  # Replace missing values with the mean of the column

X = df.iloc[:, 1:]  # Features: all columns except the first column (diagnosis column)
y = df.iloc[:, 0]   # Target: the diagnosis column

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE (if necessary)
smote = SMOTE()
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train_res)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation results
print(f"Model Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_rep}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Step 9: Visualize the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraudulent'], yticklabels=['Legitimate', 'Fraudulent'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier  # Not used in the provided code snippet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the data
application_data_path: str = "data\\LoanDefaulter\\train\\train.csv"
application_data_df: pd.DataFrame = pd.read_csv(application_data_path)

# Label encode the non-numeric features
label_encoders: dict = {}

for column in application_data_df.columns:
    if application_data_df.dtypes[column] == 'object' and column != 'loan_status':
        le: LabelEncoder = LabelEncoder()
        application_data_df[column] = le.fit_transform(application_data_df[column])
        label_encoders[column] = le

# Split the data into features and target
X: pd.DataFrame = application_data_df.drop('loan_status', axis=1)
y: pd.Series = application_data_df['loan_status']

# Label encode the target variable 'loan_status'
le_y: LabelEncoder = LabelEncoder()
y = le_y.fit_transform(application_data_df['loan_status'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Using DMatrix format for data can improve the efficiency of XGBoost
dtrain: xgb.DMatrix = xgb.DMatrix(X_train, label=y_train)
dtest: xgb.DMatrix = xgb.DMatrix(X_test, label=y_test)

# Define the model parameters
params: dict = {
    'objective': 'multi:softmax',  # Specify the learning task and the corresponding learning objective
    'num_class': 6,  # Set to the number of your classes
    'learning_rate': 0.0001  # Step size shrinkage used to prevent overfitting. Range is [0,1]
}

# Train the model
bst: xgb.Booster = xgb.train(params, dtrain)

# Predictions
y_train_pred_xgb: np.ndarray = bst.predict(dtrain)
y_test_pred_xgb: np.ndarray = bst.predict(dtest)

# Calculate accuracy
train_accuracy_xgb: float = accuracy_score(y_train, y_train_pred_xgb)
test_accuracy_xgb: float = accuracy_score(y_test, y_test_pred_xgb)

# Print the accuracies
print(f"XGBoost Training Accuracy: {train_accuracy_xgb * 100:.2f}%")
print(f"XGBoost Testing Accuracy: {test_accuracy_xgb * 100:.2f}%")

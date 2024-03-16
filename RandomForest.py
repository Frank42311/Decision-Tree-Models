import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from joblib import dump

# Load the dataset
application_data_path = "data\\LoanDefaulter\\train\\train.csv"
application_data_df = pd.read_csv(application_data_path)

# Encode non-numeric features with label encoding
label_encoders = {}

for column in application_data_df.columns:
    if application_data_df.dtypes[column] == 'object' and column != 'loan_status':
        le = LabelEncoder()
        application_data_df[column] = le.fit_transform(application_data_df[column].astype(str))
        label_encoders[column] = le

# Split the dataframe into features (X) and the target variable (y)
X = application_data_df.drop('loan_status', axis=1)
y = application_data_df['loan_status']

# Split data into training and testing sets with a test size of 20% and a random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Initialize the RandomForestClassifier with specific hyperparameters
# Note: The number of features should not exceed 530,000
rf_clf = RandomForestClassifier(
    bootstrap=True, max_samples=0.95, max_features=30, n_estimators=90, 
    min_samples_leaf=3, random_state=43, ccp_alpha=0.00000002
)

# Train the model on the training set
rf_clf.fit(X_train, y_train)

# Save the trained model to a file using joblib
model_path = "results\\RandomForest\\model.joblib"
dump(rf_clf, model_path)

# Make predictions on both the training set and the test set
y_train_pred = rf_clf.predict(X_train)
y_test_pred = rf_clf.predict(X_test)

# Calculate and print the accuracy for both the training set and the test set
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

# Plot the feature importances from the random forest model
importances = rf_clf.feature_importances_
sorted_idx = importances.argsort()

plt.figure(figsize=(10, len(X.columns)))
plt.barh(range(X.shape[1]), importances[sorted_idx], align='center')
plt.yticks(range(X.shape[1]), X.columns[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature Importances of the Random Forest Model')
plt.show()

# Import necessary libraries
from sklearn.ensemble import GradientBoostingClassifier  # Machine learning model
import pandas as pd  # Data manipulation
from sklearn.model_selection import train_test_split  # Split data into train and test sets
from sklearn.metrics import accuracy_score  # Evaluate model performance
from sklearn.preprocessing import LabelEncoder  # Encode labels

# Load the dataset
application_data_path = "data\\LoanDefaulter\\train\\train.csv"
application_data_df = pd.read_csv(application_data_path)  # Read the CSV data into a DataFrame

# Encode non-numeric features to numeric labels
label_encoders = {}  # Dictionary to store label encoders for each column

# Iterate through each column in the DataFrame
for column in application_data_df.columns:
    if application_data_df.dtypes[column] == 'object' and column != 'loan_status':
        le = LabelEncoder()  # Initialize a new label encoder for the column
        # Transform the column values to encoded labels and replace in dataframe
        application_data_df[column] = le.fit_transform(application_data_df[column])
        label_encoders[column] = le  # Store the label encoder for future inverse transformation

# Feature and label separation
X = application_data_df.drop('loan_status', axis=1)  # Features (independent variables)
y = application_data_df['loan_status']  # Target label (dependent variable)

# Splitting the dataset into training and testing sets
# test_size=0.2: 20% of the data will be used for testing, 80% for training
# random_state=43: Ensures reproducibility of the splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Initialize the Gradient Boosting Classifier
# n_estimators=1000: The number of trees in the forest
# learning_rate=0.1: Rate at which the model adapts at each iteration
# max_depth=3: Maximum depth of each tree
# random_state=43: Ensures reproducibility of the model's results
gbt = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1,
                                 max_depth=3, random_state=43)

# Fit the model to the training data
gbt.fit(X_train, y_train)

# Make predictions on both the training and test sets
y_train_pred_gbt = gbt.predict(X_train)  # Predictions on the training set
y_test_pred_gbt = gbt.predict(X_test)  # Predictions on the test set

# Calculate and print the accuracy of the model on both sets
train_accuracy_gbt = accuracy_score(y_train, y_train_pred_gbt)  # Training accuracy
test_accuracy_gbt = accuracy_score(y_test, y_test_pred_gbt)  # Test accuracy

print(f"Gradient Boosting Trees Training Accuracy: {train_accuracy_gbt * 100:.2f}%")
print(f"Gradient Boosting Trees Testing Accuracy: {test_accuracy_gbt * 100:.2f}%")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Reading the application data from the specified path
# This data is expected to contain various features related to loan applications, including a 'loan_status' column that indicates if the loan was defaulted.
application_data_path = "data\\LoanDefaulter\\train\\train.csv"
application_data_df = pd.read_csv(application_data_path)

# Initialize a dictionary to hold LabelEncoders for categorical columns
label_encoders = {}


# Loop through each column in the dataframe
for column in application_data_df.columns:
    # Check if the column is of object type and is not the 'loan_status' column
    if application_data_df.dtypes[column] == 'object' and column != 'loan_status':
        # Initialize a LabelEncoder for the column
        le = LabelEncoder()
        # Fit and transform the column with the LabelEncoder, replacing the original text data with encoded values
        application_data_df[column] = le.fit_transform(application_data_df[column])
        # Store the fitted LabelEncoder in the dictionary for potential future inverse transformations
        label_encoders[column] = le

# Prepare the feature matrix (X) by dropping the 'loan_status' column, which is the target variable
X = application_data_df.drop('loan_status', axis=1)
# Prepare the target vector (y) by selecting only the 'loan_status' column
y = application_data_df['loan_status']

# Split the dataset into training and testing sets, using a test size of 20% and a random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Initialize a DecisionTreeClassifier with specified parameters to control complexity and prevent overfitting or underfitting
# min_samples_leaf: The minimum number of samples required to be at a leaf node.
# ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning.
clf = DecisionTreeClassifier(min_samples_leaf=15, ccp_alpha=0.000008)

# Fit the Decision Tree classifier on the training data
clf.fit(X_train, y_train)

# Save the fitted model to a specified path for later use or deployment
model_path = "results\\DecisionTree\\model.joblib"
dump(clf, model_path)

# Export the trained Decision Tree model to a DOT file for visualization
dot_file_path = "results\\DecisionTree\\tree.dot"
export_graphviz(clf, out_file=dot_file_path, feature_names=X.columns)

# Predict the target values for both training and testing sets
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate and print the accuracy of the model on both training and testing sets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
print(f"Depth of the Decision Tree: {clf.tree_.max_depth}")
print(f"Number of Nodes in the Decision Tree: {clf.tree_.node_count}")
print(f"Number of Leaves in the Decision Tree: {clf.tree_.n_leaves}")

# Visualize the trained Decision Tree using matplotlib
# figsize: Specifies the size of the figure in (width, height) inches.
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns.tolist(), filled=True)
plt.show()

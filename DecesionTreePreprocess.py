import pandas as pd
import numpy as np

# Reading the CSV file
# Define the path to the CSV file containing the loan defaulter data for machine learning projects.
application_data_path = "data\\LoanDefaulter\\train\\train.csv"
# Read the CSV file using pandas, specifying low memory usage to accommodate large files.
df = pd.read_csv(application_data_path, low_memory=False)

# ------------------------------------------------------------------------------------------------------
# Handling employment-related columns
# If both 'emp_title' and 'emp_length' are missing, fill them with 'unemployed' and '0 year' respectively.
df.loc[(df['emp_title'].isnull()) & (df['emp_length'].isnull()), 'emp_title'] = 'unemployed'
df.loc[(df['emp_title'].isnull()) & (df['emp_length'].isnull()), 'emp_length'] = '0 year'

# If 'emp_title' is present but 'emp_length' is missing, fill 'emp_length' with its mode (most frequent value).
emp_length_mode = df['emp_length'].mode().iloc[0]  # Calculate the mode of 'emp_length'.
df.loc[(df['emp_title'].notnull()) & (df['emp_length'].isnull()), 'emp_length'] = emp_length_mode

# If 'emp_length' is present but 'emp_title' is missing, fill 'emp_title' with its mode.
emp_title_mode = df['emp_title'].mode().iloc[0]  # Calculate the mode of 'emp_title'.
df.loc[(df['emp_length'].notnull()) & (df['emp_title'].isnull()), 'emp_title'] = emp_title_mode


# ------------------------------------------------------------------------------------------------------
# Fill remaining columns with median or mode
# Define columns to check for filling missing values with median (for numeric data) or mode (for categorical data).
columns_to_check = [11, 20, 29, 39, 41, 46, 85]

# Loop through each specified column.
for col in columns_to_check:
    col_data = df.iloc[:, col]  # Access the column data.

    # Attempt to convert column to numeric, identifying rows that cannot be converted.
    invalid_rows = pd.to_numeric(col_data, errors='coerce').isna()

    # Keep only the rows where conversion to numeric is successful.
    df = df.loc[~invalid_rows]

# ------------------------------------------------------------------------------------------------------
# Handle columns with diverse values: fill missing with median for numeric and mode for categorical.
for column in df.columns:
    # Check if the column data type is numeric (int64 or float64).
    if df[column].dtype in ['int64', 'float64']:
        # Fill missing numeric values with column's median.
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)
    # For object data types (likely strings), fill missing values with mode.
    elif df[column].dtype == 'object':
        mode_value = df[column].mode().iloc[0]
        df[column].fillna(mode_value, inplace=True)

# ------------------------------------------------------------
# Second round of modifications

# Separate feature data from the target column.
features_data = df.drop(columns=['loan_status'])  # Drop the 'loan_status' column to isolate features.
target_column = df['loan_status']  # Isolate the target column.

# Drop numeric features with low variance (variance below a threshold indicates little variability).
numeric_features = features_data.select_dtypes(include=['number'])  # Select numeric columns.
variance_threshold = 0.1  # Define the variance threshold.
low_variance_features = numeric_features.var()[numeric_features.var() < variance_threshold].index
features_data.drop(columns=low_variance_features, inplace=True)

# Drop features where over 90% of values are the same (indicating low diversity).
threshold_same_value = 0.9  # Define the threshold for value uniformity.
for column in features_data.columns:
    max_count = features_data[column].value_counts(normalize=True).values[0]
    if max_count > threshold_same_value:
        features_data.drop(columns=[column], inplace=True)

# Drop highly correlated numeric features (to reduce multicollinearity).
numeric_features_after_drop = features_data.select_dtypes(include=['number'])
correlation_matrix = numeric_features_after_drop.corr().abs()  # Calculate absolute correlations.
upper_triangle_corr = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle_corr.columns if any(upper_triangle_corr[column] > 0.8)]
features_data.drop(columns=to_drop, inplace=True)

# Concatenate the target column back and save the processed data to a new CSV file.
new_data = pd.concat([features_data, target_column], axis=1)  # Re-include the target column.
new_data.to_csv("data\\LoanDefaulter\\train\\new_train.csv", index=False)  # Save to CSV.


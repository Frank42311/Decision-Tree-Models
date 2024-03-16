import pandas as pd
import numpy as np

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes statistical metrics for a given DataFrame, including the count and ratio of NaN values,
    variance for numeric columns, and data types of each column.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame for which statistical metrics are to be computed.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the computed statistics for each column, including
      column names, data types, NaN counts, NaN ratios, and variance for numeric columns.
    """
    # Count the number of NaN values in each column
    nan_count = df.isnull().sum()
    
    # Calculate the ratio of NaN values to the total number of rows in the DataFrame, expressed as a percentage
    nan_ratio = (nan_count / len(df)) * 100
    
    # Initialize lists to store the variance of numeric columns and data types of all columns
    variance = []
    dtype_list = [] 

    # Loop through each column in the DataFrame
    for col in df.columns:
        dtype_list.append(df[col].dtype) # Append the data type of the current column to the list
        if df[col].dtype == 'object': # For non-numeric (object) columns, append None to the variance list
            variance.append(None)
        else:
            # For numeric columns, calculate the variance, excluding NaN values, and append to the list
            variance.append(df[col].dropna().var())

    # Create and return a new DataFrame with the computed statistics
    return pd.DataFrame({
        'col_name': df.columns, # Column names
        'dtype': dtype_list,    # Data types of each column
        'nan_count': nan_count, # Number of NaN values in each column
        'nan_ratio': nan_ratio, # Ratio of NaN values in each column, as a percentage
        'std': variance,        # Variance of numeric columns (None for non-numeric columns)
    })

# Specify the path to the CSV file containing the application data
application_data_path = "data\\LoanDefaulter\\train\\train.csv"

# Load the application data into a DataFrame
application_data_df = pd.read_csv(application_data_path)

# Display the shape of the DataFrame to get an idea of its size (number of rows and columns)
print(application_data_df.shape)

# Compute the statistical metrics for the application data DataFrame
application_data_stats = compute_stats(application_data_df)

# Rename the columns of the resulting DataFrame for clarity
application_data_stats.columns = ['features', 'dtype', 'nan_count', 'nan_ratio', 'std']

# If needed, save the computed statistics to a CSV file
application_data_stats.to_csv('data\\LoanDefaulter\\preview.csv', index=False)

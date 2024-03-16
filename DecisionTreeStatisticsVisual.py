import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset from the specified path
application_data_path = "data\\LoanDefaulter\\train\\train.csv"
df = pd.read_csv(application_data_path)

# Create a DataFrame to store statistics for string features
heatmap_data_str = pd.DataFrame()

# Process each feature in the dataset
for feature in df.columns:
    # Check if the feature is of string type
    if df[feature].dtype == 'object':
        # Calculate the normalized value counts for the feature
        counts = df[feature].value_counts(normalize=True)
        # Initialize a heatmap row with zeros
        heatmap_row = np.zeros(100)
        start_idx = 0
        # Fill the heatmap row based on the percentage of each value
        for percent in counts:
            end_idx = start_idx + int(percent * 100)
            heatmap_row[start_idx:end_idx] = percent
            start_idx = end_idx
        # Add the filled row to the heatmap DataFrame
        heatmap_data_str[feature] = heatmap_row

# Plot a heatmap for string features using the 'Spectral' color map
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data_str.transpose(), cmap='Spectral')
plt.title('Heatmap for String Features')
plt.yticks(ticks=np.arange(len(heatmap_data_str.columns)), labels=heatmap_data_str.columns, fontsize="10", va="center")
plt.xticks(ticks=np.linspace(0, heatmap_data_str.shape[0], 11), labels=[f'{i/10}' for i in range(0, 11)])
plt.show()


# Select all numerical features from the dataset
numerical_data = df.select_dtypes(include=['number'])

from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Scale the numerical features to a range between 0 and 1
numerical_data_scaled = scaler.fit_transform(numerical_data)

# Create a DataFrame to store statistics for numerical features
heatmap_data_numerical = pd.DataFrame()

# Process each numerical feature
for idx, feature in enumerate(numerical_data.columns):
    # Initialize a count array for 10 bins
    counts = np.zeros(10)
    # Get the scaled values for the current feature
    values = numerical_data_scaled[:, idx]
    # Divide values into bins and count the occurrences in each bin
    for value in values:
        index = int(value * 10)
        index = min(index, 9)  # Ensure the index is within bounds
        counts[index] += 1
    # Add the count array to the heatmap DataFrame
    heatmap_data_numerical[feature] = counts

# Plot a heatmap for numerical features using the 'Greens' color map
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data_numerical.transpose(), cmap='Greens', xticklabels=[f'{i/10}' for i in range(10)])
plt.title('Heatmap for Numerical Features')
plt.yticks(ticks=np.arange(len(heatmap_data_numerical.columns)), labels=heatmap_data_numerical.columns, fontsize="10", va="center")
plt.show()

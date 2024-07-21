import pandas as pd

# Load the dataset
df = pd.read_csv('c:/Users/Aniketh/PycharmProjects/programfolder2/MI_Codes/Hackathon/train.csv')


# Calculate the correlation matrix
correlation_matrix = df.corr()

# Find the most correlated two features
max_corr = correlation_matrix.stack().idxmax()
feature1, feature2 = max_corr

# Check if feature1 and feature2 are the same
if feature1 == feature2:
    # If they are the same, find the next highest correlation
    correlation_matrix = correlation_matrix[correlation_matrix < 1.0]  # Exclude self-correlations
    max_corr = correlation_matrix.stack().idxmax()
    feature1, feature2 = max_corr

print("Most correlated features:")
print(f"{feature1} and {feature2} with a correlation of {correlation_matrix.loc[feature1, feature2]}")

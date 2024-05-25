import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_table("ncl.dat", delim_whitespace=True, header=None, names=["Year", "Claim_Count", "Claim_Amount"])

# Adjust the year by adding 1990
data["Year"] = data["Year"] + 1990

# Divide the third column by 1e+09
data["Claim_Amount"] = data["Claim_Amount"] / 1e+09

# Print the first few rows of the data
print("\nFirst few rows of the data:")
print(data.head())

# Summary statistics
print("\nSummary statistics:")
print(data.describe())

# Line plot
plt.figure(figsize=(10, 6))
plt.plot(data["Year"], data["Claim_Amount"], color="blue", linewidth=2)
plt.xlabel("Years", fontsize=14)
plt.ylabel("Adjusted PCS catastrophe claims (USD billion)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Trend of Adjusted PCS Catastrophe Claims", fontsize=16)
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data["Year"], data["Claim_Amount"], color="green", s=50)
plt.xlabel("Years", fontsize=14)
plt.ylabel("Adjusted PCS catastrophe claims (USD billion)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Scatter Plot of Adjusted PCS Catastrophe Claims", fontsize=16)
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data["Claim_Amount"], kde=True, color="orange")
plt.xlabel("Adjusted PCS catastrophe claims (USD billion)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Distribution of Adjusted PCS Catastrophe Claims", fontsize=16)
plt.show()

# Outlier detection
# 1. Z-score method
print("\nOutliers based on Z-score method:")
z_scores = np.abs((data["Claim_Amount"] - data["Claim_Amount"].mean()) / data["Claim_Amount"].std())
threshold = 3
outliers_z_score = data[z_scores > threshold]
print(outliers_z_score)

# 2. Interquartile Range (IQR) method
print("\nOutliers based on IQR method:")
Q1 = data["Claim_Amount"].quantile(0.25)
Q3 = data["Claim_Amount"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = data[(data["Claim_Amount"] < lower_bound) | (data["Claim_Amount"] > upper_bound)]
print(outliers_iqr)

# 3. Boxplot method
plt.figure(figsize=(10, 6))
plt.boxplot(data["Claim_Amount"])
plt.title("Boxplot of Claim Amounts", fontsize=16)
plt.xlabel("Claim Amounts", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
